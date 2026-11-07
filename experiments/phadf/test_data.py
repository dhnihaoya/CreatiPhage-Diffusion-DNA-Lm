"""Test reference identity and full-context restoration by the index reader."""
import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path

from .data import GeneDatasetIndex
from .inventory import build_inventory


class ReaderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='phadf_reader_')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / 'data/train').mkdir(parents=True)
        (self.root / 'data/split').mkdir(parents=True)
        self.fasta = self.root / 'data/train/training_set.fasta'
        self.fasta.write_text('>natural\nACGT\nTTAC\n')
        with (self.root / 'data/split/manifest.csv').open('w') as handle:
            writer = csv.writer(handle)
            writer.writerow(['id', 'source', 'split', 'length_bp'])
            writer.writerow(['natural', 'synthetic', 'train', 8])
        self.output = self.root / 'output'
        build_inventory(self.root, self.output)
        (self.output / 'targets').mkdir()
        self.db = self.output / 'targets/shard_0000.sqlite'
        with sqlite3.connect(self.db) as connection:
            connection.execute('''CREATE TABLE targets(target_id TEXT PRIMARY KEY,genome_idx INTEGER,
                genome_id TEXT,split TEXT,gene_start INTEGER,gene_end INTEGER,
                target_start INTEGER,target_end INTEGER,strand INTEGER)''')
            connection.execute("INSERT INTO targets VALUES('example',0,'natural','train',2,6,2,6,-1)")

    def test_full_context_and_negative_strand_keep_original_orientation(self):
        with GeneDatasetIndex(self.root, self.output) as reader:
            example = reader.read('targets/shard_0000.sqlite', 'example')
        self.assertEqual(example['sequence'], 'ACGTTTAC')
        self.assertEqual(example['sequence'][2:6], 'GTTT')
        self.assertEqual(example['target']['strand'], -1)

    def test_changed_reference_is_rejected(self):
        self.fasta.write_text('>natural\nACGT\nTTAA\n')
        with GeneDatasetIndex(self.root, self.output) as reader:
            with self.assertRaisesRegex(ValueError, 'changed'):
                reader.read('targets/shard_0000.sqlite', 'example')

    def test_invalid_target_is_rejected(self):
        with sqlite3.connect(self.db) as connection:
            connection.execute('UPDATE targets SET target_end=9')
        with GeneDatasetIndex(self.root, self.output) as reader:
            with self.assertRaisesRegex(ValueError, 'coordinates'):
                reader.read('targets/shard_0000.sqlite', 'example')

    def test_wrong_split_is_rejected(self):
        with sqlite3.connect(self.db) as connection:
            connection.execute("UPDATE targets SET split='test'")
        with GeneDatasetIndex(self.root, self.output) as reader:
            with self.assertRaisesRegex(ValueError, 'eligibility'):
                reader.read('targets/shard_0000.sqlite', 'example')

    def test_target_database_path_is_bounded(self):
        with GeneDatasetIndex(self.root, self.output) as reader:
            with self.assertRaisesRegex(ValueError, 'inside targets'):
                reader.read('../elsewhere.sqlite', 'example')


if __name__ == '__main__':
    unittest.main()
