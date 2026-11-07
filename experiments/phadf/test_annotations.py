"""Small CPU fixtures for annotation pairing and translation validation."""
import csv
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from Bio.Seq import Seq

from experiments.phadf.annotations import (
    AnnotationIntegrityError, _hist_stats, _validate_translation, build_one_shard,
)


class AnnotationTests(unittest.TestCase):
    def test_translation_codes_and_initiators(self):
        for dna, protein, code in [('ATGAAATAA', 'MK*', 11), ('GTGAAATAA', 'MK*', 11),
                                   ('ATGTGATAA', 'MW*', 4), ('ATGTAGTAA', 'MQ*', 15)]:
            self.assertIsNone(_validate_translation(dna, protein, code))
        self.assertEqual(_validate_translation('ATGTAATAA', 'M**', 11), 'internal_stop_codon')
        self.assertEqual(_validate_translation('ATGNNNTAA', 'MX*', 11), 'ambiguous_target_sequence')
        self.assertEqual(_hist_stats({3: 2, 6: 2})['median'], 4.5)

    def test_shard_acceptance_and_structural_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, out = root / 'repo', root / 'out'
            out.mkdir()
            (repo / 'data/train').mkdir(parents=True)
            annotation = repo / 'data/annotations/training_genomad/shard_0000/shard_0000_annotate'
            annotation.mkdir(parents=True)
            examples = [
                ('g1', 'ATGAAATAA', 'MK*', 1, 11, '00'),
                ('g2', str(Seq('ATGTGATAA').reverse_complement()), 'MW*', -1, 4, '00'),
                ('g3', 'ATGTAGTAA', 'MQ*', 1, 15, '00'),
                ('g4', 'ATG' + 'AAA' * 1100 + 'TAA', 'M' + 'K' * 1100 + '*', 1, 11, '00'),
                ('g5', 'ATGAAATAA', 'MK*', 1, 11, '01'),
                ('g6', 'ATGAAATAA', 'MA*', 1, 11, '00'),
            ]
            database = sqlite3.connect(out / 'genomes.sqlite')
            database.execute('''CREATE TABLE genomes(genome_idx INTEGER,genome_id TEXT,
                length_bp INTEGER,seq_offset INTEGER,seq_end INTEGER,forward_sha256 TEXT,
                eligible INTEGER,exclusion TEXT,split TEXT,source TEXT)''')
            tsv = annotation / 'shard_0000_genes.tsv'
            with (repo / 'data/train/training_set.fasta').open('wb') as fasta, tsv.open('w') as tab, \
                 (annotation / 'shard_0000_proteins.faa').open('w') as faa:
                writer = csv.writer(tab, delimiter='\t')
                writer.writerow(['gene', 'start', 'end', 'length', 'strand', 'genetic_code'])
                for index, (name, dna, protein, strand, code, partial) in enumerate(examples):
                    fasta.write(f'>{name}\n'.encode())
                    start = fasta.tell()
                    fasta.write((dna + '\n').encode())
                    end = fasta.tell()
                    database.execute('INSERT INTO genomes VALUES(?,?,?,?,?,?,?,?,?,?)',
                        (index, name, len(dna), start, end, hashlib.sha256(dna.encode()).hexdigest(),
                         1, '', 'train', 'fixture'))
                    writer.writerow([name + '_1', 1, len(dna), len(dna), strand, code])
                    faa.write(f'>{name}_1 # 1 # {len(dna)} # {strand} # partial={partial};genetic_code={code}\n{protein}\n')
            database.commit()
            database.close()
            report = build_one_shard(repo, out, 'shard_0000')
            self.assertEqual(report['counts']['accepted_targets'], 4)
            self.assertEqual(report['excluded'], {'partial_cds:01': 1, 'translation_faa_mismatch': 1})
            self.assertEqual(report['counts']['complete_eligible_candidates'], 5)
            self.assertEqual(report['counts']['complete_eligible_validation_failed'], 1)
            self.assertEqual(len(report['inventory_sha256']), 64)
            final_database = out / 'targets/shard_0000.sqlite'
            self.assertEqual(report['target_database_sha256'], hashlib.sha256(final_database.read_bytes()).hexdigest())
            with sqlite3.connect(final_database) as database:
                start, end, kind = database.execute("SELECT target_start,target_end,target_kind FROM targets WHERE gene_id='g4_1'").fetchone()
                self.assertEqual(end - start, 3000)
                self.assertEqual(start % 3, 0)
                self.assertEqual(kind, 'partial_cds_window')
            tsv.write_text(tsv.read_text().replace('g1_1', 'wrong_1', 1))
            with self.assertRaises(AnnotationIntegrityError):
                build_one_shard(repo, out, 'shard_0000')
            failed = json.loads((out / 'targets/shard_0000.audit.json').read_text())
            self.assertEqual(failed['status'], 'failed')
            self.assertEqual(hashlib.sha256(final_database.read_bytes()).hexdigest(), report['target_database_sha256'])


if __name__ == '__main__':
    unittest.main()
