"""Dataset reading, source sampling, and deterministic validation helpers."""
from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from pathlib import Path


class GeneDatasetIndex:
    """Reference immutable FASTA bytes; do not manufacture ambiguous bases."""

    def __init__(self, repo_root: Path, dataset_dir: Path):
        self.repo_root = Path(repo_root)
        self.dataset_dir = Path(dataset_dir)
        self.genomes = sqlite3.connect(f"file:{self.dataset_dir / 'genomes.sqlite'}?mode=ro", uri=True)
        self.genomes.row_factory = sqlite3.Row
        self.fasta = (self.repo_root / 'data/train/training_set.fasta').open('rb')
        self.targets = {}

    def read(self, target_db: str, target_id: str) -> dict:
        relative = Path(target_db)
        if relative.is_absolute() or '..' in relative.parts or relative.parent != Path('targets'):
            raise ValueError('Target database must be directly inside targets/')
        if target_db not in self.targets:
            connection = sqlite3.connect(f"file:{self.dataset_dir / relative}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            self.targets[target_db] = connection
        row = self.targets[target_db].execute('SELECT * FROM targets WHERE target_id=?', (target_id,)).fetchone()
        if row is None:
            raise KeyError(target_id)
        target = dict(row)
        genome = self.genomes.execute('SELECT * FROM genomes WHERE genome_idx=?', (target['genome_idx'],)).fetchone()
        if genome is None or not genome['eligible'] or genome['split'] != target['split'] or genome['genome_id'] != target['genome_id']:
            raise ValueError('Target/source eligibility mismatch')
        self.fasta.seek(genome['seq_offset'])
        sequence = b''.join(self.fasta.read(genome['seq_end'] - genome['seq_offset']).split()).upper()
        if len(sequence) != genome['length_bp'] or hashlib.sha256(sequence).hexdigest() != genome['forward_sha256']:
            raise ValueError('Source FASTA changed after dataset construction')
        start, end = target['target_start'], target['target_end']
        if not (0 <= target['gene_start'] <= start < end <= target['gene_end'] <= len(sequence)) or end - start > 3000:
            raise ValueError('Invalid target coordinates')
        if set(sequence) - set(b'ACGT'):
            raise ValueError('Ambiguous source context')
        return {'sequence': sequence.decode('ascii'), 'target': target, 'source': dict(genome)}

    def close(self):
        self.fasta.close()
        self.genomes.close()
        for connection in self.targets.values():
            connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def read_panel(path: Path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


class SourceSampler:
    """Shuffle source records and sample one eligible target per visit."""

    def __init__(self, directory, seed=42, excluded=None):
        self.rng = random.Random(seed)
        self.records, self.connections = [], {}
        excluded = set(excluded or [])
        for path in sorted((Path(directory) / 'targets').glob('shard_*.sqlite')):
            connection = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
            relative = 'targets/' + path.name
            self.connections[relative] = connection
            for idx, genome_id, count in connection.execute(
                    "SELECT genome_idx,genome_id,COUNT(*) FROM targets WHERE split='train' GROUP BY genome_idx"):
                if genome_id not in excluded:
                    self.records.append((relative, idx, genome_id, count))
        if not self.records:
            raise ValueError('No eligible training source genomes')
        self.records.sort()
        self.fingerprint = hashlib.sha256(json.dumps(self.records).encode()).hexdigest()
        self.order = list(range(len(self.records)))
        self.rng.shuffle(self.order)
        self.cursor, self.epoch = 0, 0

    def next(self):
        if self.cursor == len(self.order):
            self.epoch += 1
            self.cursor = 0
            self.rng.shuffle(self.order)
        database, idx, genome_id, count = self.records[self.order[self.cursor]]
        self.cursor += 1
        offset = self.rng.randrange(count)
        row = self.connections[database].execute(
            'SELECT target_id FROM targets WHERE genome_idx=? ORDER BY rowid LIMIT 1 OFFSET ?',
            (idx, offset)).fetchone()
        if row is None:
            raise ValueError('Offline target table changed')
        return database, row[0]

    def state_dict(self):
        return dict(fingerprint=self.fingerprint, order=list(self.order), cursor=self.cursor,
                    epoch=self.epoch, rng=self.rng.getstate())

    def load_state_dict(self, state):
        if state['fingerprint'] != self.fingerprint:
            raise ValueError('Resume dataset/sampling population changed')
        if sorted(state['order']) != list(range(len(self.records))):
            raise ValueError('Invalid resume source order')
        self.order, self.cursor, self.epoch = list(state['order']), state['cursor'], state['epoch']
        self.rng.setstate(state['rng'])


def evaluation_rng(seed, target_id, rate):
    digest = hashlib.sha256(f'{seed}/{target_id}/{rate}'.encode()).digest()
    return random.Random(int.from_bytes(digest[:8], 'big'))


def window_bounds(length, start, end, radius):
    """Crop around a target; radius None keeps the whole source."""
    if not 0 <= start < end <= length:
        raise ValueError('Primary target outside the source')
    if radius is None:
        return 0, length
    center = (start + end) // 2
    left, right = max(0, center - radius), min(length, center + radius)
    return min(left, start), max(right, end)
