"""Build strictly paired, translation-validated CDS restoration manifests."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import zip_longest
from pathlib import Path

from Bio import SeqIO
from Bio.Data import CodonTable
from Bio.Seq import Seq


class AnnotationIntegrityError(ValueError):
    """A source pairing or coordinate failure that must stop construction."""


def _atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def _file_identity(path, relative_to=None):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    display = path.relative_to(relative_to).as_posix() if relative_to is not None else path.name
    return {"path": display, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _hist_stats(hist):
    hist = {int(k): int(v) for k, v in hist.items()}
    count = sum(hist.values())
    if not count:
        return {'count': 0, 'total_bp': 0, 'histogram': {}}
    ranks = [(count - 1) // 2, count // 2]
    values, cumulative = [], 0
    for length, number in sorted(hist.items()):
        for rank in ranks:
            if cumulative <= rank < cumulative + number:
                values.append(length)
        cumulative += number
    total = sum(length * number for length, number in hist.items())
    return {'count': count, 'total_bp': total, 'min': min(hist), 'max': max(hist),
            'mean': total / count, 'median': sum(values) / 2,
            'histogram': dict(sorted(hist.items()))}


def _parse_header(record):
    fields = record.description.split(' # ')
    if len(fields) != 5 or fields[0] != record.id:
        raise AnnotationIntegrityError(f'Malformed FAA header: {record.id}')
    attributes = dict(item.split('=', 1) for item in fields[4].split(';') if '=' in item)
    if attributes.get('partial') not in {'00', '01', '10', '11'}:
        raise AnnotationIntegrityError(f'Missing/invalid partial flag: {record.id}')
    return (int(fields[1]), int(fields[2]), int(fields[3]),
            int(attributes['genetic_code']), attributes)


def _validate_translation(dna, protein, code):
    """Return one exclusion reason; normalize alternative initiators to Met."""
    if code not in {4, 11, 15}:
        return 'unsupported_genetic_code'
    if len(dna) % 3:
        return 'cds_length_not_multiple_of_three'
    if set(dna) - set('ACGT'):
        return 'ambiguous_target_sequence'
    table = CodonTable.unambiguous_dna_by_id[code]
    if dna[:3] not in table.start_codons:
        return 'invalid_start_codon'
    if dna[-3:] not in table.stop_codons:
        return 'invalid_terminal_stop_codon'
    translated = str(Seq(dna).translate(table=code, to_stop=False))
    if '*' in translated[:-1]:
        return 'internal_stop_codon'
    translated = 'M' + translated[1:]
    if not protein.endswith('*'):
        return 'faa_missing_terminal_stop'
    if translated != protein:
        return 'translation_faa_mismatch'
    return None


def build_one_shard(repo_root: Path, output_dir: Path, shard: str,
                    seed: int = 20261103) -> dict:
    """Build one shard atomically; structural source errors are fatal."""
    repo_root, output_dir = Path(repo_root), Path(output_dir)
    annotation = repo_root / 'data/annotations/training_genomad' / shard / f'{shard}_annotate'
    tsv = annotation / f'{shard}_genes.tsv'
    faa = annotation / f'{shard}_proteins.faa'
    destination = output_dir / 'targets'
    destination.mkdir(parents=True, exist_ok=True)
    database = destination / f'{shard}.sqlite'
    staging = database.with_name(database.name + f'.tmp.{os.getpid()}')
    report_path = destination / f'{shard}.audit.json'
    inventory = sqlite3.connect(f'file:{output_dir / "genomes.sqlite"}?mode=ro', uri=True)
    inventory.row_factory = sqlite3.Row
    genomes = {row['genome_id']: dict(row) for row in inventory.execute('SELECT * FROM genomes')}
    inventory.close()
    report = {'shard': shard, 'status': 'running', 'seed': seed,
              'sources': {'tsv': tsv.relative_to(repo_root).as_posix(),
                          'faa': faa.relative_to(repo_root).as_posix()},
              'coordinate_system': 'zero_based_half_open',
              'protein_hash_convention': 'FAA amino acids without the terminal stop',
              'long_gene_window': 'one deterministic codon-aligned 3000 bp window',
              'counts': Counter(), 'excluded': Counter(), 'exclusion_examples': {},
              'genetic_codes': Counter(), 'partial_flags': Counter()}
    histograms = defaultdict(Counter)
    annotated_genomes, targeted_genomes = {}, {}
    seen_genes = set()
    connection = None
    try:
        report['inventory_sha256'] = _file_identity(output_dir / 'genomes.sqlite', output_dir)['sha256']
        report['sources'] = {'tsv': _file_identity(tsv, repo_root), 'faa': _file_identity(faa, repo_root)}
        if staging.exists():
            raise AnnotationIntegrityError(f'Staging database already exists: {staging}')
        connection = sqlite3.connect(staging)
        connection.execute('''CREATE TABLE targets (
          target_id TEXT PRIMARY KEY, genome_idx INTEGER NOT NULL, genome_id TEXT NOT NULL,
          gene_id TEXT NOT NULL UNIQUE, split TEXT NOT NULL, gene_start INTEGER NOT NULL,
          gene_end INTEGER NOT NULL, strand INTEGER NOT NULL, genetic_code INTEGER NOT NULL,
          target_start INTEGER NOT NULL, target_end INTEGER NOT NULL, target_kind TEXT NOT NULL,
          protein_sha256 TEXT NOT NULL, gene_length INTEGER NOT NULL, target_length INTEGER NOT NULL,
          genome_length INTEGER NOT NULL, source TEXT NOT NULL)''')
        fasta = repo_root / 'data/train/training_set.fasta'
        current_genome, current_sequence = None, None
        with tsv.open(newline='') as tab, faa.open() as proteins, fasta.open('rb') as sequences:
            rows = csv.DictReader(tab, delimiter='\t')
            required = {'gene', 'start', 'end', 'length', 'strand', 'genetic_code'}
            if not required.issubset(rows.fieldnames or []):
                raise AnnotationIntegrityError(f'Missing TSV columns: {tsv}')
            for row, protein_record in zip_longest(rows, SeqIO.parse(proteins, 'fasta')):
                if row is None or protein_record is None:
                    raise AnnotationIntegrityError(f'TSV/FAA record count mismatch: {shard}')
                gene = row['gene']
                if gene != protein_record.id:
                    raise AnnotationIntegrityError(f'TSV/FAA order or ID mismatch: {gene} vs {protein_record.id}')
                if gene in seen_genes:
                    raise AnnotationIntegrityError(f'Duplicate gene ID: {gene}')
                seen_genes.add(gene)
                start, end, strand, code, attrs = _parse_header(protein_record)
                pair = tuple(int(row[key]) for key in ('start', 'end', 'strand', 'genetic_code'))
                if pair != (start, end, strand, code) or int(row['length']) != end - start + 1:
                    raise AnnotationIntegrityError(f'TSV/FAA coordinate/code mismatch: {gene}')
                genome_id = gene.rsplit('_', 1)[0]
                if genome_id not in genomes:
                    raise AnnotationIntegrityError(f'Uninventoried annotation source: {genome_id}')
                genome = genomes[genome_id]
                if not (1 <= start <= end <= genome['length_bp']) or strand not in {-1, 1}:
                    raise AnnotationIntegrityError(f'Invalid coordinate/strand: {gene}')
                annotated_genomes[genome['genome_idx']] = genome_id
                counts = report['counts']
                counts['source_annotations'] += 1
                report['partial_flags'][attrs['partial']] += 1
                report['genetic_codes'][code] += 1
                reason = None
                if not genome['eligible']:
                    reason = 'ineligible_genome:' + str(genome['exclusion'])
                elif attrs['partial'] != '00':
                    reason = 'partial_cds:' + attrs['partial']
                else:
                    counts['complete_eligible_candidates'] += 1
                    if current_genome != genome_id:
                        sequences.seek(genome['seq_offset'])
                        raw = sequences.read(genome['seq_end'] - genome['seq_offset'])
                        current_sequence = b''.join(raw.split()).upper()
                        if len(current_sequence) != genome['length_bp'] or hashlib.sha256(current_sequence).hexdigest() != genome['forward_sha256']:
                            raise AnnotationIntegrityError(f'FASTA offset/content identity mismatch: {genome_id}')
                        current_genome = genome_id
                    dna = current_sequence[start - 1:end].decode('ascii')
                    if strand == -1:
                        dna = str(Seq(dna).reverse_complement())
                    protein = str(protein_record.seq)
                    reason = _validate_translation(dna, protein, code)
                    if reason:
                        counts['complete_eligible_validation_failed'] += 1
                if reason:
                    report['excluded'][reason] += 1
                    examples = report['exclusion_examples'].setdefault(reason, [])
                    if len(examples) < 10:
                        examples.append(gene)
                    continue
                gene_start, gene_end = start - 1, end
                length = gene_end - gene_start
                target_start, target_end, kind = gene_start, gene_end, 'whole_cds'
                if length > 3000:
                    key = f'{seed}\0{genome_id}\0{gene}\0{gene_start}\0{gene_end}'.encode()
                    offset = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % ((length - 3000) // 3 + 1) * 3
                    target_start = gene_start + offset
                    target_end, kind = target_start + 3000, 'partial_cds_window'
                target_length = target_end - target_start
                target_id = hashlib.sha256(f'{gene}\0{target_start}\0{target_end}'.encode()).hexdigest()
                connection.execute('INSERT INTO targets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (target_id, genome['genome_idx'], genome_id, gene, genome['split'], gene_start,
                     gene_end, strand, code, target_start, target_end, kind,
                     hashlib.sha256(protein[:-1].encode()).hexdigest(), length, target_length,
                     genome['length_bp'], genome['source']))
                counts['accepted_targets'] += 1
                targeted_genomes[genome['genome_idx']] = {'genome_id': genome_id,
                    'split': genome['split'], 'length_bp': genome['length_bp']}
                histograms[f'{genome["split"]}/{kind}'][target_length] += 1
                histograms[f'{genome["split"]}/all'][target_length] += 1
        if report['counts']['source_annotations'] != report['counts']['accepted_targets'] + sum(report['excluded'].values()):
            raise AnnotationIntegrityError(f'Annotation accounting mismatch: {shard}')
        if report['counts']['complete_eligible_candidates'] != report['counts']['accepted_targets'] + report['counts']['complete_eligible_validation_failed']:
            raise AnnotationIntegrityError(f'Complete-CDS accounting mismatch: {shard}')
        connection.execute('CREATE INDEX targets_genome_idx ON targets(genome_idx)')
        connection.execute('CREATE INDEX targets_split ON targets(split)')
        connection.execute('CREATE INDEX targets_split_genome ON targets(split,genome_idx)')
        connection.commit()
        if connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise AnnotationIntegrityError(f'SQLite quick_check failed: {shard}')
        connection.close()
        connection = None
        os.replace(staging, database)
        report['target_database_sha256'] = _file_identity(database)['sha256']
        report['status'] = 'complete'
        report['length_stats'] = {key: _hist_stats(hist) for key, hist in sorted(histograms.items())}
        report['annotated_genomes'] = annotated_genomes
        report['targeted_genomes'] = targeted_genomes
        report['targeted_genome_stats'] = _genome_stats(targeted_genomes)
        _atomic_json(report_path, report)
        print(f'{shard}: accepted={report["counts"]["accepted_targets"]}, excluded={sum(report["excluded"].values())}', flush=True)
        return report
    except Exception as error:
        if connection is not None:
            connection.close()
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        report['staging_database'] = str(staging)
        _atomic_json(report_path, report)
        raise


def _genome_stats(genomes):
    result = {}
    for record in genomes.values():
        bucket = result.setdefault(record['split'], {'genomes': 0, 'total_context_bp': 0})
        bucket['genomes'] += 1
        bucket['total_context_bp'] += record['length_bp']
    return result


def build_annotations(repo_root: Path, output_dir: Path, seed: int = 20261103,
                      workers: int = 4) -> dict:
    """Validate all sixteen shards with at most four CPU workers."""
    repo_root, output_dir = Path(repo_root), Path(output_dir)
    if not 1 <= workers <= 4:
        raise ValueError('workers must be between 1 and 4')
    shards = [f'shard_{index:04d}' for index in range(16)]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(build_one_shard, repo_root, output_dir, shard, seed): shard for shard in shards}
        for future in as_completed(futures):
            results.append(future.result())
    inventory_sha256 = _file_identity(output_dir / 'genomes.sqlite')['sha256']
    if any(result['inventory_sha256'] != inventory_sha256 for result in results):
        raise AnnotationIntegrityError('Inventory changed during annotation build')
    counts, exclusions, codes, partials = Counter(), Counter(), Counter(), Counter()
    histograms = defaultdict(Counter)
    annotated, targeted = {}, {}
    for result in sorted(results, key=lambda item: item['shard']):
        for index, name in result['annotated_genomes'].items():
            if index in annotated:
                raise AnnotationIntegrityError(f'Genome occurs in multiple annotation shards: {name}')
            annotated[index] = name
        targeted.update(result['targeted_genomes'])
        counts.update(result['counts'])
        exclusions.update(result['excluded'])
        codes.update(result['genetic_codes'])
        partials.update(result['partial_flags'])
        for key, stats in result['length_stats'].items():
            histograms[key].update({int(length): number for length, number in stats['histogram'].items()})
    with sqlite3.connect(f'file:{output_dir / "genomes.sqlite"}?mode=ro', uri=True) as inventory:
        expected = dict(inventory.execute('SELECT genome_idx,genome_id FROM genomes'))
    if annotated != expected:
        missing = len(expected.keys() - annotated.keys())
        extra = len(annotated.keys() - expected.keys())
        raise AnnotationIntegrityError(f'Annotation/inventory coverage mismatch: missing={missing}, extra={extra}')
    if counts['source_annotations'] != counts['accepted_targets'] + sum(exclusions.values()):
        raise AnnotationIntegrityError('Global annotation accounting mismatch')
    if counts['complete_eligible_candidates'] != counts['accepted_targets'] + counts['complete_eligible_validation_failed']:
        raise AnnotationIntegrityError('Global complete-CDS accounting mismatch')
    report = {'status': 'complete', 'seed': seed, 'workers': workers,
              'inventory_sha256': inventory_sha256,
              'coordinate_system': 'zero_based_half_open', 'counts': counts,
              'excluded': exclusions, 'genetic_codes': codes, 'partial_flags': partials,
              'annotated_genomes': len(annotated), 'targeted_genome_stats': _genome_stats(targeted),
              'length_stats': {key: _hist_stats(hist) for key, hist in sorted(histograms.items())},
              'shard_audits': [f'targets/{shard}.audit.json' for shard in shards],
              'source_pairing': 'Exact ordered TSV/FAA ID, coordinate, strand, code and length validation',
              'translation_validation': 'Complete CDS; table 4/11/15; valid initiator and stop; no internal stop; FAA exact after initiator Met normalization',
              'long_gene_policy': 'One deterministic codon-aligned 3000 bp window; full CDS translation validated before clipping'}
    _atomic_json(output_dir / 'ANNOTATION_REPORT.json', report)
    return report
