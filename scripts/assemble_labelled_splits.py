"""Assemble an exact train/test split from trusted labelled molecule pools."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from rdkit import Chem

from _workflow.data import (atom_metadata, digest, load_labels, new_output,
                            structures, write_charges, write_json)


def load_pool(name, sdf_root, charges_path):
    records = structures(sdf_root)
    labels, verified = load_labels(charges_path, records)
    if not verified:
        raise ValueError(f'{name} labels must include a source manifest')
    manifest_path = Path(charges_path).with_name('manifest.json')
    manifest = json.loads(manifest_path.read_text())
    molecules = manifest.get('molecules', {})
    pool = []
    for identifier, source, molecule in records:
        metadata = molecules.get(identifier)
        if metadata is None:
            raise ValueError(f'{name} manifest is missing {identifier}')
        pool.append({
            'identifier': identifier,
            'source': source,
            'molecule': molecule,
            'charges': labels[identifier],
            'metadata': metadata,
            'pool': name,
            'connectivity': Chem.MolToSmiles(Chem.RemoveHs(molecule), isomericSmiles=False),
            'family': identifier[:6],
        })
    return pool


def validate_disjoint(pools):
    seen_identifiers, seen_families, seen_connectivity = set(), set(), set()
    for pool in pools:
        for item in pool:
            identifier = item['identifier']
            if identifier in seen_identifiers:
                raise ValueError(f'Duplicate identifier across labelled pools: {identifier}')
            if item['family'] in seen_families:
                raise ValueError(f'Duplicate refcode family across labelled pools: {item["family"]}')
            if item['connectivity'] in seen_connectivity:
                raise ValueError(f'Duplicate connectivity across labelled pools: {identifier}')
            seen_identifiers.add(identifier)
            seen_families.add(item['family'])
            seen_connectivity.add(item['connectivity'])


def write_split(output, split, items, source_charges):
    sdf_root = output / split / 'sdf'
    charge_root = output / split / 'charges'
    sdf_root.mkdir(parents=True)
    charge_root.mkdir()
    arrays, metadata, rows = {}, {}, []
    for item in items:
        identifier = item['identifier']
        destination = sdf_root / item['source'].name
        shutil.copyfile(item['source'], destination)
        if digest(destination) != digest(item['source']):
            raise ValueError(f'Source copy changed bytes: {identifier}')
        arrays[identifier] = item['charges']
        metadata[identifier] = {
            **item['metadata'],
            'source_pool': item['pool'],
            'assembled_split': split,
            'assembled_source_sha256': digest(destination),
        }
        rows.append({
            'CSD_identifier': identifier,
            'smiles': Chem.MolToSmiles(item['molecule']),
            'charges': json.dumps(item['charges'].tolist()),
            'atom_indices': json.dumps(atom_metadata(item['molecule'])['atom_indices']),
        })
    write_charges(charge_root, rows, arrays, metadata, 'psiresp', 'all_atom')
    manifest_path = charge_root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update({
        'format': 'ccdc-trusted-labelled-dataset-v1',
        'split': split,
        'source_charge_archives': [str(Path(path).resolve()) for path in source_charges],
        'molecules': metadata,
    })
    write_json(manifest_path, manifest)
    checked_records = structures(sdf_root)
    checked_labels, verified = load_labels(charge_root / 'charges.npz', checked_records)
    if not verified or set(checked_labels) != set(arrays):
        raise ValueError(f'Assembled split provenance check failed: {split}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-sdf', type=Path, required=True)
    parser.add_argument('--base-charges', type=Path, required=True)
    parser.add_argument('--additional-sdf', type=Path, required=True)
    parser.add_argument('--additional-charges', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-count', type=int, default=900)
    parser.add_argument('--test-count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if min(args.train_count, args.test_count) < 1:
        parser.error('Train and test counts must be positive')

    base = load_pool('base', args.base_sdf, args.base_charges)
    additional = load_pool('additional', args.additional_sdf, args.additional_charges)
    validate_disjoint([base, additional])
    if len(base) > args.train_count:
        parser.error(f'Base pool has {len(base)} molecules, more than train-count {args.train_count}')
    needed = args.train_count - len(base) + args.test_count
    if len(additional) < needed:
        parser.error(f'Need {needed} additional molecules, found {len(additional)}')

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(additional))
    train_extra_count = args.train_count - len(base)
    train_extra = [additional[i] for i in order[:train_extra_count]]
    test_items = [additional[i] for i in order[train_extra_count:needed]]
    train_items = base + train_extra
    train_items = [train_items[i] for i in rng.permutation(len(train_items))]
    output = new_output(args.output)
    write_split(output, 'train', train_items, [args.base_charges, args.additional_charges])
    write_split(output, 'test', test_items, [args.additional_charges])

    train_ids = [item['identifier'] for item in train_items]
    test_ids = [item['identifier'] for item in test_items]
    train_identity = {item['connectivity'] for item in train_items}
    test_identity = {item['connectivity'] for item in test_items}
    if train_identity & test_identity:
        raise ValueError('Assembled train/test connectivity overlap')
    train_families = {item['family'] for item in train_items}
    test_families = {item['family'] for item in test_items}
    if train_families & test_families:
        raise ValueError('Assembled train/test refcode-family overlap')
    write_json(output / 'split.json', {
        'format': 'ccdc-trusted-labelled-split-v1',
        'seed': args.seed,
        'train_count': len(train_ids),
        'test_count': len(test_ids),
        'base_count': len(base),
        'additional_count': len(additional),
        'additional_used': needed,
        'train_ids': train_ids,
        'test_ids': test_ids,
        'connectivity_disjoint': True,
        'refcode_families_disjoint': True,
        'base_sdf': str(args.base_sdf.resolve()),
        'base_charges': str(args.base_charges.resolve()),
        'additional_sdf': str(args.additional_sdf.resolve()),
        'additional_charges': str(args.additional_charges.resolve()),
        'base_charges_sha256': digest(args.base_charges),
        'additional_charges_sha256': digest(args.additional_charges),
    })
    print(f'Created {len(train_ids)} train and {len(test_ids)} test molecules: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
