"""Create a deterministic, provenance-preserving train/test split of labelled SDFs."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from rdkit import Chem

from _workflow.data import (atom_metadata, digest, load_labels, new_output,
                            structures, write_charges, write_json)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdf', type=Path, required=True)
    parser.add_argument('--charges', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-fraction', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if not 0 < args.train_fraction < 1:
        parser.error('--train-fraction must be between 0 and 1')

    records = structures(args.sdf)
    labels, verified = load_labels(args.charges, records)
    if not verified:
        parser.error('A source manifest is required for a reproducible labelled split')
    source_manifest_path = args.charges.with_name('manifest.json')
    source_manifest = json.loads(source_manifest_path.read_text())
    output = new_output(args.output)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(records))
    n_train = int(len(records) * args.train_fraction)
    if n_train < 1 or n_train >= len(records):
        parser.error('The split must contain at least one molecule in each subset')
    subsets = {'train': order[:n_train].tolist(), 'test': order[n_train:].tolist()}
    source_by_id = {identifier: (source, mol) for identifier, source, mol in records}
    split_ids = {}

    for split, indices in subsets.items():
        sdf_dir = output / split / 'sdf'
        charge_dir = output / split / 'charges'
        sdf_dir.mkdir(parents=True)
        charge_dir.mkdir()
        arrays = {}
        metadata = {}
        rows = []
        for index in indices:
            identifier, source, mol = records[index]
            destination = sdf_dir / source.name
            shutil.copyfile(source, destination)
            if digest(destination) != digest(source):
                raise ValueError(f'Source copy changed bytes: {identifier}')
            arrays[identifier] = labels[identifier]
            metadata[identifier] = dict(source_manifest['molecules'][identifier])
            rows.append({'CSD_identifier': identifier, 'smiles': Chem.MolToSmiles(mol),
                         'charges': json.dumps(labels[identifier].tolist()),
                         'atom_indices': json.dumps(atom_metadata(mol)['atom_indices'])})
        write_charges(charge_dir, rows, arrays, metadata, 'psiresp', 'all_atom')
        split_manifest_path = charge_dir / 'manifest.json'
        split_manifest = json.loads(split_manifest_path.read_text())
        split_manifest.update({'format': 'ccdc-trusted-labelled-dataset-v1',
                               'split': split, 'source_dataset': str(args.sdf.resolve()),
                               'source_dataset_sha256': digest(args.sdf) if args.sdf.is_file() else None,
                               'source_charges': str(args.charges.resolve()),
                               'source_charges_sha256': digest(args.charges),
                               'seed': args.seed, 'train_fraction': args.train_fraction,
                               'molecules': metadata})
        write_json(split_manifest_path, split_manifest)
        checked_records = structures(sdf_dir)
        checked_labels, checked = load_labels(charge_dir / 'charges.npz', checked_records)
        if not checked or set(checked_labels) != set(arrays):
            raise ValueError(f'Split provenance check failed: {split}')
        split_ids[split] = [records[index][0] for index in indices]

    train_ids = set(split_ids['train'])
    test_ids = set(split_ids['test'])
    if train_ids & test_ids:
        raise ValueError('Train/test identifier overlap')
    train_identity = {Chem.MolToSmiles(Chem.RemoveHs(source_by_id[i][1]), isomericSmiles=False)
                      for i in train_ids}
    test_identity = {Chem.MolToSmiles(Chem.RemoveHs(source_by_id[i][1]), isomericSmiles=False)
                     for i in test_ids}
    if train_identity & test_identity:
        raise ValueError('Train/test connectivity overlap')

    write_json(output / 'split.json', {
        'format': 'ccdc-labelled-split-v1', 'source_sdf': str(args.sdf.resolve()),
        'source_charges': str(args.charges.resolve()),
        'source_sdf_sha256': digest(args.sdf) if args.sdf.is_file() else None,
        'source_charges_sha256': digest(args.charges), 'seed': args.seed,
        'train_fraction': args.train_fraction, 'train_count': len(split_ids['train']),
        'test_count': len(split_ids['test']), 'train_ids': split_ids['train'],
        'test_ids': split_ids['test'], 'connectivity_disjoint': True,
    })
    print(f"Created {len(split_ids['train'])} train and {len(split_ids['test'])} test molecules: {output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
