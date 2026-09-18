"""Copy only provenance-checked, bounded RESP-labelled molecules into a dataset."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from rdkit import Chem

from _workflow.data import (absorb_hydrogen_charges, atom_metadata, digest,
                            load_labels, new_output, structures, write_charges,
                            write_json)


def batch_failure_reason(batch_root, identifier):
    """Recover a useful failure class from the per-molecule subprocess log."""
    logs = sorted((Path(batch_root) / 'molecules' / identifier).glob('attempt-*.log'), reverse=True)
    lines = []
    for log in logs:
        try:
            lines.extend(log.read_text(errors='replace').splitlines())
        except OSError:
            continue
    for line in reversed(lines):
        text = line.strip()
        if 'Untrusted RESP magnitude:' in text:
            return 'untrusted_resp_magnitude:' + text.split('Untrusted RESP magnitude:', 1)[1].strip()
        if 'not supported elements' in text and 'Br' in text:
            return 'psi_resp_missing_bromine_radius'
        if 'did not converge' in text:
            return 'resp_not_converged'
        if 'psi4:' in text or 'psi4 exited' in text:
            return 'psi4_failed:' + text[-240:]
        if 'RuntimeError:' in text:
            return 'resp_pipeline_error:' + text.split('RuntimeError:', 1)[1].strip()[-240:]
    return 'missing_label_or_psi4_failure'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdf', type=Path, required=True)
    parser.add_argument('--charges', type=Path, required=True,
                        help='Batch output directory containing charges.npz and manifest.json')
    parser.add_argument('--output', type=Path, required=True,
                        help='Fresh output directory; accepted SDFs go under sdf/')
    parser.add_argument('--max-absolute-charge', type=float, default=2.0)
    args = parser.parse_args(argv)
    if not np.isfinite(args.max_absolute_charge) or args.max_absolute_charge <= 0:
        parser.error('--max-absolute-charge must be positive and finite')

    records = structures(args.sdf)
    charge_npz = args.charges / 'charges.npz'
    if not charge_npz.is_file():
        parser.error(f'Missing charge archive: {charge_npz}')
    source_manifest = args.charges / 'manifest.json'
    if not source_manifest.is_file():
        parser.error(f'Missing charge manifest: {source_manifest}')
    source_manifest_data = json.loads(source_manifest.read_text())
    output = new_output(args.output)
    sdf_output = output / 'sdf'
    sdf_output.mkdir()
    accepted = []
    rejected = []
    with np.load(charge_npz, allow_pickle=False) as archive:
        available = set(archive.files)
        for identifier, source, mol in records:
            reason = None
            q = None
            if identifier not in available:
                reason = batch_failure_reason(args.charges, identifier)
            else:
                try:
                    labels, verified = load_labels(charge_npz, [(identifier, source, mol)])
                    if not verified:
                        reason = 'missing_source_manifest'
                    else:
                        q = labels[identifier]
                        absorbed = absorb_hydrogen_charges(mol, q)
                        max_all = float(np.max(np.abs(q)))
                        max_heavy = float(np.max(np.abs(absorbed)))
                        if max_all > args.max_absolute_charge or max_heavy > args.max_absolute_charge:
                            reason = (f'untrusted_charge_magnitude:max_all={max_all:.6g};'
                                      f'max_h_absorbed={max_heavy:.6g}')
                except (KeyError, OSError, ValueError) as exc:
                    reason = f'invalid_label:{exc}'
            if reason:
                rejected.append({'identifier': identifier, 'status': 'rejected', 'reason': reason})
                continue
            destination = sdf_output / source.name
            # Copy bytes rather than reserializing: the source hash is part of
            # the label provenance contract and must remain identical.
            shutil.copyfile(source, destination)
            metadata = source_manifest_data['molecules'][identifier]
            metadata = {**metadata, 'filtered_source_sha256': digest(destination),
                        'max_absolute_charge_limit': args.max_absolute_charge}
            accepted.append((identifier, destination, mol, q, metadata))

    arrays = {identifier: q for identifier, _, _, q, _ in accepted}
    metadata = {identifier: entry for identifier, _, _, _, entry in accepted}
    rows = [{'CSD_identifier': identifier, 'smiles': Chem.MolToSmiles(mol),
             'charges': json.dumps(q.tolist()),
             'atom_indices': json.dumps(atom_metadata(mol)['atom_indices'])}
            for identifier, _, mol, q, _ in accepted]
    write_charges(output, rows, arrays, metadata, method='psiresp', convention='all_atom')
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update({'format': 'ccdc-trusted-labelled-dataset-v1',
                     'source_sdf': str(Path(args.sdf).resolve()),
                     'source_sdf_sha256': digest(Path(args.sdf)) if Path(args.sdf).is_file() else None,
                     'source_charges': str(charge_npz.resolve()),
                     'source_charges_sha256': digest(charge_npz),
                     'max_absolute_charge': args.max_absolute_charge,
                     'accepted': len(accepted), 'rejected': len(rejected),
                     'rejections': rejected})
    write_json(manifest_path, manifest)
    with (output / 'status.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['identifier', 'status', 'reason'])
        writer.writeheader()
        writer.writerows([{'identifier': i, 'status': 'accepted', 'reason': ''} for i, *_ in accepted])
        writer.writerows(rejected)
    write_json(output / 'dataset.json', {
        'format': 'ccdc-trusted-labelled-dataset-v1', 'sdf': 'sdf',
        'charges': 'charges.npz', 'manifest': 'manifest.json',
        'accepted': len(accepted), 'rejected': len(rejected),
        'max_absolute_charge': args.max_absolute_charge,
        'rejection_policy': 'missing or provenance-invalid labels, non-finite values, charge-sum mismatch, or all-atom/H-absorbed magnitude above the bound',
    })
    print(f'Accepted {len(accepted)} molecules; rejected {len(rejected)}. Dataset: {output}')
    return 0 if accepted else 1


if __name__ == '__main__':
    raise SystemExit(main())
