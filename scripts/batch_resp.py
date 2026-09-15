"""Resumable per-molecule RESP jobs with bounded subprocess concurrency."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from rdkit import Chem
from _workflow.data import digest, load_labels, new_output, structures, write_charges, write_json


def parse_radius(value):
    try:
        element, radius = value.split('=', 1)
        radius = float(radius)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError('Expected ELEMENT=RADIUS, for example Br=1.85')
    element = element.strip().capitalize()
    if not element or not np.isfinite(radius) or radius <= 0:
        raise argparse.ArgumentTypeError('VDW radius must be a positive finite number')
    return element, radius


def failure_message(log_path):
    """Return the most useful error line from a failed child calculation."""
    try:
        lines = log_path.read_text(errors='replace').splitlines()
    except OSError:
        return 'RESP subprocess failed; see molecule log'
    for line in reversed(lines):
        text = line.strip()
        if any(marker in text for marker in ('Untrusted RESP magnitude:', 'KeyError:',
                                              'ValueError:', 'RuntimeError:', 'did not converge')):
            return text[-500:]
    return 'RESP subprocess failed; see molecule log'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdf', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--timeout', type=int, default=900)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--precise-fit', action='store_true')
    p.add_argument('--reuse-root', type=Path, help='Optional prior batch whose validated attempts may be reused')
    p.add_argument('--vdw-radius', action='append', type=parse_radius, default=[], metavar='ELEMENT=RADIUS')
    p.add_argument('--max-absolute-charge', type=float, default=2.0)
    a = p.parse_args()
    if (a.workers < 1 or a.timeout < 1 or not np.isfinite(a.max_absolute_charge)
            or a.max_absolute_charge <= 0):
        p.error('Workers, timeout, and charge bound must be positive')
    grid_radii = dict(a.vdw_radius)
    records = structures(a.sdf)
    fingerprints = {i: digest(path) for i, path, _ in records}
    output = a.output.resolve()
    if a.resume:
        old = json.loads((output/'run.json').read_text())
        if old['source_sha256'] != fingerprints: p.error('Resume input structures differ')
        if old.get('precise_fit',False) != a.precise_fit: p.error('Resume solver differs')
        if old.get('vdw_radii', {}) != grid_radii: p.error('Resume VDW radii differ')
        if old.get('max_absolute_charge', 2.0) != a.max_absolute_charge: p.error('Resume charge bound differs')
    else:
        output = new_output(output)
        write_json(output/'run.json', {'source_sha256': fingerprints, 'workers': a.workers,
            'method': 'HF/6-31G* PsiRESP defaults', 'timeout_seconds': a.timeout,
            'precise_fit': a.precise_fit, 'vdw_radii': grid_radii,
            'max_absolute_charge': a.max_absolute_charge})
    (output/'molecules').mkdir(exist_ok=True)
    script = Path(__file__).resolve().with_name('predict_charges.py')
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    def job(record):
        identifier, source, mol = record
        start = time.monotonic()
        directory = output/'molecules'/identifier
        # Completed results are immutable; validate provenance before reuse.
        roots = [output]
        if a.reuse_root:
            roots.append(a.reuse_root.resolve())
        for root in roots:
            prior_directory = root/'molecules'/identifier
            for attempt in sorted(prior_directory.glob('attempt-*'), reverse=True):
                if not (attempt/'charges.npz').exists():
                    continue
                try:
                    labels, verified = load_labels(attempt/'charges.npz', [record])
                    manifest = json.loads((attempt/'manifest.json').read_text())
                    entry = manifest['molecules'][identifier]
                    if (verified and entry.get('max_absolute_charge', 0) <= a.max_absolute_charge
                            and entry.get('max_h_absorbed_charge', 0) <= a.max_absolute_charge
                            and entry.get('vdw_radii', {}) == grid_radii):
                        return identifier, attempt, labels[identifier], 0.0, ''
                except (ValueError, OSError, KeyError, json.JSONDecodeError):
                    pass
        directory.mkdir(exist_ok=True)
        attempt = directory/f'attempt-{time.time_ns()}'
        try:
            command = [sys.executable, str(script), '--sdf', str(source),
                '--output', str(attempt), '--n-processes', '1']
            if a.precise_fit:
                command.append('--precise-fit')
            for element, radius in grid_radii.items():
                command.extend(['--vdw-radius', f'{element}={radius}'])
            command.extend(['--max-absolute-charge', str(a.max_absolute_charge)])
            with (directory/f'{attempt.name}.log').open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                    env=env, timeout=a.timeout)
            if result.returncode:
                raise ValueError(f'RESP exited {result.returncode}: {failure_message(directory / (attempt.name + ".log"))}')
            labels, verified = load_labels(attempt/'charges.npz', [record])
            if not verified: raise ValueError('Missing provenance')
            return identifier, attempt, labels[identifier], time.monotonic()-start, ''
        except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
            return identifier, attempt, None, time.monotonic()-start, str(exc)
    arrays, metadata, rows, statuses = {}, {}, [], []
    by_id = {r[0]:r for r in records}
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(job,r) for r in records]):
            identifier, attempt, q, seconds, error = future.result()
            statuses.append({'identifier':identifier, 'status':'failed' if error else 'ok',
                             'seconds':seconds, 'message':error})
            if q is not None:
                entry = json.loads((attempt/'manifest.json').read_text())['molecules'][identifier]
                arrays[identifier] = q
                metadata[identifier] = entry
                rows.append({'CSD_identifier':identifier, 'smiles':Chem.MolToSmiles(by_id[identifier][2]),
                    'charges':json.dumps(q.tolist()), 'atom_indices':json.dumps(entry['atom_indices'])})
            with (output/'status.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['identifier','status','seconds','message']); w.writeheader(); w.writerows(statuses)
            write_json(output/'progress.json', {'completed':len(statuses),'successful':len(arrays),
                'failed':len(statuses)-len(arrays),'total':len(records),'elapsed_seconds':time.monotonic()-start})
            if len(statuses)%10 == 0 or error: print(f'{len(statuses)}/{len(records)} completed; {len(arrays)} successful; {identifier}: {error or "ok"}',flush=True)
    write_charges(output, rows, arrays, metadata, 'psiresp', 'all_atom')
    write_json(output/'result.json', {'successful':len(arrays),'failed':len(records)-len(arrays),
        'elapsed_seconds':time.monotonic()-start})
    return int(len(arrays)!=len(records))


if __name__=='__main__':
    sys.exit(main())
