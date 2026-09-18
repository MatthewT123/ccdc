"""Export a fixed, molecule-disjoint small-molecule CSD train/eval dataset."""

import argparse
from collections import Counter
import csv
from pathlib import Path

import numpy as np
from rdkit import Chem

from _workflow.data import (atom_metadata, digest, new_output, read_molecule,
                            sdf_paths, write_json, write_sdf)
from _workflow.licensing import load_ccdc_license

# Intersection of model elements and the current PsiRESP HF/6-31G* basis.
# The model supports iodine, but this quantum basis does not.
SUPPORTED = {1, 6, 7, 8, 9, 15, 16, 17, 35}


def exclusion_sets(sources):
    """Return IDs, refcode families, and connectivity identities to avoid."""
    identifiers, families, identities = set(), set(), set()
    for source in sources:
        for path in sdf_paths(source):
            molecule = read_molecule(path)
            identifier = path.stem
            identifiers.add(identifier)
            families.add(identifier[:6])
            identities.add(Chem.MolToSmiles(Chem.RemoveHs(molecule), isomericSmiles=False))
    return identifiers, families, identities


def prepare(entry, min_heavy, max_heavy):
    if not entry.has_3d_structure or entry.has_disorder or entry.is_polymeric:
        raise ValueError('missing_3d_disorder_or_polymer')
    if entry.r_factor is None or not np.isfinite(entry.r_factor) or entry.r_factor > 5.0:
        raise ValueError('r_factor')
    molecule = entry.molecule
    if len(molecule.components) != 1:
        raise ValueError('multiple_components')
    source_numbers = [a.atomic_number for a in molecule.atoms]
    if 6 not in source_numbers or set(source_numbers) - SUPPORTED:
        raise ValueError('unsupported_elements')
    if not min_heavy <= sum(n != 1 for n in source_numbers) <= max_heavy:
        raise ValueError('heavy_atom_count')
    # Retain measured heavy-atom geometry; use one explicit-H preparation rule.
    mol = Chem.MolFromMolBlock(molecule.to_string('sdf'), removeHs=False)
    if mol is None:
        raise ValueError('rdkit_parse')
    numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]
    if 6 not in numbers or set(numbers) - SUPPORTED:
        raise ValueError('unsupported_elements')
    heavy_count = sum(n != 1 for n in numbers)
    if not min_heavy <= heavy_count <= max_heavy:
        raise ValueError('heavy_atom_count')
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise ValueError('charged_or_radical')
    if any(a.GetIsotope() for a in mol.GetAtoms()):
        raise ValueError('isotope')
    if any(a.GetAtomicNum() in {9, 17, 35, 53} and a.GetTotalValence() != 1 for a in mol.GetAtoms()):
        raise ValueError('nonstandard_halogen_valence')
    normal_chirality = {Chem.ChiralType.CHI_UNSPECIFIED, Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                        Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
    if any(a.GetChiralTag() not in normal_chirality for a in mol.GetAtoms()):
        raise ValueError('nonstandard_stereochemistry')
    mol = Chem.RemoveHs(mol)
    coords = mol.GetConformer().GetPositions()
    radii = np.array([Chem.GetPeriodicTable().GetRcovalent(a.GetAtomicNum()) for a in mol.GetAtoms()])
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    expected = radii[:, None] + radii[None, :]
    if np.any(distances[np.triu_indices(len(coords), 1)] < 0.65 * expected[np.triu_indices(len(coords), 1)]):
        raise ValueError('heavy_atom_clash')
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if distances[a, b] > 1.35 * expected[a, b]:
            raise ValueError('long_bond')
    identity = Chem.MolToSmiles(mol, isomericSmiles=False)
    mol = Chem.AddHs(mol, addCoords=True)
    if not mol.GetConformer().Is3D() or not np.isfinite(mol.GetConformer().GetPositions()).all():
        raise ValueError('invalid_coordinates')
    saved = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol), removeHs=False)
    if saved is None or Chem.MolToSmiles(Chem.RemoveHs(saved), isomericSmiles=False) != identity:
        raise ValueError('sdf_roundtrip_changed_identity')
    if Chem.AddHs(saved).GetNumAtoms() != saved.GetNumAtoms():
        raise ValueError('incomplete_hydrogens')
    return mol, identity, heavy_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-size', type=int, default=1000)
    parser.add_argument('--eval-size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--candidates-csv', type=Path, help='Optional fixed candidate IDs (Database identifier column); all quality checks still apply')
    parser.add_argument('--exclude-sdf', type=Path, action='append', default=[],
                        help='SDF file or directory whose identifiers, refcode families, and connectivity are excluded')
    parser.add_argument('--min-heavy', type=int, default=3)
    parser.add_argument('--max-heavy', type=int, default=10)
    args = parser.parse_args()
    if min(args.train_size, args.eval_size, args.min_heavy) < 1 or args.max_heavy < args.min_heavy:
        parser.error('Sizes must be positive and heavy-atom bounds ordered')
    load_ccdc_license()
    from ccdc import io
    from rdkit import rdBase
    with io.EntryReader('CSD') as reader:
        output = new_output(args.output)
        rng = np.random.default_rng(args.seed)
        selected, identities, families = [], set(), set()
        rejected = Counter()
        excluded_ids, excluded_families, excluded_identities = exclusion_sets(args.exclude_sdf)
        target = args.train_size + args.eval_size
        if args.candidates_csv:
            with args.candidates_csv.open(newline='') as stream:
                candidates = [r['Database identifier'] for r in csv.DictReader(stream)]
            candidates = [candidates[int(i)] for i in rng.permutation(len(candidates))]
        else:
            candidates = rng.permutation(len(reader))
        for scanned, index in enumerate(candidates, 1):
            if scanned % 10000 == 0:
                print(f'Screened {scanned} entries; selected {len(selected)}/{target}', flush=True)
            try:
                entry = reader.entry(index) if isinstance(index, str) else reader[int(index)]
                family = entry.identifier[:6]
                if entry.identifier in excluded_ids:
                    rejected['excluded_existing_identifier'] += 1
                    continue
                if family in excluded_families:
                    rejected['excluded_existing_refcode_family'] += 1
                    continue
                if family in families:
                    rejected['duplicate_refcode_family'] += 1
                    continue
                mol, identity, heavy_count = prepare(entry, args.min_heavy, args.max_heavy)
                if identity in excluded_identities:
                    rejected['excluded_existing_connectivity'] += 1
                    continue
                if identity in identities:
                    rejected['duplicate_connectivity'] += 1
                    continue
            except (ValueError, RuntimeError) as exc:
                reason = str(exc) if isinstance(exc, ValueError) else 'ccdc_error'
                rejected[reason] += 1
                continue
            identities.add(identity)
            families.add(family)
            selected.append((entry.identifier, mol, identity, heavy_count, entry.r_factor))
            if len(selected) % 100 == 0:
                print(f'Selected {len(selected)}/{target}; rejected {sum(rejected.values())}', flush=True)
            if len(selected) == target:
                break
        if len(selected) != target:
            write_json(output / 'incomplete.json', {'selected': len(selected), 'rejections': dict(rejected)})
            raise RuntimeError('Not enough eligible molecules; no completed split written')
        order = rng.permutation(target)
        manifest = {'format': 'ccdc-static-split-v1', 'seed': args.seed,
            'database_version': str(io.csd_version(return_current_version=True)),
            'database_entries': len(reader), 'rdkit_version': rdBase.rdkitVersion,
            'screened_entries': scanned,
            'excluded_sdf': [str(path.resolve()) for path in args.exclude_sdf],
            'excluded_molecules': len(excluded_ids),
            'candidates_sha256': digest(args.candidates_csv) if args.candidates_csv else None,
            'filters': {'min_heavy': args.min_heavy, 'max_heavy': args.max_heavy,
                'max_r_factor_percent': 5.0, 'single_component': True,
                'no_disorder_or_polymers': True, 'neutral_atoms_no_radicals': True,
                'no_isotopes': True, 'min_heavy_distance_covalent_radii_ratio': 0.65,
                'max_bond_length_covalent_radii_ratio': 1.35,
                'halogen_valence_one': True, 'standard_stereochemistry': True,
                'sdf_roundtrip_identity_and_hydrogens': True,
                'supported_atomic_numbers': sorted(SUPPORTED)},
            'reference_charge_method': 'HF/6-31G* (labels not generated by this script)',
            'preparation': 'Keep CSD heavy-atom coordinates; regenerate explicit H with RDKit AddHs(addCoords=True); no geometry optimization',
            'deduplication': 'Unique stereo-independent canonical SMILES and CSD refcode family across both splits',
            'eval_policy': 'Held out for later testing; use training molecules for validation/model selection',
            'rejections': dict(rejected), 'molecules': []}
        for split, indices in [('train', order[:args.train_size]), ('eval', order[args.train_size:])]:
            sdf_dir = output / split / 'sdf'
            sdf_dir.mkdir(parents=True)
            rows = []
            for index in indices:
                identifier, mol, identity, count, r_factor = selected[int(index)]
                path = sdf_dir / f'{identifier}.sdf'
                write_sdf(path, mol)
                saved = read_molecule(path)
                rows.append({'Database identifier': identifier})
                manifest['molecules'].append({'identifier': identifier, 'split': split,
                    'sdf': str(path.relative_to(output)), 'sha256': digest(path),
                    'connectivity_smiles': identity, 'heavy_atoms': count, 'r_factor': r_factor,
                    **atom_metadata(saved)})
            with (output / f'{split}.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['Database identifier'])
                writer.writeheader()
                writer.writerows(rows)
        write_json(output / 'manifest.json', manifest)
        print(f'Created {args.train_size} train and {args.eval_size} eval structures: {output}', flush=True)


if __name__ == '__main__':
    main()
