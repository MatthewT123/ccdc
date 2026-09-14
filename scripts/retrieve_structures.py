"""Retrieve CSD structures from a CSV, or import local SDFs into the same workflow."""

import argparse
import csv
from pathlib import Path
import sys

from _workflow.data import (atom_metadata, digest, new_output, read_molecule,
                            sdf_paths, valid_identifier, write_json, write_sdf)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--csv", type=Path, help="CSV with a 'Database identifier' column; requires CCDC/CSD")
    inputs.add_argument("--sdf", type=Path, help="Import a local SDF file/directory without CSD access")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args(argv)
    reader = None
    if args.csv:
        with args.csv.open(newline="") as stream:
            table = csv.DictReader(stream)
            if "Database identifier" not in (table.fieldnames or []):
                parser.error("CSV must contain 'Database identifier'")
            identifiers = [valid_identifier(r["Database identifier"]) for r in table]
        if not identifiers or len(set(identifiers)) != len(identifiers):
            parser.error("CSV must contain at least one identifier and no duplicates")
        try:
            from ccdc import io
            reader = io.EntryReader("CSD")
        except Exception as exc:
            parser.error(f"CSD access is unavailable: {exc}. Install/configure CCDC and its licence, or use --sdf.")
        jobs = [(identifier, None) for identifier in identifiers]
    else:
        jobs = [(valid_identifier(p.stem), p) for p in sdf_paths(args.sdf)]
    output = new_output(args.output)
    target = output / "sdf"
    target.mkdir()
    statuses, metadata = [], {}
    for identifier, source in jobs:
        path = target / f"{identifier}.sdf"
        try:
            if reader is not None:
                mol = reader.entry(identifier).molecule
                mol = max(mol.components, key=lambda component: len(list(component.atoms)))
                with io.MoleculeWriter(str(path)) as writer:
                    writer.write(mol)
                mol = read_molecule(path)
            else:
                # Preserve imported atom order and hydrogens; do not strip fragments silently.
                mol = read_molecule(source)
                write_sdf(path, mol)
                mol = read_molecule(path)
            metadata[identifier] = {"sdf": str(path.relative_to(output)), "sha256": digest(path), **atom_metadata(mol)}
            statuses.append({"identifier": identifier, "status": "ok", "message": ""})
        except Exception as exc:
            # Preserve failed output for diagnosis, outside the downstream SDF directory.
            if path.exists():
                failures = output / "failed"
                failures.mkdir(exist_ok=True)
                path.rename(failures / path.name)
            statuses.append({"identifier": identifier, "status": "failed", "message": str(exc)})
    with (output / "status.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["identifier", "status", "message"])
        writer.writeheader()
        writer.writerows(statuses)
    write_json(output / "manifest.json", {"source": "csd" if reader is not None else "local_sdf",
        "preparation": "largest_component" if reader is not None else "preserve_input",
        "molecules": metadata})
    failed = sum(row["status"] != "ok" for row in statuses)
    print(f"Retrieved {len(metadata)} structures; {failed} failed. Outputs: {output}")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as exc:
        sys.exit(str(exc))
