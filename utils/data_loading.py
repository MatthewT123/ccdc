import os
from tqdm import tqdm

# ML imports
import torch

# Chem imports
from rdkit import Chem

MAP_ATOM_TYPE_ONLY_TO_INDEX = {
    6: 0,
    7: 1,
    8: 2,
    9: 3,
    15: 4,
    16: 5,
    17: 6,
    35: 7,
    53: 8,
}

MAP_ATOM_TYPE_TO_ATOM_SIGN = {
    1: 'H',
    6: 'C',
    7: 'N',
    8: 'O',
    9: 'F',
    15: 'P',
    16: 'S',
    17: 'Cl',
    35: 'S',
    53: 'I',
}


def process_sdf_files_to_list(root_folder):
    """
    Recursively find all sdf files in the specified folder, process each sdf file,
    and return a list containing processing results of all molecules.
    If a molecule contains atoms not in the MAP_ATOM_TYPE_ONLY_TO_INDEX dictionary, it will be discarded.

    Args:
        root_folder (str): The path of the root folder to search.

    Returns:
        list: A list containing processing results of each molecule, where each element is a dictionary
              with keys like 'h', 'x', 'atom_num', etc.
    """
    result_list = []
    total_molecules = 0

    for foldername, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.sdf'):
                sdf_file_path = os.path.join(foldername, filename)
                suppl = Chem.SDMolSupplier(sdf_file_path)
                total_molecules += len(suppl)

    progress_bar = tqdm(total=total_molecules, desc='Processing molecules', unit='mol')

    for foldername, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.sdf'):
                sdf_file_path = os.path.join(foldername, filename)
                suppl = Chem.SDMolSupplier(sdf_file_path)
                for mol in suppl:
                    if mol is not None:
                        atoms = mol.GetAtoms()
                        atom_types = [a.GetAtomicNum() for a in atoms]

                        # Check if atom types in the molecule are all in MAP_ATOM_TYPE_ONLY_TO_INDEX
                        valid_molecule = all(atomic_num in MAP_ATOM_TYPE_ONLY_TO_INDEX for atomic_num in atom_types)

                        if valid_molecule:
                            conformer = mol.GetConformer()
                            positions = conformer.GetPositions()
                            atom_indices = [MAP_ATOM_TYPE_ONLY_TO_INDEX[atomic_num] for atomic_num in atom_types]

                            result_list.append({
                                'h': torch.tensor(atom_indices, dtype=torch.long),
                                'x': torch.tensor(positions, dtype=torch.float32),
                                'atom_num': len(atom_indices)
                            })
                        else:
                            print(f"Discarded molecule with invalid atom types: {mol.GetProp('_Name')}")
                        progress_bar.update(1)

    progress_bar.close()
    return result_list
