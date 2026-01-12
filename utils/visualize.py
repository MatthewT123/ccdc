from IPython.display import display, HTML
import tempfile, os, colorsys
import py3Dmol

from ccdc import io
from ccdc.io import MoleculeWriter

from rdkit import Chem
from rdkit.Chem import AllChem

def visualise_mol(mol, optimised_mol=None):
    """Visualise original molecule with default element colours, 
    and optional optimised molecule in a distinct colour."""

    view = py3Dmol.view(width=700, height=480)

    # ---------- model 0: original (default colours) ----------
    tmp1 = tempfile.NamedTemporaryFile(delete=False, suffix='.sdf')
    with io.MoleculeWriter(tmp1.name) as w:
        w.write(mol)
    with open(tmp1.name, 'r') as fh:
        sdf1 = fh.read()

    view.addModel(sdf1, 'sdf')   # model index = 0
    view.setStyle({'model': 0}, {'stick': {}}) 

    # ---------- model 1: optimised (custom colour) ----------
    if optimised_mol is not None:
        tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix='.sdf')
        with io.MoleculeWriter(tmp2.name) as w:
            w.write(optimised_mol)
        with open(tmp2.name, 'r') as fh:
            sdf2 = fh.read()

        view.addModel(sdf2, 'sdf')   # model index = 1
        view.setStyle({'model': 1}, {'stick': {'color': 'red'}})

    # camera, background
    view.zoomTo()
    view.setBackgroundColor('white')
    display(view)


def _distinct_colors_hex(n):
    cols = []
    for i in range(n):
        h = i / max(1, n)
        r,g,b = colorsys.hsv_to_rgb(h, 0.7, 0.9)
        cols.append('#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255)))
    return cols

def _write_conformer_sdf(conf):
    # Accept either a CCDC conformer (with .molecule) or a CCDC Molecule
    mol_obj = getattr(conf, 'molecule', conf)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.sdf')
    tmp_name = tmp.name
    tmp.close()
    with MoleculeWriter(tmp_name) as w:
        w.write(mol_obj)
    with open(tmp_name, 'r', encoding='utf-8', errors='replace') as fh:
        sdf_text = fh.read()
    return tmp_name, sdf_text

def visualise_conformers(conformers,
                         align=True,
                         preserve_first_element_colours=True,
                         stick_radius=0.12,
                         stick_opacity=0.80,
                         width=900, height=700):
    """
    Overlay many conformers in one py3Dmol view showing bonds.

    conformers : iterable of CCDC Conformer objects or CCDC Molecules
    align      : attempt to align all conformers to the first (uses RDKit if installed)
    preserve_first_element_colours : if True, model 0 uses element colours; others use distinct colours
    stick_radius, sphere_radius, sphere_opacity : visual tuning
    """
    conformers = list(conformers)
    n = len(conformers)
    if n == 0:
        raise ValueError("No conformers provided")


    sdf_texts = []
    tmp_files = []
    # Convert molblocks to RDKit and align
    if align and n > 1:
        # create rdkit mols from conformer molblocks
        rdmols = []
        for conf in conformers:
            _, sdf_text = _write_conformer_sdf(conf)
            # parse first block only to RDKit Mol
            rdm = Chem.MolFromMolBlock(sdf_text, sanitize=True, removeHs=False)
            if rdm is None:
                raise RuntimeError("RDKit failed to parse a conformer molblock.")
            rdmols.append(rdm)

        ref = rdmols[0]

        # Ensure first has coordinates; if not, embed
        if ref.GetNumConformers() == 0:
            AllChem.EmbedMolecule(ref, AllChem.ETKDG())
        
        # Align each other to the reference using atom mapping
        for i in range(1, n):
            target = rdmols[i]
            if target.GetNumConformers() == 0:
                AllChem.EmbedMolecule(target, AllChem.ETKDG())
            try:
                AllChem.AlignMol(target, ref)  # modifies target conformer coordinates
            except Exception:
                # best-effort: try AlignMolConformers or skip
                try:
                    AllChem.AlignMolConformers(target, ref)
                except Exception:
                    pass

        # Convert aligned rdkit mols back to molblocks (with coordinates) for py3Dmol
        for rdm in rdmols:
            molblock = Chem.MolToMolBlock(rdm)
            sdf = molblock + "$$$$\n"
            sdf_texts.append(sdf)
     
    else:
        # Just write conformers via CCDC writer
        for conf in conformers:
            fname, sdf = _write_conformer_sdf(conf)
            tmp_files.append(fname)
            sdf_texts.append(sdf)

    # Build colors (leave first uncoloured if preserve_first_element_colours True)
    colors = _distinct_colors_hex(n)
    legend_html = []
    view = py3Dmol.view(width=width, height=height)

    # Add each model and style
    for i, sdf in enumerate(sdf_texts):
        view.addModel(sdf, 'sdf')   # model index i

        if i == 0 and preserve_first_element_colours:
            # element-based colouring: just set stick with no forced colour
            view.setStyle({'model': i}, {'stick': {'radius': stick_radius, 'color': "blue", 'opacity': stick_opacity}})

            # Faint spheres to improve visibility of overlapped atoms
            # view.setStyle({'model': i}, {'sphere': {'radius': sphere_radius, 'opacity': stick_opacity}})
           
        else:
            col = colors[i]
            # set colored sticks and semi-transparent sticks for clarity
            view.setStyle({'model': i}, {'stick': {'radius': stick_radius, 'color': col, 'opacity': stick_opacity}})
          
        # legend entry
        legend_html.append(f'<div style="display:inline-block;margin-right:8px;">'
                           f'<span style="display:inline-block;width:14px;height:12px;background:{colors[i]};margin-right:6px;"></span>'
                           f'Conf {i}</div>')

    # Camera and UI tweaks
    view.zoomTo()
    view.setBackgroundColor('0xFFFFFF')

    # Display legend + viewer
    legend = "<div style='font-family:sans-serif;margin-bottom:6px'>" + "".join(legend_html) + "</div>"
    display(HTML(legend))
    display(view)

    # Cleanup temporary files if any (the RDKit path does not create files)
    for f in tmp_files:
        try:
            os.remove(f)
        except Exception:
            pass

