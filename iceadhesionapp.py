# -*- coding: utf-8 -*-
"""
Created on Fri July 19 17:41:37 2024

@author: Gerardo Casanola
"""


# ============================================================
# Importing libraries
# ============================================================

import os
import re
import glob
import base64
import joblib
from pathlib import Path
from io import BytesIO

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from sklearn.preprocessing import MinMaxScaler

from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Ice adhesion performance predictor",
    page_icon=":computer:",
    layout="wide"
)


# ============================================================
# HEADER IMAGE
# ============================================================

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


header_path = "cropped_header.png"

if os.path.exists(header_path):
    image = Image.open(header_path)
    st.image(image, use_container_width=True)
else:
    st.warning("Header image 'cropped_header.png' was not found.")


# ============================================================
# INTERNAL PATHS — hidden from user interface
# ============================================================

DATA_PATH = "data/data_MLR_2var_weighted_descriptors_observed_vs_predicted.csv"
MODEL_FOLDER = "models"
MODEL_FILENAME = "mlr_model.pkl"


# ============================================================
# INTERNAL SETTINGS — hidden from user interface
# ============================================================

INTERNAL_MODEL_FEATURES = [
    "nAB_weighted",
    "F07_C_O_weighted"
]

# Fixed internally
SEQUENCE_MODE = "even"
CAP_UNITS = 2
PERCENT_BASIS = "total_units_including_caps"


# ============================================================
# User-facing labels
# ============================================================

PUBLIC_LABELS = {
    "Oil_ID": "Oil ID",
    "Oil_ID_clean": "Cleaned Oil ID",
    "family": "Oil family",
    "total_units_including_caps": "Total units including caps",
    "cap_units": "Cap-ending units",
    "internal_repeat_units": "Internal repeat units",
    "n_total_repeat_units": "Total internal repeat units",
    "achieved_percent_DM_total_units": "Achieved DM % based on total units",
    "achieved_percent_PM_total_units": "Achieved PM % based on total units",
    "achieved_percent_DP_total_units": "Achieved DP % based on total units",
    "total_MW_input": "Input total MW",
    "backbone_MW_input": "Estimated backbone MW",
    "X_type": "Second repeat unit",
    "percent_DM_input": "Input DM (%)",
    "percent_X_input": "Input PM/DP (%)",
    "percent_basis": "Composition basis",
    "n_DM": "Estimated DM units",
    "n_PM": "Estimated PM units",
    "n_DP": "Estimated DP units",
    "n_total": "Total repeat units",
    "achieved_backbone_MW": "Achieved backbone MW",
    "achieved_total_MW": "Achieved total MW",
    "achieved_percent_DM_repeat": "Achieved DM repeat %",
    "achieved_percent_PM_repeat": "Achieved PM repeat %",
    "achieved_percent_DP_repeat": "Achieved DP repeat %",
    "achieved_percent_DM_weight": "Achieved DM weight %",
    "achieved_percent_PM_weight": "Achieved PM weight %",
    "achieved_percent_DP_weight": "Achieved DP weight %",
    "oil_loading_percent": "Oil loading (%)",
    "oil_loading_fraction": "Oil loading fraction",
    "sequence_mode": "Sequence mode",
    "sequence": "Representative sequence",
    "SMILES": "Representative SMILES",
    "nAB": "Aromatic-bond descriptor count",
    "F07_C_O": "C–O topological descriptor count",
    "UFF_initial_energy_kcal_mol": "Initial UFF energy",
    "UFF_final_energy_kcal_mol": "Final UFF energy",
    "UFF_energy_change_kcal_mol": "UFF energy change",
    "UFF_converged": "UFF converged",
    "UFF_error": "UFF optimization error",
    "3D_embedding_status": "3D embedding status",
    "3D_source": "3D structure source",
    "3D_embedding_error": "3D embedding error",
    "uploaded_sdf_filename": "Uploaded SDF filename",
}


# ============================================================
# Constants
# ============================================================

UNIT_MW = {
    "DM": 76.17,
    "PM": 136.241,
    "DP": 200.312,
}

LEFT_END_MW = 74.198
RIGHT_END_MW = 90.197


# ============================================================
# Molecule construction
# ============================================================

def add_atom(rwmol, symbol, aromatic=False):
    atom = Chem.Atom(symbol)
    atom.SetIsAromatic(aromatic)
    return rwmol.AddAtom(atom)


def add_methyl(rwmol, si_idx):
    c_idx = add_atom(rwmol, "C")
    rwmol.AddBond(si_idx, c_idx, Chem.BondType.SINGLE)
    return c_idx


def add_phenyl(rwmol, si_idx):
    ring_atoms = []

    for _ in range(6):
        c_idx = add_atom(rwmol, "C", aromatic=True)
        ring_atoms.append(c_idx)

    for i in range(6):
        rwmol.AddBond(
            ring_atoms[i],
            ring_atoms[(i + 1) % 6],
            Chem.BondType.AROMATIC
        )

    rwmol.AddBond(si_idx, ring_atoms[0], Chem.BondType.SINGLE)
    return ring_atoms


def add_substituents(rwmol, si_idx, unit_type):
    unit_type = unit_type.upper()

    if unit_type == "DM":
        add_methyl(rwmol, si_idx)
        add_methyl(rwmol, si_idx)

    elif unit_type == "PM":
        add_methyl(rwmol, si_idx)
        add_phenyl(rwmol, si_idx)

    elif unit_type == "DP":
        add_phenyl(rwmol, si_idx)
        add_phenyl(rwmol, si_idx)

    else:
        raise ValueError("unit_type must be DM, PM, or DP.")


def make_even_sequence(n_DM, n_X, X_type):
    total = n_DM + n_X

    if total <= 0:
        raise ValueError("Total repeat units must be > 0.")

    sequence = ["DM"] * total

    if n_X > 0:
        positions = [
            round((i + 1) * total / (n_X + 1)) - 1
            for i in range(n_X)
        ]

        used = set()

        for pos in positions:
            pos = max(0, min(total - 1, pos))

            while pos in used and pos < total - 1:
                pos += 1

            while pos in used and pos > 0:
                pos -= 1

            used.add(pos)
            sequence[pos] = X_type

    return sequence


def build_silicone_oil(sequence):
    """
    Build Me3Si-O-[SiR2-O]n-SiMe3 connectivity.
    """

    rwmol = Chem.RWMol()

    left_si = add_atom(rwmol, "Si")
    add_methyl(rwmol, left_si)
    add_methyl(rwmol, left_si)
    add_methyl(rwmol, left_si)

    previous_si = left_si

    for unit in sequence:
        o_idx = add_atom(rwmol, "O")
        rwmol.AddBond(previous_si, o_idx, Chem.BondType.SINGLE)

        si_idx = add_atom(rwmol, "Si")
        rwmol.AddBond(o_idx, si_idx, Chem.BondType.SINGLE)

        add_substituents(rwmol, si_idx, unit)

        previous_si = si_idx

    o_idx = add_atom(rwmol, "O")
    rwmol.AddBond(previous_si, o_idx, Chem.BondType.SINGLE)

    right_si = add_atom(rwmol, "Si")
    rwmol.AddBond(o_idx, right_si, Chem.BondType.SINGLE)

    add_methyl(rwmol, right_si)
    add_methyl(rwmol, right_si)
    add_methyl(rwmol, right_si)

    mol = rwmol.GetMol()
    Chem.SanitizeMol(mol)

    return mol


# ============================================================
# Descriptor calculations
# ============================================================

def calc_nAB(mol):
    """
    nAB = number of aromatic bonds.
    """
    return sum(1 for bond in mol.GetBonds() if bond.GetIsAromatic())


def calc_F07_C_O(mol):
    """
    F07[C-O] = number of C-O atom pairs at topological distance 7.
    """

    distance_matrix = Chem.GetDistanceMatrix(mol)
    atoms = mol.GetAtoms()

    count = 0

    for i, atom_i in enumerate(atoms):
        zi = atom_i.GetAtomicNum()

        for j in range(i + 1, len(atoms)):
            zj = atoms[j].GetAtomicNum()

            is_C_O_pair = (
                (zi == 6 and zj == 8) or
                (zi == 8 and zj == 6)
            )

            if is_C_O_pair and int(distance_matrix[i, j]) == 7:
                count += 1

    return count


# ============================================================
# Oil-ID parser and repeat-count estimation
# ============================================================

def format_percent_code(percent_X):
    """
    Convert a percentage to the Oil ID code.

    Examples:
        5   -> 005
        20  -> 020
        7.5 -> 07.5

    The app uses integer percentages by default, but this function also
    supports decimal values if you later allow them in the interface.
    """

    percent_X = float(percent_X)

    if percent_X.is_integer():
        return f"{int(round(percent_X)):03d}"

    percent_code = f"{percent_X:.2f}".rstrip("0").rstrip(".")

    if percent_X < 10:
        percent_code = "0" + percent_code

    return percent_code


def build_oil_id_from_inputs(X_type, percent_X, total_units_including_caps):
    """
    Build Oil ID from user-selected inputs.

    Examples:
        X_type="DP", percent_X=20, total_units_including_caps=24
            -> DPDM-020-024

        X_type="PM", percent_X=5, total_units_including_caps=180
            -> PMDM-005-180
    """

    X_type = str(X_type).strip().upper()

    if X_type not in ["DP", "PM"]:
        raise ValueError("Second repeat unit must be DP or PM.")

    percent_X = float(percent_X)
    total_units_including_caps = int(total_units_including_caps)

    if percent_X < 0 or percent_X > 100:
        raise ValueError("The DP/PM percentage must be between 0 and 100.")

    if total_units_including_caps <= CAP_UNITS:
        raise ValueError(
            "Total units must be greater than 2 because two units are cap-ending units."
        )

    family = f"{X_type}DM"
    percent_code = format_percent_code(percent_X)
    total_units_code = f"{total_units_including_caps:03d}"

    return f"{family}-{percent_code}-{total_units_code}"


def parse_oil_id(oil_id):
    """
    Parse oil IDs such as:
        DPDM-020-024
        PMDM-005-180

    Format:
        FAMILY-PERCENT-TOTALUNITS

    FAMILY:
        DPDM = diphenylsiloxane/dimethylsiloxane
        PMDM = phenylmethylsiloxane/dimethylsiloxane

    PERCENT:
        020 = 20% DP or PM
        005 = 5% DP or PM

    TOTALUNITS:
        total siloxane units including the two cap-ending units.
    """

    oil_id_clean = oil_id.strip().upper().replace("_", "-")

    pattern = r"^([A-Z]+)-(\d+(?:\.\d+)?)-(\d+)$"
    match = re.match(pattern, oil_id_clean)

    if match is None:
        raise ValueError(
            "Oil ID must follow the format DPDM-020-024 or PMDM-005-180."
        )

    family, percent_str, total_units_str = match.groups()

    if family == "DPDM":
        X_type = "DP"
    elif family == "PMDM":
        X_type = "PM"
    else:
        raise ValueError(
            "Oil family must be DPDM or PMDM. "
            "Examples: DPDM-020-024, PMDM-005-180."
        )

    percent_X = float(percent_str)
    total_units_including_caps = int(total_units_str)

    if percent_X < 0 or percent_X > 100:
        raise ValueError("The DP/PM percentage must be between 0 and 100.")

    if total_units_including_caps <= CAP_UNITS:
        raise ValueError(
            "Total units must be greater than 2 because two units are cap-ending units."
        )

    internal_repeat_units = total_units_including_caps - CAP_UNITS

    return {
        "Oil_ID_clean": oil_id_clean,
        "family": family,
        "X_type": X_type,
        "percent_X_input": percent_X,
        "percent_DM_input": 100.0 - percent_X,
        "total_units_including_caps": total_units_including_caps,
        "cap_units": CAP_UNITS,
        "internal_repeat_units": internal_repeat_units,
    }


def estimate_repeat_counts_from_oil_id(oil_id):
    """
    Estimate DM and DP/PM repeat-unit counts from the new Oil ID format.

    The percentage is applied to the total number of units including caps,
    following the requested naming convention.

    Example:
        DPDM-020-024
        total units = 24
        internal repeat units = 22
        n_DP = round(0.20 * 24) = 5
        n_DM = 22 - 5 = 17
    """

    parsed = parse_oil_id(oil_id)

    X_type = parsed["X_type"]
    percent_X = parsed["percent_X_input"]
    total_units = parsed["total_units_including_caps"]
    internal_repeat_units = parsed["internal_repeat_units"]

    # Apply percentage to total units including caps.
    n_X = round((percent_X / 100.0) * total_units)

    # X units cannot exceed internal repeat units.
    n_X = min(n_X, internal_repeat_units)

    n_DM = internal_repeat_units - n_X

    if n_DM < 0:
        raise ValueError(
            "Calculated DM units are negative. Check the percentage and total units."
        )

    achieved_backbone_MW = (
        n_DM * UNIT_MW["DM"] +
        n_X * UNIT_MW[X_type]
    )

    achieved_total_MW = achieved_backbone_MW + LEFT_END_MW + RIGHT_END_MW

    achieved_percent_X_total_units = 100.0 * n_X / total_units
    achieved_percent_DM_total_units = 100.0 * n_DM / total_units

    achieved_percent_X_repeat = 100.0 * n_X / internal_repeat_units
    achieved_percent_DM_repeat = 100.0 * n_DM / internal_repeat_units

    achieved_percent_DM_weight = (
        100.0 * n_DM * UNIT_MW["DM"] / achieved_backbone_MW
        if achieved_backbone_MW > 0 else 0.0
    )

    achieved_percent_X_weight = 100.0 - achieved_percent_DM_weight

    return {
        **parsed,
        "percent_basis": PERCENT_BASIS,
        "n_DM": n_DM,
        f"n_{X_type}": n_X,
        "n_total_repeat_units": n_DM + n_X,
        "achieved_backbone_MW": achieved_backbone_MW,
        "achieved_total_MW": achieved_total_MW,
        "achieved_percent_DM_total_units": achieved_percent_DM_total_units,
        f"achieved_percent_{X_type}_total_units": achieved_percent_X_total_units,
        "achieved_percent_DM_repeat": achieved_percent_DM_repeat,
        f"achieved_percent_{X_type}_repeat": achieved_percent_X_repeat,
        "achieved_percent_DM_weight": achieved_percent_DM_weight,
        f"achieved_percent_{X_type}_weight": achieved_percent_X_weight,
    }


# ============================================================
# 3D generation and UFF optimization
# ============================================================

def embed_3d_no_optimization(mol, random_seed=42):
    mol_3d = Chem.AddHs(mol)

    status = AllChem.EmbedMolecule(
        mol_3d,
        randomSeed=random_seed,
        useRandomCoords=True,
        maxAttempts=1000
    )

    if status != 0:
        raise RuntimeError(
            "3D embedding failed. Try a smaller molecule, different composition, or another random seed."
        )

    return mol_3d


def optimize_existing_3d_with_uff(mol_3d, max_iters=5000):
    mol_opt = Chem.Mol(mol_3d)

    if not AllChem.UFFHasAllMoleculeParams(mol_opt):
        st.warning("UFF parameters may be missing for some atoms.")

    ff = AllChem.UFFGetMoleculeForceField(mol_opt, confId=0)

    if ff is None:
        raise RuntimeError("Could not create UFF force field.")

    ff.Initialize()

    initial_energy = ff.CalcEnergy()
    status = ff.Minimize(maxIts=max_iters)
    final_energy = ff.CalcEnergy()

    converged = status == 0

    return mol_opt, initial_energy, final_energy, converged


def mol_to_xyz_string(mol, comment="Generated by Streamlit app"):
    conf = mol.GetConformer()
    lines = [str(mol.GetNumAtoms()), comment]

    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        lines.append(
            f"{atom.GetSymbol():<2} "
            f"{pos.x:12.6f} "
            f"{pos.y:12.6f} "
            f"{pos.z:12.6f}"
        )

    return "\n".join(lines)


def read_3d_sdf_upload(uploaded_file):
    """
    Read the first valid molecule from an uploaded SDF file.

    The uploaded SDF is used only to provide 3D coordinates for visualization
    and downloads when RDKit embedding fails for a large silicone oil.
    Prediction still uses the descriptor values calculated from the generated
    Oil-ID-based structure.
    """

    if uploaded_file is None:
        return None

    sdf_bytes = uploaded_file.getvalue()

    if len(sdf_bytes) == 0:
        raise ValueError("The uploaded SDF file is empty.")

    supplier = Chem.ForwardSDMolSupplier(
        BytesIO(sdf_bytes),
        sanitize=True,
        removeHs=False
    )

    mols = [mol for mol in supplier if mol is not None]

    if len(mols) == 0:
        raise ValueError("No valid molecule could be read from the uploaded SDF file.")

    mol = mols[0]

    if mol.GetNumConformers() == 0:
        raise ValueError("The uploaded SDF does not contain 3D coordinates.")

    conf = mol.GetConformer()

    if hasattr(conf, "Is3D") and not conf.Is3D():
        raise ValueError("The uploaded SDF has coordinates, but the conformer is not marked as 3D.")

    return mol


# ============================================================
# Public result tables
# ============================================================

def make_public_result_table(result):
    """
    Detailed output table for download.
    Internal model descriptors and MW difference are hidden.
    """

    hidden_keys = {
        "nAB_weighted",
        "F07_C_O_weighted",
        "MW_error"
    }

    rows = []

    for key, value in result.items():
        if key in hidden_keys:
            continue

        label = PUBLIC_LABELS.get(key, key)

        rows.append({
            "Property": label,
            "Value": value
        })

    return pd.DataFrame(rows)


def make_short_result_table(result):
    """
    Short table shown in the app.
    More detailed calculated values are only available in the downloadable CSV.
    """

    input_mode = result.get("input_mode", "generated_from_inputs")

    if input_mode == "uploaded_sdf_only":
        rows = [
            {
                "Property": "Structure source",
                "Value": "Uploaded 3D SDF"
            },
            {
                "Property": "Oil ID / sample name",
                "Value": result.get("Oil_ID", "Uploaded_SDF")
            },
            {
                "Property": "Uploaded SDF filename",
                "Value": result.get("uploaded_sdf_filename", "")
            },
            {
                "Property": "Oil loading (%)",
                "Value": result.get("oil_loading_percent", "")
            },
            {
                "Property": "Aromatic-bond descriptor count",
                "Value": result.get("nAB", "")
            },
            {
                "Property": "C–O topological descriptor count",
                "Value": result.get("F07_C_O", "")
            },
        ]

        return pd.DataFrame(rows)

    X_type = result["X_type"]

    rows = [
        {
            "Property": "Structure source",
            "Value": "Generated from input"
        },
        {
            "Property": "Oil ID",
            "Value": result["Oil_ID"]
        },
        {
            "Property": "Oil family",
            "Value": result["family"]
        },
        {
            "Property": "Second repeat unit",
            "Value": X_type
        },
        {
            "Property": f"Input {X_type} (%)",
            "Value": result["percent_X_input"]
        },
        {
            "Property": "Calculated DM (%)",
            "Value": result["percent_DM_input"]
        },
        {
            "Property": "Total units including caps",
            "Value": result["total_units_including_caps"]
        },
        {
            "Property": "Cap-ending units",
            "Value": result["cap_units"]
        },
        {
            "Property": "Internal repeat units",
            "Value": result["internal_repeat_units"]
        },
        {
            "Property": "Estimated DM units",
            "Value": result["n_DM"]
        },
        {
            "Property": f"Estimated {X_type} units",
            "Value": result[f"n_{X_type}"]
        },
        {
            "Property": "Estimated total MW",
            "Value": result["achieved_total_MW"]
        },
        {
            "Property": "Oil loading (%)",
            "Value": result["oil_loading_percent"]
        },
    ]

    return pd.DataFrame(rows)

# ============================================================
# Main generation function
# ============================================================

def generate_silicone_oil(
    oil_id,
    oil_loading_percent,
    random_seed,
    optimize_uff=False,
):
    """
    Generate the 2D silicone oil structure, descriptors, and optionally a 3D
    conformation. If 3D embedding or UFF optimization fails, the app keeps the
    descriptor result and lets the user upload a prepared 3D SDF file.
    """

    counts = estimate_repeat_counts_from_oil_id(oil_id)

    X_type = counts["X_type"]

    n_DM = counts["n_DM"]
    n_X = counts[f"n_{X_type}"]

    # Sequence mode fixed internally to even.
    sequence = make_even_sequence(n_DM, n_X, X_type)

    mol = build_silicone_oil(sequence)
    smiles = Chem.MolToSmiles(mol, canonical=True)

    nAB = calc_nAB(mol)
    F07_C_O = calc_F07_C_O(mol)

    oil_loading_fraction = oil_loading_percent / 100.0

    result = {
        "input_mode": "generated_from_inputs",
        "Oil_ID": oil_id,
        **counts,
        "oil_loading_percent": oil_loading_percent,
        "oil_loading_fraction": oil_loading_fraction,
        "sequence_mode": SEQUENCE_MODE,
        "sequence": "-".join(sequence),
        "SMILES": smiles,
        "nAB": nAB,
        "F07_C_O": F07_C_O,

        # Internal model descriptors — hidden from display and downloads.
        "nAB_weighted": nAB * oil_loading_fraction,
        "F07_C_O_weighted": F07_C_O * oil_loading_fraction,

        # 3D status fields.
        "3D_embedding_status": "not_attempted",
        "3D_source": "none",
        "3D_embedding_error": "",
        "uploaded_sdf_filename": "",
    }

    mol_3d = None

    try:
        mol_3d = embed_3d_no_optimization(mol, random_seed=random_seed)
        result["3D_embedding_status"] = "generated"
        result["3D_source"] = "RDKit embedding"

    except Exception as embed_error:
        result["3D_embedding_status"] = "failed"
        result["3D_source"] = "none"
        result["3D_embedding_error"] = str(embed_error)
        mol_3d = None

    if mol_3d is not None and optimize_uff:
        try:
            mol_3d, initial_E, final_E, converged = optimize_existing_3d_with_uff(mol_3d)

            result["UFF_initial_energy_kcal_mol"] = initial_E
            result["UFF_final_energy_kcal_mol"] = final_E
            result["UFF_energy_change_kcal_mol"] = final_E - initial_E
            result["UFF_converged"] = converged

            if result["3D_source"] == "RDKit embedding":
                result["3D_source"] = "RDKit embedding + UFF"

        except Exception as uff_error:
            result["UFF_converged"] = False
            result["UFF_error"] = str(uff_error)
            st.warning(
                "The 3D structure was embedded, but UFF optimization failed. "
                "The unoptimized 3D structure is still available."
            )

    return mol, mol_3d, result


# ============================================================
# Uploaded-SDF direct prediction helper
# ============================================================

def clean_identifier_for_filename(value, default="uploaded_sdf_oil"):
    """
    Clean a user-supplied name so it can be safely used in downloaded filenames.
    """

    value = str(value or "").strip()

    if value == "":
        value = default

    value = value.replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")

    return value or default


def smiles_from_mol(mol):
    """
    Generate a canonical SMILES string for an uploaded or generated molecule.
    Hydrogens are removed for a cleaner SMILES if possible.
    """

    try:
        mol_no_h = Chem.RemoveHs(mol)
        return Chem.MolToSmiles(mol_no_h, canonical=True)
    except Exception:
        return Chem.MolToSmiles(mol, canonical=True)


def generate_result_from_uploaded_sdf(
    uploaded_mol_3d,
    oil_id,
    oil_loading_percent,
    uploaded_filename="uploaded_structure.sdf",
):
    """
    Build the result dictionary directly from an uploaded 3D SDF.

    In this mode, no Oil-ID sequence reconstruction is required. The app reads
    the uploaded molecule, calculates the same descriptors used by the internal
    model, weights them by oil loading, and then allows prediction.
    """

    if uploaded_mol_3d is None:
        raise ValueError("No uploaded molecule was provided.")

    oil_id_clean = clean_identifier_for_filename(oil_id or Path(uploaded_filename).stem)
    oil_loading_fraction = oil_loading_percent / 100.0

    nAB = calc_nAB(uploaded_mol_3d)
    F07_C_O = calc_F07_C_O(uploaded_mol_3d)
    smiles = smiles_from_mol(uploaded_mol_3d)

    result = {
        "input_mode": "uploaded_sdf_only",
        "Oil_ID": oil_id_clean,
        "Oil_ID_clean": oil_id_clean,
        "family": "Uploaded SDF",
        "X_type": "from SDF",
        "percent_X_input": "",
        "percent_DM_input": "",
        "percent_basis": "uploaded_sdf_structure",
        "total_units_including_caps": "",
        "cap_units": "",
        "internal_repeat_units": "",
        "n_DM": "",
        "n_PM": "",
        "n_DP": "",
        "n_total_repeat_units": "",
        "achieved_backbone_MW": "",
        "achieved_total_MW": "",
        "achieved_percent_DM_total_units": "",
        "achieved_percent_PM_total_units": "",
        "achieved_percent_DP_total_units": "",
        "achieved_percent_DM_repeat": "",
        "achieved_percent_PM_repeat": "",
        "achieved_percent_DP_repeat": "",
        "achieved_percent_DM_weight": "",
        "achieved_percent_PM_weight": "",
        "achieved_percent_DP_weight": "",
        "oil_loading_percent": oil_loading_percent,
        "oil_loading_fraction": oil_loading_fraction,
        "sequence_mode": "uploaded_sdf",
        "sequence": "Uploaded 3D SDF; no Oil-ID sequence was generated.",
        "SMILES": smiles,
        "nAB": nAB,
        "F07_C_O": F07_C_O,
        "nAB_weighted": nAB * oil_loading_fraction,
        "F07_C_O_weighted": F07_C_O * oil_loading_fraction,
        "3D_embedding_status": "uploaded_sdf_direct",
        "3D_source": "uploaded SDF",
        "3D_embedding_error": "",
        "uploaded_sdf_filename": uploaded_filename,
    }

    return result

# ============================================================
# Model loading, scaling, and prediction
# ============================================================

@st.cache_data
def load_original_descriptor_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("The internal original descriptor data file was not found.")

    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model_from_folder():
    if MODEL_FILENAME is not None and str(MODEL_FILENAME).strip() != "":
        model_path = os.path.join(MODEL_FOLDER, MODEL_FILENAME)
    else:
        pkl_files = glob.glob(os.path.join(MODEL_FOLDER, "*.pkl"))

        if len(pkl_files) == 0:
            raise FileNotFoundError("The internal prediction model was not found.")

        model_path = pkl_files[0]

    loaded = joblib.load(model_path)

    if isinstance(loaded, dict):
        model = loaded.get("model", loaded.get("estimator", None))

        if model is None:
            raise ValueError("The internal model file could not be read correctly.")
    else:
        model = loaded

    return model


def prepare_internal_model_input(result):
    return pd.DataFrame([{
        "nAB_weighted": result["nAB_weighted"],
        "F07_C_O_weighted": result["F07_C_O_weighted"],
    }])


def normalize_and_predict_from_descriptors(result):
    df_data = load_original_descriptor_data()

    missing = [col for col in INTERNAL_MODEL_FEATURES if col not in df_data.columns]

    if missing:
        raise ValueError("The internal original data file is missing required model columns.")

    model = load_model_from_folder()

    X_original = df_data[INTERNAL_MODEL_FEATURES].copy()
    X_original = X_original.apply(pd.to_numeric, errors="coerce")

    if X_original.isnull().any().any():
        raise ValueError("The internal original descriptor data contains non-numeric or missing values.")

    scaler = MinMaxScaler()
    scaler.fit(X_original)

    X_new = prepare_internal_model_input(result)
    X_new_scaled = scaler.transform(X_new)

    try:
        prediction = model.predict(
            pd.DataFrame(X_new_scaled, columns=INTERNAL_MODEL_FEATURES)
        )[0]
    except Exception:
        prediction = model.predict(X_new_scaled)[0]

    return prediction


# ============================================================
# Prediction interpretation
# ============================================================

def interpret_prediction(raw_prediction):
    """
    Interpret raw model output for display and download.

    Negative ice-adhion values are physically unrealistic. The raw
    model output is kept for transparency, but the reported physical
    value is clipped at 0 kPa and flagged as extrapolation.
    """

    if raw_prediction is None:
        return None, None, "Prediction not available."

    raw_prediction = float(raw_prediction)

    if raw_prediction < 0:
        reported_prediction = 0.0
        prediction_note = (
            "Raw model output was negative and physically unrealistic. "
            "The reported physical value was set to 0 kPa for interpretation. "
            "This result should be treated as model extrapolation or outside the reliable prediction domain."
        )
    else:
        reported_prediction = raw_prediction
        prediction_note = "Prediction is within the physically meaningful range."

    return raw_prediction, reported_prediction, prediction_note


# ============================================================
# Session state
# ============================================================

if "mol" not in st.session_state:
    st.session_state.mol = None

if "mol_3d" not in st.session_state:
    st.session_state.mol_3d = None

if "result" not in st.session_state:
    st.session_state.result = None

if "prediction_value" not in st.session_state:
    st.session_state.prediction_value = None

if "uploaded_sdf_filename" not in st.session_state:
    st.session_state.uploaded_sdf_filename = None

if "last_structure_source_mode" not in st.session_state:
    st.session_state.last_structure_source_mode = None


# ============================================================
# Streamlit interface
# ============================================================

st.title("Silicone Oil Ice-Adhesion Predictor")

st.write(
    "Choose how to provide the silicone oil structure. You can either generate a representative structure "
    "from composition inputs or upload an existing 3D SDF file. The app calculates the descriptor values, "
    "weights them by oil loading, and runs the internally stored ice-adhesion prediction model."
)


# ============================================================
# Sidebar — structure source and input controls
# ============================================================

with st.sidebar:
    st.header("Structure source")

    structure_source_mode = st.radio(
        "How do you want to provide the structure?",
        options=["Generate from input", "Load 3D SDF"],
        index=0,
        help=(
            "Generate from input builds a representative silicone oil from DP/PM percentage and total siloxane units. "
            "Load 3D SDF reads an existing 3D structure and calculates the model descriptors directly from that file."
        )
    )

    if st.session_state.last_structure_source_mode != structure_source_mode:
        st.session_state.mol = None
        st.session_state.mol_3d = None
        st.session_state.result = None
        st.session_state.prediction_value = None
        st.session_state.uploaded_sdf_filename = None
        st.session_state.last_structure_source_mode = structure_source_mode

    st.header("Common prediction input")

    oil_loading_percent = st.number_input(
        "Oil loading (%)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=1.0
    )

    generate_button = False
    load_sdf_button = False
    uploaded_sdf_direct = None
    direct_sdf_oil_id = "Uploaded_SDF_oil"

    if structure_source_mode == "Generate from input":
        st.header("Generate from input")

        X_type = st.selectbox(
            "Phenyl-containing repeat unit",
            options=["DP", "PM"],
            index=0,
            help=(
                "DP = diphenylsiloxane unit. "
                "PM = phenylmethylsiloxane unit."
            )
        )

        percent_X = st.number_input(
            f"{X_type} (%)",
            min_value=0.0,
            max_value=100.0,
            value=20.0 if X_type == "DP" else 5.0,
            step=1.0,
            help=(
                "Enter the percentage of the selected DP or PM unit. "
                "The remaining percentage is assigned to dimethylsiloxane units."
            )
        )

        calculated_DM = 100.0 - percent_X

        total_units_including_caps = st.number_input(
            "Total siloxane units including caps",
            min_value=3,
            max_value=1000,
            value=24,
            step=1,
            help=(
                "Total number of siloxane units including the two cap-ending units. "
                "Example: 24 means 22 internal repeat units plus 2 cap-ending units."
            )
        )

        try:
            oil_id = build_oil_id_from_inputs(
                X_type=X_type,
                percent_X=percent_X,
                total_units_including_caps=total_units_including_caps
            )

            parsed_preview = parse_oil_id(oil_id)

            st.success("Oil ID generated successfully.")
            st.markdown(
                f"""
                <div style="background-color:#e8f4ff; padding:14px; border-radius:8px; border-left:6px solid #1f77b4;">
                    <div style="font-size:15px; font-weight:600;">Generated Oil ID</div>
                    <div style="font-size:28px; font-weight:800; font-family:monospace; margin-top:6px; margin-bottom:8px;">
                        {oil_id}
                    </div>
                    <div style="font-size:14px;">
                        Family: {parsed_preview['family']}<br>
                        Second repeat unit: {parsed_preview['X_type']}<br>
                        {parsed_preview['X_type']} (%): {parsed_preview['percent_X_input']:.2f}<br>
                        Calculated DM (%): {calculated_DM:.2f}<br>
                        Total units including caps: {parsed_preview['total_units_including_caps']}<br>
                        Internal repeat units: {parsed_preview['internal_repeat_units']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:
            oil_id = None
            st.warning(f"Oil ID could not be generated: {e}")

        st.caption("Composition basis: total units including cap-ending units")
        st.caption("Sequence mode: even")

        random_seed = st.number_input(
            "Random seed for 3D generation",
            min_value=1,
            max_value=999999,
            value=42,
            step=1
        )

        optimize_uff = st.checkbox(
            "Optimize 3D structure with UFF",
            value=False
        )

        st.caption(
            "Large structures such as PMDM-005-180 can be generated, but 3D embedding and UFF optimization may be slow. "
            "For large systems, keep UFF optimization disabled first. If 3D embedding fails, the generated Oil ID is still shown "
            "and a prepared 3D SDF can be uploaded in the main panel."
        )

        generate_button = st.button("1. Generate structure", disabled=(oil_id is None))

    else:
        st.header("Load 3D SDF")

        direct_sdf_oil_id = st.text_input(
            "Oil ID or sample name",
            value="Uploaded_SDF_oil",
            help="This name is used only for display and downloaded filenames."
        )

        uploaded_sdf_direct = st.file_uploader(
            "Upload 3D SDF structure",
            type=["sdf"],
            key="direct_sdf_upload",
            help=(
                "Upload a valid SDF file containing 3D coordinates. The app will calculate nAB and F07_C_O directly from this structure."
            )
        )

        st.caption(
            "In this mode, the app does not reconstruct the polymer sequence from DP/PM inputs. "
            "It uses the uploaded 3D SDF structure to calculate descriptors and run the prediction."
        )

        load_sdf_button = st.button(
            "1. Load SDF structure",
            disabled=(uploaded_sdf_direct is None)
        )


# ============================================================
# Step 1A — Generate structure from input
# ============================================================

if generate_button:
    st.session_state.prediction_value = None
    st.session_state.uploaded_sdf_filename = None

    try:
        if oil_id is None:
            raise ValueError("Oil ID was not generated. Check the input values.")

        mol, mol_3d, result = generate_silicone_oil(
            oil_id=oil_id,
            oil_loading_percent=oil_loading_percent,
            random_seed=int(random_seed),
            optimize_uff=optimize_uff,
        )

        st.session_state.mol = mol
        st.session_state.mol_3d = mol_3d
        st.session_state.result = result

        if mol_3d is None:
            st.warning(
                "The Oil ID, sequence, and descriptors were generated, but the 3D embedding failed. "
                "You can still review the generated descriptors. To visualize/download a 3D structure, upload a prepared 3D SDF below."
            )
        else:
            st.success("Structure generated successfully.")

    except Exception as e:
        st.error(f"Structure generation failed: {e}")


# ============================================================
# Step 1B — Load uploaded SDF directly
# ============================================================

if load_sdf_button:
    st.session_state.prediction_value = None

    try:
        uploaded_mol_3d = read_3d_sdf_upload(uploaded_sdf_direct)

        result = generate_result_from_uploaded_sdf(
            uploaded_mol_3d=uploaded_mol_3d,
            oil_id=direct_sdf_oil_id,
            oil_loading_percent=oil_loading_percent,
            uploaded_filename=uploaded_sdf_direct.name,
        )

        st.session_state.mol = uploaded_mol_3d
        st.session_state.mol_3d = uploaded_mol_3d
        st.session_state.result = result
        st.session_state.uploaded_sdf_filename = uploaded_sdf_direct.name

        st.success("Uploaded 3D SDF loaded successfully.")

    except Exception as e:
        st.error(f"Uploaded SDF could not be loaded: {e}")


# ============================================================
# Sidebar — Step 2 prediction button
# ============================================================

predict_button = False

if st.session_state.result is not None:
    with st.sidebar:
        st.header("Prediction")
        predict_button = st.button("2. Run prediction")


# ============================================================
# Show generated or uploaded structure and calculated values
# ============================================================

if st.session_state.result is not None:
    result = st.session_state.result
    mol_3d = st.session_state.mol_3d
    input_mode = result.get("input_mode", "generated_from_inputs")

    # --------------------------------------------------------
    # Descriptor counts only
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    col1.metric(
        "Aromatic-bond descriptor count",
        result["nAB"]
    )

    col2.metric(
        "C–O topological descriptor count",
        result["F07_C_O"]
    )

    # --------------------------------------------------------
    # Short visible table
    # --------------------------------------------------------

    st.subheader("Input summary")

    short_result_df = make_short_result_table(result)
    st.dataframe(short_result_df, use_container_width=True)

    st.caption(
        "A table with more detailed calculated values is available in the downloadable output CSV."
    )

    # --------------------------------------------------------
    # Representative sequence
    # --------------------------------------------------------

    if input_mode == "generated_from_inputs":
        st.subheader("Representative sequence")
        st.code(result["sequence"], language="text")
    else:
        st.subheader("Uploaded SDF mode")
        st.info(
            "The structure was loaded directly from a 3D SDF file. No Oil-ID-based representative sequence was generated. "
            "The prediction uses descriptor values calculated from the uploaded SDF."
        )

    # --------------------------------------------------------
    # 3D structure upload / fallback / visualization
    # --------------------------------------------------------

    st.subheader("3D structure")

    if input_mode == "generated_from_inputs":
        if result.get("3D_embedding_status") == "failed" and mol_3d is None:
            st.warning(
                "The app generated the Oil ID, sequence, and descriptors, but RDKit could not generate the embedded 3D structure. "
                "This can happen for large silicone oils such as PMDM-005-180."
            )

            st.info(
                f"Please contact the app maintainer and provide this generated Oil ID: "
                f"{result['Oil_ID_clean']}. A pre-generated 3D SDF can be prepared and uploaded below."
            )

            with st.expander("Show technical embedding error"):
                st.write(result.get("3D_embedding_error", "No error message available."))

        with st.expander("Optional: upload or replace 3D SDF for this generated oil", expanded=(mol_3d is None)):
            uploaded_sdf_fallback = st.file_uploader(
                "Upload a 3D SDF for this generated oil",
                type=["sdf"],
                key="fallback_sdf_upload",
                help=(
                    "Use this option if automatic 3D embedding fails or if you already have a better optimized 3D SDF. "
                    "The prediction descriptors remain based on the generated Oil-ID structure; the uploaded SDF is used for visualization and downloads."
                )
            )

            if uploaded_sdf_fallback is not None:
                try:
                    uploaded_mol_3d = read_3d_sdf_upload(uploaded_sdf_fallback)

                    st.session_state.mol_3d = uploaded_mol_3d
                    st.session_state.uploaded_sdf_filename = uploaded_sdf_fallback.name

                    st.session_state.result["3D_embedding_status"] = "uploaded_sdf"
                    st.session_state.result["3D_source"] = "uploaded SDF"
                    st.session_state.result["3D_embedding_error"] = ""
                    st.session_state.result["uploaded_sdf_filename"] = uploaded_sdf_fallback.name

                    result = st.session_state.result
                    mol_3d = st.session_state.mol_3d

                    st.success(f"Uploaded 3D SDF loaded successfully: {uploaded_sdf_fallback.name}")

                    uploaded_nAB = calc_nAB(mol_3d)
                    uploaded_F07_C_O = calc_F07_C_O(mol_3d)

                    if uploaded_nAB != result["nAB"] or uploaded_F07_C_O != result["F07_C_O"]:
                        st.warning(
                            "The uploaded SDF descriptor counts do not exactly match the generated Oil-ID structure. "
                            "Please confirm that the uploaded SDF corresponds to the same generated oil. "
                            f"Generated descriptors: nAB={result['nAB']}, F07_C_O={result['F07_C_O']}. "
                            f"Uploaded SDF descriptors: nAB={uploaded_nAB}, F07_C_O={uploaded_F07_C_O}."
                        )

                except Exception as e:
                    st.error(f"Uploaded SDF could not be read: {e}")

    if mol_3d is not None:
        mol_block = Chem.MolToMolBlock(mol_3d)

        viewer = py3Dmol.view(width=1000, height=650)
        viewer.addModel(mol_block, "mol")
        viewer.setStyle({
            "stick": {"radius": 0.15},
            "sphere": {"scale": 0.22}
        })
        viewer.setBackgroundColor("white")
        viewer.zoomTo()

        components.html(viewer._make_html(), height=680)

    else:
        st.info(
            "No 3D structure is currently available. Upload a 3D SDF file above to visualize the structure "
            "and download MOL/SDF/XYZ files. The prediction itself uses the calculated descriptors."
        )

    # ========================================================
    # Step 2 — Prediction
    # ========================================================

    st.subheader("Ice adhesion prediction")

    if predict_button:
        try:
            prediction_value = normalize_and_predict_from_descriptors(result)
            st.session_state.prediction_value = prediction_value

            st.success("Prediction completed.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    if st.session_state.prediction_value is not None:
        raw_prediction, reported_prediction, prediction_note = interpret_prediction(
            st.session_state.prediction_value
        )

        if raw_prediction < 0:
            st.metric(
                "Reported physical ice adhesion",
                f"{reported_prediction:.4f} kPa"
            )

            st.warning(
                f"The raw model output was {raw_prediction:.4f} kPa, which is physically unrealistic "
                "because ice adhesion cannot be negative.\n\n"
                "This result should be interpreted as model extrapolation or as a formulation "
                "outside the reliable prediction domain."
            )

            st.info(
                "For physical interpretation, the reported value is set to 0 kPa. "
                "The raw model output is kept in the prediction CSV for transparency."
            )

        else:
            st.metric(
                "Predicted ice adhesion",
                f"{reported_prediction:.4f} kPa"
            )

            st.info(
                "Prediction was calculated using the internal model and the original training-data scaling."
            )

    else:
        st.info("Use the left sidebar button to run the prediction.")

    # ========================================================
    # Downloads
    # ========================================================

    public_result_df = make_public_result_table(result)

    st.subheader("Download files")

    file_id = result.get("Oil_ID_clean", clean_identifier_for_filename(result.get("Oil_ID", "silicone_oil")))

    public_csv = public_result_df.to_csv(index=False).encode("utf-8")
    smi_data = f"{result['SMILES']}\t{result['Oil_ID']}\n".encode("utf-8")

    raw_prediction, reported_prediction, prediction_note = interpret_prediction(
        st.session_state.prediction_value
    )

    prediction_summary = pd.DataFrame([{
        "Structure source": "Uploaded 3D SDF" if input_mode == "uploaded_sdf_only" else "Generated from input",
        "Oil ID": result.get("Oil_ID", ""),
        "Cleaned Oil ID": result.get("Oil_ID_clean", ""),
        "Oil family": result.get("family", ""),
        "Second repeat unit": result.get("X_type", ""),
        "PM/DP input (%)": result.get("percent_X_input", ""),
        "DM input (%)": result.get("percent_DM_input", ""),
        "Total units including caps": result.get("total_units_including_caps", ""),
        "Cap-ending units": result.get("cap_units", ""),
        "Internal repeat units": result.get("internal_repeat_units", ""),
        "DM units": result.get("n_DM", ""),
        "PM units": result.get("n_PM", ""),
        "DP units": result.get("n_DP", ""),
        "Estimated total MW": result.get("achieved_total_MW", ""),
        "Oil loading (%)": result.get("oil_loading_percent", ""),
        "nAB": result.get("nAB", ""),
        "F07_C_O": result.get("F07_C_O", ""),
        "nAB_weighted": result.get("nAB_weighted", ""),
        "F07_C_O_weighted": result.get("F07_C_O_weighted", ""),
        "3D embedding status": result.get("3D_embedding_status", ""),
        "3D source": result.get("3D_source", ""),
        "Uploaded SDF filename": result.get("uploaded_sdf_filename", ""),
        "Raw model output (kPa)": raw_prediction,
        "Reported physical ice adhesion (kPa)": reported_prediction,
        "Prediction note": prediction_note,
    }])

    prediction_summary_csv = prediction_summary.to_csv(index=False).encode("utf-8")

    if mol_3d is not None:
        mol_data = Chem.MolToMolBlock(mol_3d, forceV3000=True).encode("utf-8")
        sdf_data = (
            Chem.MolToMolBlock(mol_3d, forceV3000=True) + "\n$$$$\n"
        ).encode("utf-8")
        xyz_data = mol_to_xyz_string(
            mol_3d,
            comment=f"{result['Oil_ID']}: generated or uploaded by Streamlit silicone-oil app"
        ).encode("utf-8")

        if input_mode == "generated_from_inputs":
            sequence_data = f"{result['sequence']}\n".encode("utf-8")
            d1, d2, d3, d4, d5, d6, d7 = st.columns(7)
        else:
            d1, d2, d4, d5, d6, d7 = st.columns(6)
            d3 = None

        d1.download_button(
            "Output CSV",
            public_csv,
            file_name=f"{file_id}_output_values.csv",
            mime="text/csv"
        )

        d2.download_button(
            "Prediction CSV",
            prediction_summary_csv,
            file_name=f"{file_id}_prediction_summary.csv",
            mime="text/csv"
        )

        if input_mode == "generated_from_inputs":
            d3.download_button(
                "Sequence",
                sequence_data,
                file_name=f"{file_id}_sequence.txt",
                mime="text/plain"
            )

        d4.download_button(
            "SMI",
            smi_data,
            file_name=f"{file_id}.smi",
            mime="text/plain"
        )

        d5.download_button(
            "MOL",
            mol_data,
            file_name=f"{file_id}_3D.mol",
            mime="text/plain"
        )

        d6.download_button(
            "SDF",
            sdf_data,
            file_name=f"{file_id}_3D.sdf",
            mime="text/plain"
        )

        d7.download_button(
            "XYZ",
            xyz_data,
            file_name=f"{file_id}_3D.xyz",
            mime="text/plain"
        )

    else:
        if input_mode == "generated_from_inputs":
            sequence_data = f"{result['sequence']}\n".encode("utf-8")
            d1, d2, d3, d4 = st.columns(4)
        else:
            d1, d2, d4 = st.columns(3)
            d3 = None

        d1.download_button(
            "Output CSV",
            public_csv,
            file_name=f"{file_id}_output_values.csv",
            mime="text/csv"
        )

        d2.download_button(
            "Prediction CSV",
            prediction_summary_csv,
            file_name=f"{file_id}_prediction_summary.csv",
            mime="text/csv"
        )

        if input_mode == "generated_from_inputs":
            d3.download_button(
                "Sequence",
                sequence_data,
                file_name=f"{file_id}_sequence.txt",
                mime="text/plain"
            )

        d4.download_button(
            "SMI",
            smi_data,
            file_name=f"{file_id}.smi",
            mime="text/plain"
        )

        st.caption(
            "MOL, SDF, and XYZ downloads are available only after a 3D structure has been generated or uploaded."
        )
