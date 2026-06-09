# -*- coding: utf-8 -*-
"""
Created on Fri July 19 17:41:37 2024

@author: Gerardo Casanola
"""


#%% Importing libraries

from pathlib import Path
import pandas as pd
import pickle
from molvs import Standardizer
from rdkit import Chem
from openbabel import openbabel
from mordred import Calculator, descriptors
from multiprocessing import freeze_support
import numpy as np
from rdkit.Chem import AllChem
import plotly.graph_objects as go
import networkx as nx

#Import Libraries
import math 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

# packages for streamlit
import streamlit as st
from PIL import Image
import io
import base64

from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, rdFingerprintGenerator, Descriptors, Draw
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.DataStructs import cDataStructs
from io import StringIO
from mordred import Calculator, descriptors
import numpy as np
import pandas as pd
import seaborn as sns
import sys, os, shutil
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_ketcher import st_ketcher
import time
import subprocess
from PIL import Image
import uuid
from filelock import Timeout, FileLock

#%% PAGE CONFIG

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='Coating perfomance predictor', page_icon=":computer:", layout='wide')

######
# Function to put a picture as header   
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

image = Image.open('cropped_header2.png')
st.image(image)

# app.py

import io
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol


# ============================================================
# Constants
# ============================================================

UNIT_MW = {
    "DM": 76.17,      # dimethylsiloxane repeat unit
    "PM": 136.241,   # phenylmethylsiloxane repeat unit
    "DP": 200.312,   # diphenylsiloxane repeat unit
}

LEFT_END_MW = 74.198
RIGHT_END_MW = 90.197


# ============================================================
# Helper functions for molecule construction
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
# Repeat-count estimation
# ============================================================

def estimate_repeat_counts(
    total_MW,
    percent_DM,
    X_type,
    percent_basis="repeat_fraction",
    left_end_mw=LEFT_END_MW,
    right_end_mw=RIGHT_END_MW,
):
    X_type = X_type.upper()

    if X_type not in ["PM", "DP"]:
        raise ValueError("X_type must be PM or DP.")

    if percent_DM < 0 or percent_DM > 100:
        raise ValueError("percent_DM must be between 0 and 100.")

    f_DM = percent_DM / 100.0
    f_X = 1.0 - f_DM

    backbone_MW = total_MW - left_end_mw - right_end_mw

    if backbone_MW <= 0:
        raise ValueError("Backbone MW is <= 0. Check total MW and end-group masses.")

    if percent_basis == "repeat_fraction":
        avg_repeat_MW = f_DM * UNIT_MW["DM"] + f_X * UNIT_MW[X_type]
        n_total_float = backbone_MW / avg_repeat_MW

        n_total = max(1, round(n_total_float))
        n_DM = round(f_DM * n_total)
        n_X = n_total - n_DM

    elif percent_basis == "weight_fraction":
        mass_DM = f_DM * backbone_MW
        mass_X = f_X * backbone_MW

        n_DM = max(0, round(mass_DM / UNIT_MW["DM"]))
        n_X = max(0, round(mass_X / UNIT_MW[X_type]))
        n_total = n_DM + n_X

        if n_total == 0:
            raise ValueError("Estimated zero repeat units. Check input MW.")

    else:
        raise ValueError("percent_basis must be repeat_fraction or weight_fraction.")

    achieved_backbone_MW = (
        n_DM * UNIT_MW["DM"] +
        n_X * UNIT_MW[X_type]
    )

    achieved_total_MW = achieved_backbone_MW + left_end_mw + right_end_mw

    achieved_percent_DM_repeat = 100.0 * n_DM / (n_DM + n_X)
    achieved_percent_X_repeat = 100.0 - achieved_percent_DM_repeat

    achieved_percent_DM_weight = (
        100.0 * n_DM * UNIT_MW["DM"] / achieved_backbone_MW
    )
    achieved_percent_X_weight = 100.0 - achieved_percent_DM_weight

    return {
        "total_MW_input": total_MW,
        "backbone_MW_input": backbone_MW,
        "X_type": X_type,
        "percent_DM_input": percent_DM,
        "percent_X_input": 100.0 - percent_DM,
        "percent_basis": percent_basis,
        "n_DM": n_DM,
        f"n_{X_type}": n_X,
        "n_total": n_DM + n_X,
        "achieved_backbone_MW": achieved_backbone_MW,
        "achieved_total_MW": achieved_total_MW,
        "MW_error": achieved_total_MW - total_MW,
        "achieved_percent_DM_repeat": achieved_percent_DM_repeat,
        f"achieved_percent_{X_type}_repeat": achieved_percent_X_repeat,
        "achieved_percent_DM_weight": achieved_percent_DM_weight,
        f"achieved_percent_{X_type}_weight": achieved_percent_X_weight,
    }


# ============================================================
# 3D generation
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


# ============================================================
# Main generation function
# ============================================================

def generate_silicone_oil(
    oil_id,
    total_MW,
    percent_DM,
    X_type,
    oil_loading_percent,
    percent_basis,
    sequence_mode,
    random_seed,
    optimize_uff=False,
):
    X_type = X_type.upper()

    counts = estimate_repeat_counts(
        total_MW=total_MW,
        percent_DM=percent_DM,
        X_type=X_type,
        percent_basis=percent_basis,
    )

    n_DM = counts["n_DM"]
    n_X = counts[f"n_{X_type}"]

    if sequence_mode == "even":
        sequence = make_even_sequence(n_DM, n_X, X_type)

    elif sequence_mode == "block":
        sequence = ["DM"] * n_DM + [X_type] * n_X

    else:
        raise ValueError("sequence_mode must be even or block.")

    mol = build_silicone_oil(sequence)
    smiles = Chem.MolToSmiles(mol, canonical=True)

    nAB = calc_nAB(mol)
    F07_C_O = calc_F07_C_O(mol)

    oil_loading_fraction = oil_loading_percent / 100.0

    result = {
        "Oil_ID": oil_id,
        **counts,
        "oil_loading_percent": oil_loading_percent,
        "oil_loading_fraction": oil_loading_fraction,
        "sequence_mode": sequence_mode,
        "sequence": "-".join(sequence),
        "SMILES": smiles,
        "nAB": nAB,
        "F07_C_O": F07_C_O,
        "nAB_loading_weighted": nAB * oil_loading_fraction,
        "F07_C_O_loading_weighted": F07_C_O * oil_loading_fraction,
    }

    mol_3d = embed_3d_no_optimization(mol, random_seed=random_seed)

    if optimize_uff:
        mol_3d, initial_E, final_E, converged = optimize_existing_3d_with_uff(mol_3d)

        result["UFF_initial_energy_kcal_mol"] = initial_E
        result["UFF_final_energy_kcal_mol"] = final_E
        result["UFF_energy_change_kcal_mol"] = final_E - initial_E
        result["UFF_converged"] = converged

    return mol, mol_3d, result


# ============================================================
# Streamlit interface
# ============================================================

st.set_page_config(
    page_title="Silicone Oil Descriptor Generator",
    layout="wide"
)

st.title("Silicone Oil Descriptor Generator")
st.write(
    "Generate a representative silicone oil from total MW, %DM, and PM/DP type. "
    "The app calculates nAB and F07[C-O] from graph connectivity and can generate a 3D structure."
)

with st.sidebar:
    st.header("Input parameters")

    oil_id = st.text_input("Oil ID", value="DP90_1600_5oil")

    total_MW = st.number_input(
        "Total silicone oil MW",
        min_value=200.0,
        max_value=100000.0,
        value=1600.0,
        step=100.0
    )

    percent_DM = st.number_input(
        "% DM",
        min_value=0.0,
        max_value=100.0,
        value=90.0,
        step=1.0
    )

    X_type = st.selectbox(
        "Second unit type",
        options=["PM", "DP"],
        index=1
    )

    percent_basis = st.selectbox(
        "Basis for %DM",
        options=["repeat_fraction", "weight_fraction"],
        index=1
    )

    sequence_mode = st.selectbox(
        "Sequence mode",
        options=["even", "block"],
        index=0
    )

    oil_loading_percent = st.number_input(
        "Oil loading (%)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=1.0
    )

    random_seed = st.number_input(
        "Random seed for 3D embedding",
        min_value=1,
        max_value=999999,
        value=42,
        step=1
    )

    optimize_uff = st.checkbox(
        "Optimize generated 3D structure with UFF",
        value=False
    )

    run_button = st.button("Generate silicone oil")


if run_button:
    try:
        mol, mol_3d, result = generate_silicone_oil(
            oil_id=oil_id,
            total_MW=total_MW,
            percent_DM=percent_DM,
            X_type=X_type,
            oil_loading_percent=oil_loading_percent,
            percent_basis=percent_basis,
            sequence_mode=sequence_mode,
            random_seed=int(random_seed),
            optimize_uff=optimize_uff,
        )

        df_result = pd.DataFrame([result])

        st.success("Silicone oil generated successfully.")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("nAB", result["nAB"])
        col2.metric("F07[C-O]", result["F07_C_O"])
        col3.metric("n total", result["n_total"])
        col4.metric("MW error", f"{result['MW_error']:.2f}")

        st.subheader("Calculated output values")
        st.dataframe(df_result.T, use_container_width=True)

        st.subheader("SMILES")
        st.code(result["SMILES"], language="text")

        st.subheader("3D structure")

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

        # Files for download
        csv_data = df_result.to_csv(index=False).encode("utf-8")
        smi_data = f"{result['SMILES']}\t{oil_id}\n".encode("utf-8")
        mol_data = Chem.MolToMolBlock(mol_3d, forceV3000=True).encode("utf-8")
        sdf_data = (Chem.MolToMolBlock(mol_3d, forceV3000=True) + "\n$$$$\n").encode("utf-8")
        xyz_data = mol_to_xyz_string(
            mol_3d,
            comment=f"{oil_id}: generated by Streamlit silicone oil app"
        ).encode("utf-8")

        st.subheader("Download files")

        d1, d2, d3, d4, d5 = st.columns(5)

        d1.download_button(
            "CSV",
            csv_data,
            file_name=f"{oil_id}_descriptors.csv",
            mime="text/csv"
        )

        d2.download_button(
            "SMI",
            smi_data,
            file_name=f"{oil_id}.smi",
            mime="text/plain"
        )

        d3.download_button(
            "MOL",
            mol_data,
            file_name=f"{oil_id}_3D.mol",
            mime="text/plain"
        )

        d4.download_button(
            "SDF",
            sdf_data,
            file_name=f"{oil_id}_3D.sdf",
            mime="text/plain"
        )

        d5.download_button(
            "XYZ",
            xyz_data,
            file_name=f"{oil_id}_3D.xyz",
            mime="text/plain"
        )

    except Exception as e:
        st.error(f"Error: {e}")
