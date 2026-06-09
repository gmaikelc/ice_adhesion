# -*- coding: utf-8 -*-
"""
Created on Thu Jun 4 8:41:37 2026

@author: Gerardo Casanola
"""


#%% Importing libraries


# app.py

import os
import glob
import base64
import joblib
from pathlib import Path

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
# Internal model feature names — used internally only
# Do not display these names in the user interface
# ============================================================

INTERNAL_MODEL_FEATURES = [
    "nAB_weighted",
    "F07_C_O_weighted"
]


# ============================================================
# User-facing labels
# ============================================================

PUBLIC_LABELS = {
    "Oil_ID": "Oil ID",
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
    "MW_error": "MW difference",
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
    "nAB": "Aromatic-bond descriptor",
    "F07_C_O": "C–O topological descriptor",
    "UFF_initial_energy_kcal_mol": "Initial UFF energy",
    "UFF_final_energy_kcal_mol": "Final UFF energy",
    "UFF_energy_change_kcal_mol": "UFF energy change",
    "UFF_converged": "UFF converged",
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
        raise ValueError("Second repeat unit must be PM or DP.")

    if percent_DM < 0 or percent_DM > 100:
        raise ValueError("DM percentage must be between 0 and 100.")

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
        raise ValueError("Composition basis must be repeat_fraction or weight_fraction.")

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


# ============================================================
# Public result table
# ============================================================

def make_public_result_table(result):
    """
    Create a user-facing table while hiding internal model descriptors.
    """

    hidden_keys = {
        "nAB_weighted",
        "F07_C_O_weighted"
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
        raise ValueError("Sequence mode must be even or block.")

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

        # Internal model descriptors — hidden from the app display
        "nAB_weighted": nAB * oil_loading_fraction,
        "F07_C_O_weighted": F07_C_O * oil_loading_fraction,
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
# Model loading, scaling, and prediction
# ============================================================

def load_original_descriptor_data(data_path="data/data.csv"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find original descriptor data: {data_path}")

    return pd.read_csv(data_path)


def load_model_from_folder(model_folder="model", model_filename=None):
    if model_filename is not None and model_filename.strip() != "":
        model_path = os.path.join(model_folder, model_filename)
    else:
        pkl_files = glob.glob(os.path.join(model_folder, "*.pkl"))

        if len(pkl_files) == 0:
            raise FileNotFoundError(f"No .pkl model found in folder: {model_folder}")

        model_path = pkl_files[0]

    loaded = joblib.load(model_path)

    if isinstance(loaded, dict):
        model = loaded.get("model", loaded.get("estimator", None))

        if model is None:
            raise ValueError(
                "The .pkl file is a dictionary, but no key named 'model' or 'estimator' was found."
            )
    else:
        model = loaded

    return model, model_path


def prepare_internal_model_input(result):
    """
    Build internal model input using hidden weighted descriptor names.
    """

    return pd.DataFrame([{
        "nAB_weighted": result["nAB_weighted"],
        "F07_C_O_weighted": result["F07_C_O_weighted"],
    }])


def normalize_and_predict_from_descriptors(
    result,
    data_path="data/data.csv",
    model_folder="model",
    model_filename=None,
):
    """
    1. Load original descriptor data.
    2. Fit MinMaxScaler on original internal descriptor columns.
    3. Normalize calculated internal descriptors with that scaler.
    4. Predict using the trained model.
    """

    df_data = load_original_descriptor_data(data_path)

    missing = [col for col in INTERNAL_MODEL_FEATURES if col not in df_data.columns]

    if missing:
        raise ValueError(
            "The original data file is missing required model descriptor columns."
        )

    model, model_path = load_model_from_folder(
        model_folder=model_folder,
        model_filename=model_filename
    )

    X_original = df_data[INTERNAL_MODEL_FEATURES].copy()
    X_original = X_original.apply(pd.to_numeric, errors="coerce")

    if X_original.isnull().any().any():
        raise ValueError(
            "The original descriptor data contains non-numeric or missing values."
        )

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

    return {
        "prediction": prediction,
        "model_path": model_path,
    }


# ============================================================
# Streamlit interface
# ============================================================

st.title("Silicone Oil Ice-Adhesion Predictor")

st.write(
    "Generate a representative silicone oil from total MW, DM percentage, and PM/DP chemistry. "
    "The app calculates connectivity-based descriptors, scales them internally using the original "
    "training data, and predicts ice adhesion."
)

with st.sidebar:
    st.header("Silicone oil input")

    oil_id = st.text_input("Oil ID", value="DP90_1600_5oil")

    total_MW = st.number_input(
        "Total silicone oil MW",
        min_value=200.0,
        max_value=100000.0,
        value=1600.0,
        step=100.0
    )

    percent_DM = st.number_input(
        "DM (%)",
        min_value=0.0,
        max_value=100.0,
        value=90.0,
        step=1.0
    )

    X_type = st.selectbox(
        "Second repeat unit",
        options=["PM", "DP"],
        index=1
    )

    percent_basis = st.selectbox(
        "Composition basis",
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

    st.header("Prediction settings")

    run_model_prediction = st.checkbox(
        "Run ice-adhesion prediction",
        value=True
    )

    data_path = st.text_input(
        "Original data path",
        value="data/data.csv"
    )

    model_folder = st.text_input(
        "Model folder",
        value="model"
    )

    model_filename = st.text_input(
        "Model file name",
        value="",
        help="Leave empty to use the first .pkl file in the model folder."
    )

    run_button = st.button("Generate and predict")


# ============================================================
# Run app
# ============================================================

if run_button:
    prediction_value = None

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

        st.success("Silicone oil generated successfully.")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Aromatic-bond descriptor", result["nAB"])
        col2.metric("C–O topological descriptor", result["F07_C_O"])
        col3.metric("Total repeat units", result["n_total"])
        col4.metric("MW difference", f"{result['MW_error']:.2f}")

        st.subheader("Calculated output values")

        public_result_df = make_public_result_table(result)
        st.dataframe(public_result_df, use_container_width=True)

        st.subheader("Representative SMILES")
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

        # ====================================================
        # Prediction
        # ====================================================

        if run_model_prediction:
            st.subheader("Ice adhesion prediction")

            try:
                prediction_output = normalize_and_predict_from_descriptors(
                    result=result,
                    data_path=data_path,
                    model_folder=model_folder,
                    model_filename=model_filename,
                )

                prediction_value = prediction_output["prediction"]

                st.success("Prediction completed.")

                st.metric(
                    "Predicted ice adhesion",
                    f"{prediction_value:.4f} kPa"
                )

                with st.expander("Model details"):
                    st.write("Model file loaded successfully.")
                    st.code(prediction_output["model_path"], language="text")
                    st.write(
                        "The calculated descriptors were internally scaled using "
                        "the original training-data range."
                    )

            except Exception as e:
                st.error(f"Model prediction failed: {e}")

        # ====================================================
        # Downloads
        # ====================================================

        st.subheader("Download files")

        public_csv = public_result_df.to_csv(index=False).encode("utf-8")
        smi_data = f"{result['SMILES']}\t{oil_id}\n".encode("utf-8")
        mol_data = Chem.MolToMolBlock(mol_3d, forceV3000=True).encode("utf-8")
        sdf_data = (
            Chem.MolToMolBlock(mol_3d, forceV3000=True) + "\n$$$$\n"
        ).encode("utf-8")
        xyz_data = mol_to_xyz_string(
            mol_3d,
            comment=f"{oil_id}: generated by Streamlit silicone-oil app"
        ).encode("utf-8")

        prediction_summary = pd.DataFrame([{
            "Oil ID": oil_id,
            "Total MW": total_MW,
            "DM (%)": percent_DM,
            "Second repeat unit": X_type,
            "Oil loading (%)": oil_loading_percent,
            "Predicted ice adhesion (kPa)": prediction_value,
        }])

        prediction_summary_csv = prediction_summary.to_csv(index=False).encode("utf-8")

        d1, d2, d3, d4, d5, d6 = st.columns(6)

        d1.download_button(
            "Output CSV",
            public_csv,
            file_name=f"{oil_id}_output_values.csv",
            mime="text/csv"
        )

        d2.download_button(
            "Prediction CSV",
            prediction_summary_csv,
            file_name=f"{oil_id}_prediction_summary.csv",
            mime="text/csv"
        )

        d3.download_button(
            "SMI",
            smi_data,
            file_name=f"{oil_id}.smi",
            mime="text/plain"
        )

        d4.download_button(
            "MOL",
            mol_data,
            file_name=f"{oil_id}_3D.mol",
            mime="text/plain"
        )

        d5.download_button(
            "SDF",
            sdf_data,
            file_name=f"{oil_id}_3D.sdf",
            mime="text/plain"
        )

        d6.download_button(
            "XYZ",
            xyz_data,
            file_name=f"{oil_id}_3D.xyz",
            mime="text/plain"
        )

    except Exception as e:
        st.error(f"Error: {e}")