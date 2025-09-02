import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
import pickle
from rdkit import Chem
from rdkit.Chem import Descriptors
from padelpy import padeldescriptor
import warnings

# === Configurations === #
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
warnings.filterwarnings("ignore", message=".*physical cores < 1.*")
warnings.filterwarnings("ignore")

MODEL_PATH = "model/LGBM_model.pkl"
DESCRIPTORS_PATH = "model/selected_descriptors.txt"


# === Helper Functions === #
def calculate_rdkit_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return {name: func(mol) for name, func in Descriptors.descList}
    else:
        return {name: np.nan for name, func in Descriptors.descList}


def run_prediction(input_df):
    temp_dir = tempfile.mkdtemp(dir=os.getcwd())
    try:
        # ----- Validate Input Format ----- #
        if "Compound_ID" not in input_df.columns or "SMILES" not in input_df.columns:
            raise ValueError("Input must contain 'Compound_ID' and 'SMILES' columns.")

        # ----- Validate SMILES ----- #
        invalid_smiles = []
        for idx, smi in enumerate(input_df["SMILES"]):
            if not smi.strip():
                invalid_smiles.append((input_df.loc[idx, "Compound_ID"], "Empty SMILES"))
            elif Chem.MolFromSmiles(smi) is None:
                invalid_smiles.append((input_df.loc[idx, "Compound_ID"], smi))

        if invalid_smiles:
            raise ValueError(
                "Invalid SMILES found:\n"
                + "\n".join([f"{cid}: {smi}" for cid, smi in invalid_smiles])
            )

        # --------- PaDEL Descriptors ---------- #
        smiles_list = input_df["SMILES"].tolist()
        temp_smi_file = os.path.join(temp_dir, "temp_input.smi")
        input_df["SMILES"].to_csv(temp_smi_file, index=False, header=False)

        padel_output = os.path.join(temp_dir, "padel_descriptors.csv")
        padeldescriptor(
            mol_dir=temp_smi_file,
            d_file=padel_output,
            d_2d=True,
            d_3d=False,
            fingerprints=True,
            retainorder=True,
            threads=2,
        )

        df_padel = pd.read_csv(padel_output, encoding="ISO-8859-1")
        if "Compound_ID" not in df_padel.columns:
            df_padel.insert(0, "Compound_ID", input_df["Compound_ID"])

        # ---- RDKit Descriptors ---- #
        df_rdkit = pd.DataFrame([calculate_rdkit_descriptors(smi) for smi in smiles_list])
        df_rdkit.insert(0, "Compound_ID", input_df["Compound_ID"])

        # ---- Combine Descriptors ---- #
        with open(DESCRIPTORS_PATH, "r") as f:
            selected_descriptors = [line.strip() for line in f.readlines()]

        df_combined = pd.merge(df_padel, df_rdkit, on="Compound_ID", how="inner")

        missing_descriptors = [
            desc for desc in selected_descriptors if desc not in df_combined.columns
        ]
        if missing_descriptors:
            raise ValueError("Missing descriptors. Check input SMILES or model files.")

        df_selected = df_combined[["Compound_ID"] + selected_descriptors]

        # ---- Load Model ---- #
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        X_test = df_selected[selected_descriptors].replace([np.inf, -np.inf], np.nan)
        X_test = X_test.fillna(X_test.mean())
        X_test = np.clip(
            X_test, -np.finfo(np.float32).max, np.finfo(np.float32).max
        ).astype(np.float32)

        # ---- Predictions ---- #
        probs = model.predict_proba(X_test)
        preds = model.predict(X_test)

        df_selected["Prediction_Probability"] = probs[:, 1].round(2)
        df_selected["Prediction"] = [
            "Anticancer" if p == 1 else "Non-anticancer" for p in preds
        ]

        output_df = df_selected[["Compound_ID", "Prediction", "Prediction_Probability"]]
        return output_df

    finally:
        shutil.rmtree(temp_dir)


# === Streamlit UI === #
def main():
    st.set_page_config(page_title="ACLPred", layout="wide", page_icon="🧬")
    st.title("🧬 ACLPred")
    st.write("Predict anticancer ligands using a pre-trained ML model.")

    # Example input
#     example_text = """CHEMBL3301610,CCN1CCN(Cc2ccc(Nc3ncc(F)c(-c4cc(F)c5nc(C)n(C(C)C)c5c4)n3)nc2)CC1
# CHEMBL254328,C[C@]12CC[C@H](O)CC1=CC[C@@H]1[C@@H]2CC[C@]2(C)C(c3cccnc3)=CC[C@@H]12
# CHEMBL1399,CC(C)(C#N)c1cc(Cn2cncn2)cc(C(C)(C)C#N)c1"""
    example_text = """CHEMBL3301610,CCN1CCN(Cc2ccc(Nc3ncc(F)c(-c4cc(F)c5nc(C)n(C(C)C)c5c4)n3)nc2)CC1
CHEMBL254328,C[C@]12CC[C@H](O)CC1=CC[C@@H]1[C@@H]2CC[C@]2(C)C(c3cccnc3)=CC[C@@H]12
CHEMBL1399,CC(C)(C#N)c1cc(Cn2cncn2)cc(C(C)(C)C#N)c1
Nilotinib,CC1=CN(C=N1)C1=CC(NC(=O)C2=CC=C(C)C(NC3=NC=CC(=N3)C3=CN=CC=C3)=C2)=CC(=C1)C(F)(F)F
Aspirin,O=C(C)Oc1ccccc1C(=O)O
Nimbolide,CC1=C2C(CC1C3=COC=C3)OC4C2(C(C5(C6C4OC(=O)C6(C=CC5=O)C)C)CC(=O)OC)C
Nimbin,CC1=C2C(CC1C3=COC=C3)OC4C2(C(C5(C(C4OC(=O)C)C(C=CC5=O)(C)C(=O)OC)C)CC(=O)OC)C
Paracetamol,CC(=O)Nc1ccc(cc1)O
"""

    input_method = st.radio("Choose input method:", ["Paste SMILES", "Upload CSV"])

    df_input = None
    if input_method == "Paste SMILES":
        text_data = st.text_area("Enter data (Compound_ID,SMILES):", example_text, height=150)
        if text_data.strip():
            lines = text_data.strip().split("\n")
            compounds = [line.split(",") for line in lines if "," in line]
            df_input = pd.DataFrame(compounds, columns=["Compound_ID", "SMILES"])
    else:
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        if uploaded_file:
            df_input = pd.read_csv(uploaded_file)

    if df_input is not None:
        st.write("### Input Data Preview")
        st.dataframe(df_input.head())

        if st.button("🔮 Run Prediction"):
            with st.spinner("Running prediction..."):
                try:
                    output_df = run_prediction(df_input)
                    df["interpretation"] = df["Predictiin"].apply(lambda x: "Anticancer Compound" if x == "Anticancer" else "Not an Anticancer Agent")

                    st.success("Prediction completed successfully!")
                    st.dataframe(output_df)

                    # Download option
                    csv = output_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Results", csv, "Output_result.csv", "text/csv"
                    )
                except Exception as e:
                    st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
