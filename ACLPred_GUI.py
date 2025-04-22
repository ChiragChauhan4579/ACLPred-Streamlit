#!/usr/bin/env python
# coding: utf-8

# In[33]:

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import pandas as pd
import os
import tempfile
import shutil
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from padelpy import padeldescriptor
import warnings
import threading

os.environ["LOKY_MAX_CPU_COUNT"] = "1" 
warnings.filterwarnings("ignore", message=".*physical cores < 1.*")

warnings.filterwarnings("ignore")

MODEL_PATH = "model/LGBM_model.pkl"
DESCRIPTORS_PATH = "model/selected_descriptors.txt"

def calculate_rdkit_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return {name: func(mol) for name, func in Descriptors.descList}
    else:
        return {name: np.nan for name, func in Descriptors.descList}

def run_prediction(input_path, pasted_data, log_box, progress_bar, btn, root):
    def log(msg):
        root.after(0, lambda: (log_box.insert(tk.END, msg + "\n"), log_box.see(tk.END)))

    temp_dir = tempfile.mkdtemp(dir=os.getcwd())
    log(f"[INFO] Temporary directory: {temp_dir}")

    try:
        root.after(0, lambda: progress_bar.config(value=5))

        if input_path:
            df_smiles = pd.read_csv(input_path, encoding="ISO-8859-1")
        elif pasted_data.strip():
            lines = pasted_data.strip().split("\n")
            compounds = [line.split(",") for line in lines if "," in line]
            df_smiles = pd.DataFrame(compounds, columns=["Compound_ID", "SMILES"])
        else:
            raise ValueError("No input data provided.")

        # ----- Validate Input Format ----- #
        if "Compound_ID" not in df_smiles.columns or "SMILES" not in df_smiles.columns:
            raise ValueError("Input file must contain 'Compound_ID' and 'SMILES' columns.")

        # ----- Validate SMILES ----- #
        invalid_smiles = []
        for idx, smi in enumerate(df_smiles["SMILES"]):
            if not smi.strip():
                invalid_smiles.append((df_smiles.loc[idx, "Compound_ID"], "Empty SMILES"))
            elif Chem.MolFromSmiles(smi) is None:
                invalid_smiles.append((df_smiles.loc[idx, "Compound_ID"], smi))

        if invalid_smiles:
            msg = "[ERROR] Invalid SMILES found:\n"
            for cid, smi in invalid_smiles:
                msg += f"Compound_ID: {cid}, SMILES: {smi}\n"
            raise ValueError(msg)

        #---------PaDEL Descriptors-------------- #
        smiles_list = df_smiles["SMILES"].tolist()
        invalids = [smi for smi in smiles_list if Chem.MolFromSmiles(smi) is None]
        if invalids:
            log(f"[WARNING] Invalid SMILES detected: {invalids}")

        temp_smi_file = os.path.join(temp_dir, "temp_input.smi")
        df_smiles["SMILES"].to_csv(temp_smi_file, index=False, header=False)

        padel_output = os.path.join(temp_dir, "padel_descriptors.csv")
        log("[INFO] Calculating PaDEL descriptors...")

        try:
            padeldescriptor(
                mol_dir=temp_smi_file,
                d_file=padel_output,
                d_2d=True,
                d_3d=False,
                fingerprints=True,
                retainorder=True,
                threads=2
            )
        except Exception as e:
            raise RuntimeError(f"PaDEL descriptor calculation failed. Ensure Java is installed. Details: {e}")

        df_padel = pd.read_csv(padel_output, encoding="ISO-8859-1")
        if "Compound_ID" not in df_padel.columns:
            df_padel.insert(0, "Compound_ID", df_smiles["Compound_ID"])

        root.after(0, lambda: progress_bar.config(value=40))
        log("[INFO] Calculating RDKit descriptors...")
        df_rdkit = pd.DataFrame([calculate_rdkit_descriptors(smi) for smi in smiles_list])
        df_rdkit.insert(0, "Compound_ID", df_smiles["Compound_ID"])

        with open(DESCRIPTORS_PATH, "r") as f:
            selected_descriptors = [line.strip() for line in f.readlines()]

        df_combined = pd.merge(df_padel, df_rdkit, on="Compound_ID", how="inner")

        # Check if all required descriptors are present
        missing_descriptors = [desc for desc in selected_descriptors if desc not in df_combined.columns]
        if missing_descriptors:
            raise ValueError(
                f"Prediction failed due to invalid SMILES or wrong input format."
            )

        df_selected = df_combined[["Compound_ID"] + selected_descriptors]


        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        X_test = df_selected[selected_descriptors].replace([np.inf, -np.inf], np.nan)
        X_test = X_test.fillna(X_test.mean())
        X_test = np.clip(X_test, -np.finfo(np.float32).max, np.finfo(np.float32).max).astype(np.float32)

        root.after(0, lambda: progress_bar.config(value=70))
        log("[INFO] Running predictions...")
        probs = model.predict_proba(X_test)
        preds = model.predict(X_test)

        df_selected["Prediction_Probability"] = probs[:, 1].round(2)
        df_selected["Prediction"] = ["Anticancer" if p == 1 else "Non-anticancer" for p in preds]

        output_df = df_selected[["Compound_ID", "Prediction", "Prediction_Probability"]]
        output_path = os.path.join(os.getcwd(), "Output_result.csv")
        output_df.to_csv(output_path, index=False)

        root.after(0, lambda: progress_bar.config(value=100))
        log(f"[SUCCESS] Predictions saved to: {output_path}")

    except Exception as e:
        root.after(0, lambda: messagebox.showerror("Error", str(e)))
        log(f"[ERROR] {e}")
        root.after(0, lambda: progress_bar.config(value=0))
    finally:
        shutil.rmtree(temp_dir)
        log("[INFO] Cleaned up temporary files.")
        root.after(0, lambda: btn.config(state=tk.NORMAL))

def launch_gui():
    root = tk.Tk()
    root.title("ACLPred")
    root.configure(bg="#1e1e1e")
    root.geometry("860x730")

    # === Canvas and Scrollbars ===
    outer_frame = tk.Frame(root)
    outer_frame.pack(fill="both", expand=True)

    canvas = tk.Canvas(outer_frame, bg="#1e1e1e", highlightthickness=0)
    canvas.pack(side="left", fill="both", expand=True)

    v_scrollbar = tk.Scrollbar(outer_frame, orient="vertical", command=canvas.yview)
    v_scrollbar.pack(side="right", fill="y")

    h_scrollbar = tk.Scrollbar(root, orient="horizontal", command=canvas.xview)
    h_scrollbar.pack(side="bottom", fill="x")

    canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

    # === Scrollable Frame Inside Canvas ===
    scrollable_frame = tk.Frame(canvas, bg="#1e1e1e")
    window = canvas.create_window((0, 0), window=scrollable_frame, anchor="n")

    def on_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def on_canvas_resize(event):
        canvas.itemconfig(window, width=max(event.width, 860))  # keeps it centered or scrollable

    scrollable_frame.bind("<Configure>", on_configure)
    canvas.bind("<Configure>", on_canvas_resize)

    # === Mousewheel Scroll (Windows/macOS/Linux) ===
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def _on_shift_mousewheel(event):
        canvas.xview_scroll(int(-1*(event.delta/120)), "units")

    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    canvas.bind_all("<Shift-MouseWheel>", _on_shift_mousewheel)

    # === GUI Content Starts Here ===
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TLabel", background="#1e1e1e", foreground="white")
    style.configure("TButton", background="#333", foreground="white")
    style.configure("TProgressbar", troughcolor="#444", background="#00cc66", thickness=20)

    file_path_var = tk.StringVar()

    header = tk.Frame(scrollable_frame, bg="#1e1e1e")
    header.pack(pady=(15, 5))
    tk.Label(header, text="ACLPred", font=("Segoe UI", 22, "bold"), bg="#1e1e1e", fg="#00ff99").pack()
    tk.Label(header, text="Predict anticancer ligands using machine learning model",
             font=("Segoe UI", 12), bg="#1e1e1e", fg="#cccccc").pack()

    container = tk.Frame(scrollable_frame, bg="#1e1e1e")
    container.pack(pady=10)

    frame_input = tk.LabelFrame(container, text="1. Input Data", padx=10, pady=5, font=("Arial", 12),
                                bg="#1e1e1e", fg="white", width=800)
    frame_input.pack(padx=10, pady=10)

    tk.Label(frame_input, text="Paste SMILES data (Compound_ID,SMILES):", bg="#1e1e1e", fg="white").pack(anchor="w")

    text_paste = scrolledtext.ScrolledText(
        frame_input,
        height=7,
        width=80,
        bg="#2d2d2d",
        fg="white",
        insertbackground="white",
        wrap="none",
        font=("Courier New", 10)
    )
    text_paste.pack(anchor="w")

    def insert_example_data():
        example_text = """CHEMBL3301610,CCN1CCN(Cc2ccc(Nc3ncc(F)c(-c4cc(F)c5nc(C)n(C(C)C)c5c4)n3)nc2)CC1
CHEMBL254328,C[C@]12CC[C@H](O)CC1=CC[C@@H]1[C@@H]2CC[C@]2(C)C(c3cccnc3)=CC[C@@H]12
CHEMBL1399,CC(C)(C#N)c1cc(Cn2cncn2)cc(C(C)(C)C#N)c1"""
        text_paste.delete("1.0", tk.END)
        text_paste.insert(tk.END, example_text)

    tk.Button(frame_input, text="Example Input", command=insert_example_data,
              bg="#444", fg="white", font=("Arial", 9)).pack(anchor="w", pady=(2, 6))

    tk.Label(frame_input, text="────── OR ──────", bg="#1e1e1e", fg="#00cc99", font=("Arial", 12, "bold")).pack(pady=5)

    tk.Label(frame_input, text="Upload CSV file:", bg="#1e1e1e", fg="white").pack(anchor="w")
    input_row = tk.Frame(frame_input, bg="#1e1e1e")
    input_row.pack(anchor="w")
    entry_file = tk.Entry(input_row, textvariable=file_path_var, width=55, bg="#f0eded", fg="black", insertbackground="black")
    entry_file.pack(side="left", padx=(0, 5))
    tk.Button(input_row, text="Browse", command=lambda: file_path_var.set(filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])),
              bg="#444", fg="white").pack(side="left")

    frame_progress = tk.LabelFrame(container, text="2. Prediction Progress", padx=10, pady=5, font=("Arial", 12),
                                   bg="#1e1e1e", fg="white", width=800)
    frame_progress.pack(padx=10, pady=10)

    log_box = scrolledtext.ScrolledText(frame_progress, height=10, width=80, bg="#2d2d2d", fg="white", insertbackground="white")
    log_box.pack(pady=5)

    progress_bar = ttk.Progressbar(frame_progress, length=500, mode="determinate")
    progress_bar.pack(pady=5)

    btn_predict = tk.Button(scrollable_frame, text="Run Prediction", font=("Arial", 12, "bold"),
                            bg="#0078D7", fg="white", activebackground="#005999", width=12)
    btn_predict.pack(pady=15)

    def start_prediction():
        log_box.delete(1.0, tk.END)
        btn_predict.config(state=tk.DISABLED)
        threading.Thread(target=run_prediction,
                         args=(file_path_var.get(), text_paste.get("1.0", tk.END), log_box, progress_bar, btn_predict, root),
                         daemon=True).start()

    btn_predict.config(command=start_prediction)

    root.mainloop()

    root.mainloop()
if __name__ == "__main__":
    launch_gui()
    
    
    # In[ ]:
