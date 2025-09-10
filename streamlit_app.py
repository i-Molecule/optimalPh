import os
import shutil
import sys
import tempfile
from pathlib import Path
import io
import subprocess
from typing import Tuple, Union

import pandas as pd
import streamlit as st


# Repo paths
REPO_ROOT = Path(__file__).resolve().parent
CODE_DIR = REPO_ROOT / "code"
WEIGHTS_DIR = REPO_ROOT / "ophnet_weights"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from predict import predict_all_models

# No path listing; models are chosen by type and loaded from ophnet_weights/


def validate_input_df(df: pd.DataFrame, seq_col: str) -> None:
    if df.empty:
        raise ValueError("Uploaded CSV is empty.")
    if seq_col not in df.columns:
        raise KeyError(f"Column `{seq_col}` not found in the uploaded CSV.")


def weights_path_for(model_type: str) -> Path:
    p = WEIGHTS_DIR / f"model_{model_type}"
    if not p.exists():
        raise FileNotFoundError(
            f"Weights for '{model_type}' not found. Please add them to the app."
        )
    return p


def main():
    st.set_page_config(
        page_title="Optimal pH Predictor", page_icon="🧪", layout="centered"
    )
    st.title("🧪 Optimal pH Predictor")
    st.caption("Upload a CSV of sequences, select a model, and download predictions.")

    with st.expander("Input options", expanded=True):
        uploaded_csv = st.file_uploader(
            "Upload CSV containing sequences", type=["csv"], accept_multiple_files=False
        )
        seq_col = st.text_input("Sequence column name", value="sequence")
    
    st.divider()
    col_left, col_right = st.columns([1, 1])
    with col_left:
        run_btn = st.button("Run prediction", type="primary", use_container_width=True)
    with col_right:
        st.write("")
        st.write("")

    tmp_dir = None
    if run_btn:
        if uploaded_csv is None:
            st.error("Please upload an input CSV.")
            st.stop()
        # Read uploaded file once into memory and validate it
        file_bytes = uploaded_csv.getvalue()
        input_df = pd.read_csv(io.BytesIO(file_bytes))
        validate_input_df(input_df, seq_col)

        try:
            st.info("This may take several minutes.")
            with st.spinner("Running prediction…"):
                # Write to a temporary CSV so downstream code can read it multiple times
                tmp_dir = tempfile.mkdtemp(prefix="oph_pred_")
                tmp_input_csv = Path(tmp_dir) / "input.csv"
                with open(tmp_input_csv, "wb") as fout:
                    fout.write(file_bytes)
                pred_df = predict_all_models(str(tmp_input_csv), seq_col)
                
        except Exception as e:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            st.error(f"Prediction failed: {e}")
            st.stop()


        csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
        st.success("Prediction complete.")
        st.download_button(
            label="Download predictions CSV",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if tmp_dir is not None:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
