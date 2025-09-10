import os
import shutil
import sys
import tempfile
from pathlib import Path
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


def predict_kmers(
    input_csv_path: Path, seq_col: str, model_path: Path, output_csv_path: Path
) -> pd.DataFrame:

    command = [
        "python3",
        str(CODE_DIR / "predict.py"),
        "--input_csv",
        str(input_csv_path),
        "--seq_col",
        seq_col,
        "--model_fname",
        str(model_path),
        "--output_csv",
        str(output_csv_path),
    ]
    subprocess.run(command, check=True)
    return pd.read_csv(output_csv_path)


def predict_esm_backed(
    input_csv_path: Path, seq_col: str, model_path: Path, output_csv_path: Path
) -> None:
    # Lazy import heavy deps only when needed
    import predict as predictor  # type: ignore

    predictor.predict(
        input_csv=str(input_csv_path),
        seq_col=seq_col,
        model_fname=str(model_path),
        output_csv=str(output_csv_path),
    )


def run_prediction(
    input_df: pd.DataFrame, seq_col: str, model_type: str
) -> Tuple[pd.DataFrame, Union[str, os.PathLike]]:
    validate_input_df(input_df, seq_col)
    model_path = weights_path_for(model_type)

    with tempfile.TemporaryDirectory(delete=False) as tmpdir:
        tmpdir_path = Path(tmpdir)
        tmp_input = tmpdir_path / "input.csv"
        tmp_output = tmpdir_path / "output.csv"
        input_df.to_csv(tmp_input, index=False)

        if model_type == "kmers":
            df = predict_kmers(tmp_input, seq_col, model_path, tmp_output)
        else:
            df = predict_esm_backed(tmp_input, seq_col, model_path, tmp_output)

    return pd.read_csv(tmp_output), tmpdir_path


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

    model_type = st.selectbox(
        "Model",
        options=["kmers", "knn", "xgboost"],
        index=0,
        help="Models are loaded from bundled weights.",
    )

    st.divider()
    col_left, col_right = st.columns([1, 1])
    with col_left:
        run_btn = st.button("Run prediction", type="primary", use_container_width=True)
    with col_right:
        st.write("")
        st.write("")
        st.caption("Note: ESM-based models (knn/xgboost) can be slow on CPU.")

    tmp_dir = None
    if run_btn:
        if uploaded_csv is None:
            st.error("Please upload an input CSV.")
            st.stop()

        try:
            input_df = pd.read_csv(uploaded_csv)
            if model_type in {"knn", "xgboost"}:
                st.info(
                    "Using ESM embeddings backend. This may take several minutes, especially on CPU."
                )
            with st.spinner("Running prediction…"):
                pred_df, tmp_dir = run_prediction(input_df, seq_col, model_type)
        except Exception as e:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            st.error(f"Prediction failed: {e}")
            st.stop()

        merged = pd.concat(
            [input_df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1
        )

        csv_bytes = merged.to_csv(index=False).encode("utf-8")
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
