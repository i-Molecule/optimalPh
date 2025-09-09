import os
import subprocess
import sys
import pickle
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st


# Repo paths
REPO_ROOT = Path(__file__).resolve().parent
CODE_DIR = REPO_ROOT / "code"
WEIGHTS_DIR = REPO_ROOT / "ophnet_weights"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


# No path listing; models are chosen by type and loaded from ophnet_weights/


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

    if run_btn:
        if uploaded_csv is None:
            st.error("Please upload an input CSV.")
            st.stop()

        # Read uploaded CSV into a DataFrame to validate before writing to temp file
        try:
            input_df = pd.read_csv(uploaded_csv)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()
        if seq_col not in input_df.columns:
            st.error(f"Column `{seq_col}` not found in the uploaded CSV.")
            st.stop()
        if input_df.empty:
            st.error("Uploaded CSV is empty.")
            st.stop()

        # Write inputs to temp files for predictor (expects file paths)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_in:
            input_df.to_csv(tmp_in.name, index=False)
            tmp_input_path = Path(tmp_in.name)
        # Resolve model weights path inside ophnet_weights/
        model_path = WEIGHTS_DIR / f"model_{model_type}"
        if not model_path.exists():
            st.error(
                f"Weights for '{model_type}' not found. Please add them to the app."
            )
            st.stop()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_out:
            tmp_output_path = Path(tmp_out.name)

        if model_type in {"knn", "xgboost"}:
            info = (
                "Using ESM embeddings backend. This may take several minutes, "
                "especially on CPU. A GPU is recommended."
            )
        elif model_type in {"kmers"}:
            info = "Using k-mer frequency features. This should be fast."

        st.info(info)

        command = "python3 code/predict.py --input_csv {} --seq_col {} --model_fname {} --output_csv {}".format(
            str(tmp_input_path),
            seq_col,
            str(model_path),
            str(tmp_output_path),
        )
        subprocess.call(command, shell=True)

        if model_type in {"knn", "xgboost"}:

            # Lazy import to avoid requiring heavy deps unless needed
            try:
                import predict as predictor  # type: ignore
            except Exception as e:
                st.error(
                    "Could not import predictor for ESM-based models.\n"
                    "Install required deps: torch, fair-esm, fairscale, xgboost, scikit-learn, pandas.\n"
                    f"Import error: {e}"
                )
                st.stop()

            try:
                with st.spinner("Running prediction…"):
                    predictor.predict(
                        input_csv=str(tmp_input_path),
                        seq_col=seq_col,
                        model_fname=str(model_path),
                        output_csv=str(tmp_output_path),
                    )
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                # Cleanup temp files
                for p in [tmp_input_path, tmp_output_path]:
                    try:
                        os.unlink(p)
                    except Exception:
                        pass
                st.stop()
        elif model_type in {"kmers"}:
            try:
                command = "python3 code/predict.py --input_csv {} --seq_col {} --model_fname {} --output_csv {}".format(
                str(tmp_input_path),
                seq_col,
                str(model_path),
                str(tmp_output_path),
                )
                with st.spinner("Running prediction…"):
                    subprocess.call(command, shell=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                # Cleanup temp files
                for p in [tmp_input_path, tmp_output_path]:
                    try:
                        os.unlink(p)
                    except Exception:
                        pass
                st.stop()
        else:
            st.error("Unsupported model type.")
            for p in [tmp_input_path, tmp_output_path]:
                try:
                    os.unlink(p)
                except Exception:
                    pass
            st.stop()

        # Load predictions and merge back with input
        try:
            pred_df = pd.read_csv(tmp_output_path)
        except Exception as e:
            st.error(f"Failed to read prediction output: {e}")
            st.stop()

        merged = pd.concat(
            [input_df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1
        )

        # Prepare download
        csv_bytes = merged.to_csv(index=False).encode("utf-8")
        st.success("Prediction complete.")
        st.download_button(
            label="Download predictions CSV",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Cleanup temp files (best-effort)
        for p in [tmp_input_path, tmp_output_path]:
            try:
                os.unlink(p)
            except Exception:
                pass


if __name__ == "__main__":
    main()
