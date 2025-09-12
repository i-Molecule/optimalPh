import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
import io
import subprocess
from typing import List, Tuple, Union
import numbers

import pandas as pd
import streamlit as st
import numpy as np

# Repo paths
REPO_ROOT = Path(__file__).resolve().parent
CODE_DIR = REPO_ROOT / "code"
WEIGHTS_DIR = REPO_ROOT / "ophnet_weights"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from predict import predict_all_models
from fasta_utils import fasta_to_dataframe, looks_like_fasta

# Supported extensions (deduped constants)
FASTA_EXTS = (".fasta", ".fa", ".faa", ".fna")
UPLOAD_TYPES = ["csv", "fasta", "fa", "faa", "fna", "txt"]

standard_amino_acids = list("ACDEFGHIKLMNPQRSTVWY")


def _is_fasta_upload(filename: Union[str, os.PathLike], data: bytes) -> bool:
    name = str(filename).lower()
    return name.endswith(FASTA_EXTS) or looks_like_fasta(data)


def _to_input_df(
    filename: Union[str, os.PathLike], data: bytes, seq_col: str
) -> pd.DataFrame:
    if _is_fasta_upload(filename, data):
        df = fasta_to_dataframe(data, include_ids=True, seq_col=seq_col)
    else:
        df = pd.read_csv(io.BytesIO(data))
    return df


def numeric_or_none_only(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return True
    # For object/other dtypes: every non-null must be a number
    return s.dropna().map(lambda x: isinstance(x, numbers.Number)).all()


def validate_input_df(df: pd.DataFrame, seq_col: str) -> None:
    if df.empty:
        st.error("Uploaded input is empty.")
    if seq_col not in df.columns:
        st.error(f"Column `{seq_col}` not found in input.")

    sequenes = df[seq_col].values
    validate_sequences(sequenes)


def validate_sequences(sequences: List[str]) -> None:
    invalid_seqs = []
    starts_without_M = []
    for i, seq in enumerate(sequences):
        if not seq:
            invalid_seqs.append((i + 1, seq))  # 1-based line number
            continue
        if seq[0] != "M":
            starts_without_M.append((i + 1, seq))

        for aa in seq:
            if aa not in standard_amino_acids:
                invalid_seqs.append((i + 1, seq))
                break
    if invalid_seqs:
        st.warning(
            f"Found {len(invalid_seqs)} invalid sequences (empty or non-standard amino acids)."
        )
    if starts_without_M:
        st.warning(
            f"Found {len(starts_without_M)} sequences not starting with 'M' (Methionine)."
        )


def extract_header(file_bytes: bytes) -> Tuple[list, int]:
    header_df = pd.read_csv(io.BytesIO(file_bytes), nrows=0)
    all_cols = header_df.columns.tolist()
    if not all_cols:
        st.error("No columns found in the uploaded CSV.")
    default_idx = all_cols.index("sequence") if "sequence" in all_cols else 0
    return (all_cols, default_idx)


def rearrange_columns(df: pd.DataFrame, first_cols: List[str]) -> pd.DataFrame:
    cols = df.columns.tolist()
    for c in reversed(first_cols):
        if c in cols:
            cols.insert(0, cols.pop(cols.index(c)))
    return df[cols]


def parse_keyboard_input(text: str) -> pd.DataFrame:
    """Parse raw text input into a DataFrame, handling FASTA or raw sequences."""
    if looks_like_fasta(text):
        st.info("FASTA format detected in typed/pasted input.")
        df = fasta_to_dataframe(text, include_ids=True, seq_col="sequence")
    else:
        # Single raw sequence or comma/whitespace-separated
        seqs = [s.strip() for s in re.split(r"[\s,]+", text.strip()) if s.strip()]
        df = pd.DataFrame({"sequence": seqs})
    return df


def main():
    st.set_page_config(page_title="Optimal pH Predictor", page_icon="🧪", layout="wide")
    st.title("🧪 Optimal pH Predictor")
    st.caption("Upload protein sequences and download predictions.")

    # Keep predictions across reruns so widget changes don't hide results
    if "pred_df" not in st.session_state:
        st.session_state.pred_df = None

    # Two-column layout: left = links + image + inputs, right = results
    left_col, right_col = st.columns([1, 1])

    with left_col:
        # Top header: GitHub button and pipeline image
        repo_url = "https://github.com/i-Molecule/optimalPh"
        paper_url = "https://pubs.acs.org/doi/full/10.1021/acssynbio.4c00465"
        try:
            st.link_button("Open on GitHub", repo_url, use_container_width=True)
            st.link_button("Paper", paper_url, use_container_width=True)
        except Exception:
            # Fallback for older Streamlit versions
            st.markdown(f"[Open on GitHub]({repo_url})")
            st.markdown(f"[Paper]({paper_url})")

        img_path = REPO_ROOT / "pictures" / "img_pipeline.jpeg"
        if img_path.exists():
            st.image(str(img_path), caption="Pipeline", use_container_width=True)
        else:
            st.warning("Image not found at pictures/img_pipeline.jpeg")

        # Input controls below the header
        with st.expander("Input options", expanded=True):
            # Choose input mode: file upload or manual typing
            input_mode = st.radio(
                "Choose input method",
                ["Upload file", "Type/Paste sequence(s)"],
                horizontal=True,
            )

            # Prepare shared variables
            uploaded_file = None
            file_bytes = None
            seq_col = "sequence"
            is_fasta = False
            typed_text = None

            if input_mode == "Upload file":
                uploaded_file = st.file_uploader(
                    "Upload CSV or FASTA containing sequences",
                    type=UPLOAD_TYPES,
                    accept_multiple_files=False,
                )

                if uploaded_file is not None:
                    # Read once; reuse later to avoid duplication
                    file_bytes = uploaded_file.getvalue()
                    # Heuristic: by extension or content
                    is_fasta = _is_fasta_upload(uploaded_file.name, file_bytes)

                    if not is_fasta:
                        all_cols, default_idx = extract_header(file_bytes)
                        seq_col = st.selectbox(
                            "Sequence column",
                            options=all_cols,
                            index=default_idx,
                        )
                    else:
                        st.info(
                            "FASTA detected. Sequences will be loaded into the sequence column in a resulted CSV."
                        )
                        seq_col = "sequence"
                else:
                    # Fallback when no file is uploaded yet
                    seq_col = st.text_input(
                        "Sequence column name (CSV)", value="sequence"
                    )
            else:
                # Manual typing mode
                st.write(
                    "Enter one or more sequences. Accepts raw sequence or FASTA format."
                )
                typed_text = st.text_area(
                    "Type or paste sequence(s)",
                    height=140,
                    placeholder=(
                        "Example (single sequence):\n"
                        "MKTAYIAKQRQISFVKSHFSRQDILDLIK...\n\n"
                        "Or FASTA format (multiple):\n"
                        ">seq1\nMKTAYIAKQRQISFVKSHF...\n"
                        ">seq2\nMNNNKDIIAL..."
                    ),
                )
                seq_col = "sequence"

        st.divider()
        run_btn = st.button("Run prediction", type="primary", use_container_width=True)

    if run_btn:
        # Build input DataFrame based on the chosen mode
        if input_mode == "Upload file":
            if uploaded_file is None:
                st.error("Please upload an input CSV or FASTA.")
                st.stop()
            # Convert to DataFrame once; uses same detection as above
            # Reuse file_bytes from earlier block
            input_df = _to_input_df(uploaded_file.name, file_bytes, seq_col)
        else:
            # if not typed_text or not typed_text.strip():
            if not typed_text.strip():
                st.error("Please type or paste at least one sequence.")
                st.stop()
            input_df = parse_keyboard_input(typed_text)
            is_fasta = True  # ensure we write DataFrame later

        # Validate presence of the sequence column
        try:
            validate_input_df(input_df, seq_col)
        except Exception as e:
            st.error(str(e))
            st.stop()

        try:
            st.info("This may take several minutes.")
            with st.spinner("Running prediction…"):
                # Write to a temporary CSV so downstream code can read it multiple times
                tmp_dir = tempfile.mkdtemp(prefix="oph_pred_")
                tmp_input_csv = Path(tmp_dir) / "input.csv"

                # If user uploaded a non-FASTA CSV, keep original bytes; otherwise write DF
                if (
                    input_mode == "Upload file"
                    and not is_fasta
                    and file_bytes is not None
                ):
                    with open(tmp_input_csv, "wb") as fout:
                        fout.write(file_bytes)
                else:
                    input_df.to_csv(tmp_input_csv, index=False)

                pred_df = predict_all_models(str(tmp_input_csv), seq_col)

        except Exception as e:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            st.error(f"Prediction failed: {e}")
            st.stop()

        # Cleanup temp artifacts now that prediction is done
        shutil.rmtree(tmp_dir, ignore_errors=True)

        pred_df = rearrange_columns(pred_df, [seq_col, "y_pred_knn", "y_pred_xgboost"])
        # Save results for persistence across reruns
        st.session_state.pred_df = pred_df
        st.success("Prediction complete.")

    # Show results and quick plots (persist across reruns) on the right column
    if st.session_state.pred_df is not None:
        with right_col:
            pred_df = st.session_state.pred_df

            csv_bytes = pred_df.to_csv(index=False).encode("utf-8")

            # Right-align the download button within the right column
            st.markdown(
                """
                    <style>
                    div[data-testid="stDownloadButton"] > button {
                        background-color: #22c55e;
                        color: white;
                        border-color: #22c55e;
                    }
                    div[data-testid="stDownloadButton"] > button:hover {
                        background-color: #16a34a;
                        color: white;
                        border-color: #16a34a;
                    }
                    div[data-testid="stDownloadButton"] > button:focus:not(:active) {
                        box-shadow: 0 0 0 0.2rem rgba(34,197,94,0.35);
                    }
                    </style>
                    """,
                unsafe_allow_html=True,
            )
            st.download_button(
                label="Download predictions CSV",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.divider()
            st.subheader("Results Preview")
            st.dataframe(pred_df, use_container_width=True)

            # Quick visualization of prediction columns
            # Ensure prediction columns come first and keep a stable order
            numeric_cols = extract_numeric_cols(
                pred_df, always_numeric_cols=["y_pred_xgboost", "y_pred_knn"]
            )

            # Scatter comparison if 2+ prediction columns exist
            st.write("")
            plot_scatter_chart(pred_df, numeric_cols)


def extract_numeric_cols(
    df: pd.DataFrame, always_numeric_cols: list = ["y_pred_xgboost", "y_pred_knn"]
) -> list:
    numeric_cols = always_numeric_cols
    for c in df.columns:
        if c not in numeric_cols and numeric_or_none_only(df[c]):
            numeric_cols.append(c)
    return numeric_cols


def plot_scatter_chart(df: pd.DataFrame, numeric_cols: list) -> None:
    default_x_idx = (
        numeric_cols.index("y_pred_xgboost") if "y_pred_xgboost" in numeric_cols else 0
    )
    default_y_idx = (
        numeric_cols.index("y_pred_knn")
        if "y_pred_knn" in numeric_cols
        else (1 if len(numeric_cols) > 1 else 0)
    )
    x_col = st.selectbox("Axis X", numeric_cols, index=default_x_idx, key="pred_x")
    y_col = st.selectbox("Axis Y", numeric_cols, index=default_y_idx, key="pred_y")
    st.scatter_chart(df, x=x_col, y=y_col, use_container_width=True)


if __name__ == "__main__":
    main()
