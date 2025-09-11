import os
import shutil
import sys
import tempfile
from pathlib import Path
import io
import subprocess
from typing import Tuple, Union
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

# No path listing; models are chosen by type and loaded from ophnet_weights/


def numeric_or_none_only(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return True
    # For object/other dtypes: every non-null must be a number
    return s.dropna().map(lambda x: isinstance(x, numbers.Number)).all()


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
    st.set_page_config(page_title="Optimal pH Predictor", page_icon="🧪", layout="wide")
    st.title("🧪 Optimal pH Predictor")
    st.caption("Upload a CSV of sequences and download predictions.")

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
            uploaded_csv = st.file_uploader(
                "Upload CSV containing sequences",
                type=["csv"],
                accept_multiple_files=False,
            )
            # Allow selecting the sequence column from the uploaded CSV's headers
            seq_col = "sequence"
            if uploaded_csv is not None:
                try:
                    # Read only the header to list columns
                    header_df = pd.read_csv(io.BytesIO(uploaded_csv.getvalue()), nrows=0)
                    all_cols = header_df.columns.tolist()
                    if not all_cols:
                        st.warning("No columns found in the uploaded CSV header.")
                    else:
                        default_idx = all_cols.index("sequence") if "sequence" in all_cols else 0
                        seq_col = st.selectbox(
                            "Sequence column",
                            options=all_cols,
                            index=default_idx,
                        )
                except Exception as e:
                    st.warning(f"Couldn't read columns from CSV: {e}")
            else:
                # Fallback when no file is uploaded yet
                seq_col = st.text_input("Sequence column name", value="sequence")

        st.divider()
        run_btn = st.button("Run prediction", type="primary", use_container_width=True)

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
            pred_cols = ["y_pred_knn", "y_pred_xgboost"]
            numeric_cols = ["y_pred_knn", "y_pred_xgboost"]
            for c in pred_df.columns:
                if numeric_or_none_only(pred_df[c]):
                    numeric_cols.append(c)
            numeric_cols = list(set(numeric_cols))  # unique only

            if pred_cols:
                # st.subheader("Prediction Plots")
                # st.caption("Histograms per model and optional scatter for comparison.")

                # # Histograms for each prediction column
                # cols = st.columns(min(3, len(pred_cols)))
                # for i, c in enumerate(pred_cols):
                #     with cols[i % len(cols)]:
                #         vals = pred_df[c].dropna().to_numpy()
                #         if vals.size:
                #             counts, edges = np.histogram(vals, bins=len(pred_df))
                #             centers = (edges[:-1] + edges[1:]) / 2
                #             hist_df = pd.DataFrame({"bin": centers, "count": counts})
                #             hist_df["bin"] = hist_df["bin"].round(1)
                #             st.bar_chart(
                #                 hist_df.set_index("bin"),
                #                 x_label=f"{c}",
                #                 use_container_width=True,
                #             )
                #         else:
                #             st.info(f"No numeric data to plot for {c}.")

                # Scatter comparison if 2+ prediction columns exist
                if len(numeric_cols) >= 2:
                    st.write("")
                    x_col = st.selectbox("Axis X", numeric_cols, index=0, key="pred_x")
                    y_col = st.selectbox("Axis Y", numeric_cols, index=1, key="pred_y")
                    st.scatter_chart(
                        pred_df, x=x_col, y=y_col, use_container_width=True
                    )
            else:
                # Fallback: allow plotting any numeric column
                num_cols = pred_df.select_dtypes(include=[np.number]).columns.tolist()
                if num_cols:
                    st.subheader("Numeric Column Plot")
                    sel = st.selectbox("Select column", num_cols)
                    st.line_chart(pred_df[sel], use_container_width=True)


def rearrange_columns(df: pd.DataFrame, first_cols: list) -> pd.DataFrame:
    cols = df.columns.tolist()
    for c in reversed(first_cols):
        if c in cols:
            cols.insert(0, cols.pop(cols.index(c)))
    return df[cols]


if __name__ == "__main__":
    main()
