import pickle
import pandas as pd
import json
import fire
from typing import Union
import numpy as np

import os
import sys
from pathlib import Path

filepath = Path(__file__).resolve().parent
esm_code = filepath.parent.joinpath("esm_embeddings/ml")
sys.path.append(str(esm_code))

from dataloader_v1 import process_dataset, sequences2embeddings


def dataset2embeddings(input_csv: Union[str, os.PathLike], seq_col: str) -> np.ndarray:
    df = pd.read_csv(input_csv)
    seqs = df[seq_col].values
    tmp_df = pd.DataFrame(None)
    tmp_df["sequence"] = seqs
    tmp_df["mean_pH"] = 0
    embeddings, _ = process_dataset(tmp_df)
    return embeddings


def predict(
    input_csv: Union[str, os.PathLike],
    seq_col: str,
    model_fname: Union[str, os.PathLike],
    output_csv: Union[str, os.PathLike],
) -> None:

    # load model
    with open(model_fname, "rb") as fin:
        model = pickle.load(fin)

    if model.__class__.__name__.lower() == "kmers":
        embeddings = pd.read_csv(input_csv)[seq_col].values
    else:
        embeddings = dataset2embeddings(input_csv, seq_col)

    # make predictions
    predictions = model.predict(embeddings)

    df = pd.DataFrame(None)
    df["y_pred"] = predictions.flatten().tolist()
    df.to_csv(output_csv)

    return


def predict_all_models(
    input_data: Union[str, os.PathLike, pd.DataFrame],
    seq_col: str,
) -> pd.DataFrame:
    """Take either a CSV file path or a DataFrame as input.
    Return a DataFrame with predictions from knn and xgboost models.
    """
    if isinstance(input_data, pd.DataFrame):
        sequences = input_data[seq_col].values.tolist()
        embeddings = sequences2embeddings(sequences)
        df = input_data
    elif isinstance(input_data, (str, os.PathLike)):
        embeddings = dataset2embeddings(input_data, seq_col)
        df = pd.read_csv(input_data)

    all_predictions = {}
    for model_name in ["knn", "xgboost"]:
        model_path = filepath.parent.joinpath("ophnet_weights", f"model_{model_name}")
        with open(model_path, "rb") as fin:
            model = pickle.load(fin)

        predictions = model.predict(embeddings)

        all_predictions[f"y_pred_{model_name}"] = predictions.flatten().tolist()

    for key, value in all_predictions.items():
        df[key] = value

    return df


if __name__ == "__main__":
    fire.Fire(predict)
