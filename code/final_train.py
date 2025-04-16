import json
import os
import numpy as np

from typing import Union

import pandas as pd

from models import get_model

import fire
from pathlib import Path
import pickle

def read_json(fname: Union[str, os.PathLike]) -> dict:
    with open(fname, "r") as fin:
        d = json.load(fin)
    return d


def main(
    models: Union[str, os.PathLike],
    data: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    prefix: str,
    csv_input:Union[str, os.PathLike] = None
):

    models = read_json(models)

    data = np.load(data)
    X, y = data["X"], data["y"]
    if csv_input is not None:
        print(csv_input)
        df = pd.read_csv(csv_input)
        sequences = df['sequence'].values
        y = df['mean_pH'].values

    for i, m in models.items():

        model = get_model(**m)
        if m.get('type') == 'kmers':
            model.fit(sequences, y)
        else:
            model.fit(X, y)

        output_name = Path(output_dir).joinpath(f"{prefix}_{m.get("type")}")
        output_name.parent.mkdir(exist_ok=True, parents=True)
        try:
            with open(output_name, "wb") as fout:
                pickle.dump(model, fout)
        # model.save(output_name)
        except:
            print("error with saving")


if __name__ == "__main__":
    fire.Fire(main)
