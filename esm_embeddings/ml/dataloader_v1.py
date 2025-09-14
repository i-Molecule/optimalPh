from pathlib import Path
from typing import List, Tuple, Union
import pandas as pd
import numpy as np

import torch
import esm
import fire

from scipy.special import softmax
from sklearn.decomposition import PCA

import gc

from tqdm import tqdm
from time import time

import os

from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap


def load_esm_model_v2():
    url = "tcp://localhost:23456"
    torch.distributed.init_process_group(
        backend="nccl", init_method=url, world_size=1, rank=0
    )

    # download model data from the hub
    model_name = "esm2_t33_650M_UR50D"
    model_data, regression_data = esm.pretrained._download_model_and_regression_data(
        model_name
    )

    # token_map = {'L': 0, 'A': 1, 'G': 2, 'V': 3, 'S': 4, 'E': 5, 'R': 6, 'T': 7, 'I': 8, 'D': 9, 'P': 10,
    #  'K': 11, 'Q': 12, 'N': 13, 'F': 14, 'Y': 15, 'M': 16, 'H': 17, 'W': 18, 'C': 19}

    # initialize the model with FSDP wrapper
    fsdp_params = dict(
        mixed_precision=True,
        flatten_parameters=True,
        state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
        cpu_offload=True,  # enable cpu offloading
    )
    with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
        model, vocab = esm.pretrained.load_model_and_alphabet_core(
            model_name, model_data, regression_data
        )
        batch_converter = vocab.get_batch_converter()
        model.eval()

        # Wrap each layer in FSDP separately
        for name, child in model.named_children():
            if name == "layers":
                for layer_name, layer in child.named_children():
                    wrapped_layer = wrap(layer)
                    setattr(child, layer_name, wrapped_layer)
        model = wrap(model)

    return model, batch_converter


def load_esm_model():

    token_map = {
        "L": 0,
        "A": 1,
        "G": 2,
        "V": 3,
        "S": 4,
        "E": 5,
        "R": 6,
        "T": 7,
        "I": 8,
        "D": 9,
        "P": 10,
        "K": 11,
        "Q": 12,
        "N": 13,
        "F": 14,
        "Y": 15,
        "M": 16,
        "H": 17,
        "W": 18,
        "C": 19,
    }

    t_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    return t_model, batch_converter


def prepare_batches(sequences, max_tokens_per_batch):

    sizes = [(len(s), i) for i, s in enumerate(sequences)]
    sizes.sort()
    batches = []
    buf = []
    max_len = 0

    def _flush_current_buf():
        nonlocal max_len, buf
        if len(buf) == 0:
            return
        batches.append(buf)
        buf = []
        max_len = 0

    for sz, i in sizes:
        if max(sz, max_len) * (len(buf) + 1) > max_tokens_per_batch:
            _flush_current_buf()
        max_len = max(max_len, sz)
        buf.append(i)
    _flush_current_buf()

    return batches


def load_dataset(fname: Union[str, os.PathLike]) -> pd.DataFrame:

    print("START DATALOADER")

    # STAY WITH SINGLE DATAFRAME
    # MAKE QUALITY CHECK

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input', '-i', type=str, help='input .csv file with the required_columns', default='./data/file1.csv')
    # args = parser.parse_args()

    # REQUIRED COLUMNS IN THE DATAFRAME
    required_columns = ["sequence", "mean_pH"]

    # if os.path.exists(args.input):
    # if args.input.endswith('.csv'):
    # input_dir, input_fn = os.path.split(args.input)
    # else:
    # raise ValueError(f'The input file should be .csv : {args.input}')
    # else:
    # raise ValueError(f'No such file: {args.input}')

    df = pd.read_csv(fname)
    if not set(required_columns).issubset(df.columns.to_list()):
        print("required columns: ", required_columns)
        print(df.columns)
        raise ValueError("the input dataframe does not contain required columns")

    print("END DATALOADER")

    return df


def get_esm_embeddings_batched(
    sequences: List[str],
    model,
    batch_converter,
    max_tokens_per_batch: int = 10000,
    use_cuda: bool = True,
) -> np.ndarray:
    model.eval()
    batches = prepare_batches(
        sequences, max_tokens_per_batch
    )  # group sequences of the comparable length into batches
    data = [(i, s) for i, s in enumerate(sequences)]  # assing index to each sequence

    data_loader = torch.utils.data.DataLoader(
        data, collate_fn=batch_converter, batch_sampler=batches
    )

    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    print(f"Embeddings will be calculated with device: {device}")

    embeddings = np.zeros(shape=(len(data), 1280))  # 1280 is embedding dimension
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in tqdm(
            enumerate(data_loader), total=len(batches)
        ):

            toks = toks.to(device)
            results = model(toks, repr_layers=[33])
            logits = (results["logits"].detach().cpu().numpy()[0,].T)[4:24, 1:-1]
            probability = softmax(logits, axis=0)
            results = results["representations"][33].detach().cpu().numpy()

            for r, s, l in zip(results, strs, labels):
                embeddings[l] = r[1 : len(s) + 1].mean(0)

            if (batch_idx % 10) == 0:

                del toks
                del results
                del probability
                del logits
                gc.collect()
                torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()

    return embeddings


def PCA_reduction(
    embeddings_train: np.ndarray, embeddings_test: np.ndarray, n: int = 32
) -> np.ndarray:
    reduction = PCA(n_components=n)
    reduction.fit(embeddings_train)
    return reduction.transform(embeddings_test)


def process_dataset(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    model, batch_converter = load_esm_model()
    sequences = df["sequence"].values
    y = df["mean_pH"].values
    embeddings = get_esm_embeddings_batched(sequences, model, batch_converter)
    return embeddings, y

def sequences2embeddings(sequences:List[str]) -> np.ndarray:
    model, batch_converter = load_esm_model()
    embeddings = get_esm_embeddings_batched(sequences, model, batch_converter)
    return embeddings

def main(
    input_csv: Union[str, os.PathLike],
    seq_col: str,
    target_col: str,
    output_emb: Union[str, os.PathLike],
) -> None:

    df = pd.read_csv(input_csv)
    sequences = df[seq_col].values.tolist()
    targets = df[target_col].values.tolist()

    model, batch_converter = load_esm_model()

    start = time()
    embeddings = get_esm_embeddings_batched(sequences, model, batch_converter)
    finish = time()

    print(f"Time elapsed: {round(finish - start, 3)} seconds")

    X = np.zeros(len(embeddings), dtype=[("X", "f4", (1280,)), ("y", "f4")])
    X["X"] = embeddings
    X["y"] = targets

    output_emb = Path(output_emb)
    output_emb.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_emb, X)


if __name__ == "__main__":
    fire.Fire(main)
