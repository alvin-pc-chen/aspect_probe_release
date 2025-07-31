import torch
import pandas as pd
from transformers import AutoTokenizer

import os
import pickle
from glob import glob


def make_probing_data(
    infile: str,
    hl_dir: str,
    outdir: str,
    model_name: str = "google-bert/bert-large-uncased",
    device: str = "auto",
):
    """
    Saves input and label tensors for probing experiments as
    `outdir`/layer`i`/position`j` where `j` spans the positions ±10 from the
    first token in the verb. Files are saved as `(ls, ts)` in `.pkl` format where
    `ls` and `ts` are lists of tensors of labels and embeddings. Depending on
    verb position, each pickled file may have lists of different length. Files
    should be loaded and put in the `CustomDataset` class from
    `src.probing.custom_dataset`.

    :param str infile: Path to `.tsv` file where sentences, labels, and verb
        character positions are stored. File should have columns "sentence",
        "label", and "start_char".
    :param str hl_dir: Path to directory where all hidden states are stored.
        Directory should be organized by the structure given in the
        `get_hidden_states` method.
    :param str outdir: Path to directory where output files are saved.
    :param str model_name: HuggingFace tokenizer to use for extracting positions.
        Use the same model used for `get_hidden_states`. Defaults to
        "bert-large-uncased".
    :param str device: Device used to process tensors.
    """

    # Load the data
    df = pd.read_csv(infile, sep="\t")
    df["label"] = df["label"].apply(lambda x: 1 if x == "DYNAMIC" else 0)
    labels = df["label"].to_list()
    texts = df["sentence"].to_list()
    spans = df["start_char"].to_list()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # For each input sentence, find valid token positions within range ±10
    # from the first token in the verb.
    layer_indices = {}
    for i in range(len(texts)):
        tokens = tokenizer(texts[i], return_tensors="pt")
        center = tokens.char_to_token(spans[i])
        for j in range(21):
            idx = center - 10 + j
            if idx < 0 or idx >= tokens.input_ids.shape[1]:
                continue
            if i not in layer_indices:
                layer_indices[i] = []
            layer_indices[i].append(idx)

    layer_paths = glob(f"{hl_dir}/layer*")
    layer_paths.sort(key=lambda x: int(x.split("/")[-1].split("layer")[-1]))

    for layer_count, layer_path in enumerate(layer_paths):
        outpath = f"{outdir}/layer{layer_count}/"
        os.makedirs(outpath, exist_ok=True)

        input_path = glob(f"{layer_path}/input*.pt")
        input_path.sort(
            key=lambda x: int(x.split("/")[-1].split("input")[1].split(".pt")[0])
        )

        # TODO does it matter if tensors are loaded onto device?
        tensors = [torch.load(t, map_location=device) for t in input_path]

        ts = [[] for _ in range(21)]
        ls = [[] for _ in range(21)]
        for idx, tensor in enumerate(tensors):
            tensor = tensor.squeeze(0)
            for i, j in enumerate(layer_indices[idx]):
                try:
                    ts[i].append(tensor[j])
                    ls[i].append(
                        torch.tensor((0, 1), dtype=torch.float16, device=device)
                        if labels[idx] == 1
                        else torch.tensor((1, 0), dtype=torch.float16, device=device)
                    )
                except IndexError:
                    continue
        for j in range(len(ts)):
            pickle.dump((ls[j], ts[j]), open(f"{outpath}/position{j}.pkl", "wb"))
