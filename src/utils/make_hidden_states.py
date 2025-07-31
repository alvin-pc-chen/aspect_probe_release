import tqdm
import torch
import pandas as pd
from transformers import AutoModelForMaskedLM, AutoTokenizer

import os


def make_hidden_states(
    infile: str,
    outdir: str,
    model_name: str = "google-bert/bert-large-uncased",
):
    """
    Passes input sentences through model and extracts output tensors at each
    hidden state. Tensors are saved as `outdir`/layer`i`/input`j`. Models
    should be bidirectional to ensure embeddings at later time steps do not
    contain additional information. Defaults to bert-large-uncased.

    :param str infile: Path to `.tsv` file with sentences. Must have sentences
        in column with header "sentence".
    :param str outdir: Root directory to save tensors.
    :param str model_name: HuggingFace model to use. Defaults to
        "bert-large-uncased".
    """

    df = pd.read_csv(infile, sep="\t")
    input_data = df["sentence"].to_list()

    model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )

    for i, input in enumerate(tqdm.tqdm(input_data)):
        inputs = tokenizer(input, return_tensors="pt", padding=True, truncation=True)
        inputs = inputs.to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states

        for j, hs in enumerate(hidden_states):
            os.makedirs(f"{outdir}/layer{j}/", exist_ok=True)
            torch.save(hs, f"{outdir}/layer{j}/input{i}.pt")
