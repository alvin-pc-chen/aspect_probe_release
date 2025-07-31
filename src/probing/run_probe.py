import torch
from torch.utils.data import DataLoader
from sklearn import metrics
from glob import glob

import logging
import os
import pickle
from typing import List

from .custom_dataset import CustomDataset


def run_probe_experiment(
    probe,
    train_dir: str,
    test_dir: str,
    epochs: int = 8,
    train_batch_size: int = 16,
    train_shuffle: bool = True,
    learn_rate: float = 0.01,
    momentum: float = 0,
    test_batch_size: int = 16,
    loss_fn=torch.nn.CrossEntropyLoss(),
    # TODO: allow other optimizers
    device: str = "mps",
    save_path: str = "",
) -> List[dict]:
    """
    Entry point for running probing experiments. Train and test datasets are
    loaded sequentially and deleted after each (layer, position) experiment is
    completed to prevent memory issues.

    :param str train_dir: Input directory of pickled files created by
        `make_probing_data` for training.
    :param str test_dir: Input directory of pickled files created by
        `make_probing_data` for testing.
    :param int epochs: Number of epochs to train each probe.
    :param int train_batch_size: Batch size for training.
    :param bool train_shuffle: Whether to shuffle training data or not.
    :param float learn_rate: Learn rate of optimizers.
    :param float momentum: Momentum of optimizers.
    :param int test_batch_size: Batch size for testing.
    :param loss_fn: Loss function to use for training. Currently only works
        with `CrossEntropyLoss`.
    :param probe: Base model to use for probing. Custom probes need to share
        input args with torch.nn.Linear.
    :param str device: Device to use for training. If not mps, change
    :param str save_path: Path to save trained probes, doesn't save by default.

    :rtype: dict
    """

    def _train_one_epoch(model, optimizer, dataloader):
        for batch in dataloader:
            optimizer.zero_grad()
            labels = batch[0].to(device)
            hidden_states = batch[1].to(device)
            loss = loss_fn(
                model(hidden_states).squeeze(1),
                labels,
            )
            loss.backward()
            optimizer.step()
            return loss.item()

    def _eval_one_epoch(model, dataloader):
        pred, true = [], []
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                labels = batch[0].to(device)
                hidden_states = batch[1].to(device)
                pred.extend(model(hidden_states).argmax(1).cpu().numpy())
                true.extend(labels.argmax(1).cpu().numpy())
        return true, pred

    torch.set_default_dtype(torch.float16)

    train_layers = glob(f"{train_dir}/layer*")
    train_layers.sort(key=lambda x: int(x.split("/")[-1].split("layer")[-1]))
    test_layers = glob(f"{test_dir}/layer*")
    test_layers.sort(key=lambda x: int(x.split("/")[-1].split("layer")[-1]))

    logging.debug("Starting training...")

    results = []

    for layer in range(len(train_layers)):

        train_position_paths = glob(f"{train_layers[layer]}/position*.pkl")
        train_position_paths.sort(
            key=lambda x: int(x.split("/")[-1].split("position")[1].split(".pkl")[0])
        )

        test_position_paths = glob(f"{test_layers[layer]}/position*.pkl")
        test_position_paths.sort(
            key=lambda x: int(x.split("/")[-1].split("position")[1].split(".pkl")[0])
        )
        if save_path:
            os.makedirs(f"{save_path}/layer{layer}/", exist_ok=True)

        for position in range(len(train_position_paths)):

            labels, tensors = pickle.load(open(train_position_paths[position], "rb"))
            ds = CustomDataset(labels, tensors, tensors[0].shape.numel())
            train_dataloader = DataLoader(
                ds,
                batch_size=train_batch_size,
                shuffle=train_shuffle,
            )

            logging.debug(
                f"Training on layer {layer}, position {position} with dataset of {type(ds)} and {ds.hs_dim} dimensions."
            )

            classifier = probe(in_features=ds.hs_dim, out_features=2).to(device)
            optimizer = torch.optim.SGD(
                classifier.parameters(),
                lr=learn_rate,
                momentum=momentum,
            )

            for epoch in range(epochs):
                classifier.train(mode=True)
                loss = _train_one_epoch(classifier, optimizer, train_dataloader)
                logging.info(
                    f"Layer {layer}, Classifier {position}, Epoch {epoch}, Loss: {loss}"
                )

            test_labels, test_tensors = pickle.load(
                open(test_position_paths[position], "rb")
            )
            test_ds = CustomDataset(
                test_labels, test_tensors, test_tensors[0].shape.numel()
            )
            test_dataloader = DataLoader(
                test_ds,
                batch_size=test_batch_size,
                shuffle=False,
            )

            logging.debug(
                f"Evaluating on layer {layer}, classifier {position} with dataset of {type(test_ds)} and {test_ds.hs_dim} dimensions."
            )
            true, pred = _eval_one_epoch(classifier, test_dataloader)
            acc = metrics.accuracy_score(true, pred)
            macro_f1 = metrics.f1_score(true, pred, average="macro")
            micro_f1 = metrics.f1_score(true, pred, average="micro")
            results.append(
                {
                    "hidden_layer": layer,
                    "token_position": position,
                    "accuracy": acc,
                    "macro_f1": macro_f1,
                    "micro_f1": micro_f1,
                }
            )

            logging.info(
                f"""Layer {layer}, Classifier {position} test results:
                \tAccuracy: {acc},
                \tMacro F1: {macro_f1},
                \tMicro F1: {micro_f1}"""
            )

            if save_path:
                current_save_path = f"{save_path}/layer{layer}/position{position}.pt"
                torch.save(classifier.state_dict(), current_save_path)
                logging.info(
                    f"Probe for Layer {layer}, Position {position} saved at {current_save_path}."
                )

            torch.mps.empty_cache()

    return results
