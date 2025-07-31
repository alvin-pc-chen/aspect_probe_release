import csv
import logging
from typing import List


def save_results(outfile: str, results: List[dict], training_kwargs: dict):
    """
    Saves results and training hyperparameters from a single probing experiment
    in `.csv` format.

    :param str outfile: Path to `.csv` file to save results.
    :param List[dict] results: Results from `run_probe` experiment.
    :param dict training_kwargs: Same dict passed to `run_probe`.
    """

    logging.debug(f"Saving results to {outfile}")
    with open(outfile, newline="", mode="w") as results_file:
        results_writer = csv.writer(results_file)
        results_writer.writerow(
            [
                "Epochs",
                "Train Batch Size",
                "Test Batch Size",
                "Learning Rate",
                "Momentum",
                "Hidden Layer",
                "Token Position",
                "Accuracy",
                "Macro F1",
                "Micro F1",
            ]
        )

        for res in results:
            results_writer.writerow(
                [
                    training_kwargs["epochs"],
                    training_kwargs["train_batch_size"],
                    training_kwargs["test_batch_size"],
                    training_kwargs["learn_rate"],
                    training_kwargs["momentum"],
                    res["hidden_layer"],
                    res["token_position"],
                    res["accuracy"],
                    res["macro_f1"],
                    res["micro_f1"],
                ]
            )
    logging.debug(f"Results saved to {outfile}")
