import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import logging


def make_heatmap(infile: str, outfile: str, name: str = "", metric: str = "Accuracy"):
    """
    Generates a heatmap from the results of a probing experiment using a chosen
    metric.

    :param str infile: Path to results file in `.csv` format.
    :param str outfile: Path to save heatmap in `.png` format.
    :param str name: Name of experiment to display. Defaults to empty string.
    :param str metric: Metric to display, choices are "Accuracy", "Macro F1",
        and "Micro F1". Defaults to "Accuracy".
    """

    logging.debug("Making heatmap")
    data = pd.read_csv(infile)
    if not name:
        logging.debug(f"No given name, making one using {outfile}")
        name = outfile.split("/")[-1].split(".png")[0]

    pivot_data = data.pivot(
        index="Hidden Layer", columns="Token Position", values=metric
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pivot_data, annot=True, cmap="viridis", fmt=".2f", annot_kws={"size": 8}
    )
    plt.title(f"Experiment {name}: {metric} Heatmap")
    plt.gca().invert_yaxis()
    plt.savefig(outfile)
    logging.debug("Heatmap saved")
