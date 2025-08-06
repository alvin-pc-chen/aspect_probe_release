import sys
import logging

from utils import cli
from utils.make_clean_data import make_clean_data
from utils.make_hidden_states import make_hidden_states
from utils.make_probing_data import make_probing_data
from utils.save_results import save_results
from utils.make_heatmap import make_heatmap
from probing.run_probe import run_probe_experiment
from probing.multilinear_probe import MultiLinearProbe


def main(argv):
    # Set subcommand handlers.
    cli.make_parser.set_defaults(func=make)
    cli.probe_parser.set_defaults(func=probe)

    # Parse CL opts.
    args = cli.main_parser.parse_args(argv)

    # Apply logging CL opts.
    logging.basicConfig(
        style="{",
        format="[{asctime}] [{levelname}] [{filename}:{lineno} {funcName}] {msg}",
        level=args.log_level,
        filename=args.log_file,
        filemode=args.log_filemode,
    )

    # Create dict that removes kwargs not recognized by subcommand handlers.
    clkwargs = args.__dict__.copy()
    clkwargs.pop("subcommand")
    clkwargs.pop("log_level")
    clkwargs.pop("log_file")
    clkwargs.pop("log_filemode")
    clkwargs.pop("func")

    # Execute subcommand handler.
    args.func(**clkwargs)


def make(
    original_path: str,
    cleaned_path: str,
    hidden_dir: str,
    experiment_dir: str,
    model_name: str,
    device: str,
):
    """
    Runs through all steps necessary to prepare data for experiment.
    """
    make_clean_data(infile=original_path, outfile=cleaned_path)
    make_hidden_states(
        infile=cleaned_path, outdir=hidden_dir, model_name=model_name, device=device
    )
    make_probing_data(
        infile=cleaned_path,
        hl_dir=hidden_dir,
        outdir=experiment_dir,
        model_name=model_name,
        device=device,
    )


def probe(
    exp_name: str,
    results_path: str,
    heatmap_path: str,
    train_dir: str,
    test_dir: str,
    epochs: int = 8,
    train_batch_size: int = 16,
    train_shuffle: bool = True,
    learn_rate: float = 0.01,
    momentum: float = 0,
    test_batch_size: int = 16,
    loss_fn: str = "cross_entropy",
    probe: str = "linear",
    device: str = "auto",
    save_path: str = "",
):

    import torch

    if loss_fn == "cross_entropy":
        loss_function = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError("Only cross_entropy implemented.")

    if probe == "linear":
        probe_model = torch.nn.Linear
    elif probe == "multi":
        probe_model = MultiLinearProbe
    else:
        raise NotImplementedError("Only linear and multi implemented.")

    # TODO: better way of resolving kwargs
    training_kwargs = {
        "probe": probe_model,
        "train_dir": train_dir,
        "test_dir": test_dir,
        "epochs": epochs,
        "train_batch_size": train_batch_size,
        "train_shuffle": train_shuffle,
        "learn_rate": learn_rate,
        "momentum": momentum,
        "test_batch_size": test_batch_size,
        "loss_fn": loss_function,
        "device": device,
        "save_path": save_path,
    }

    results = run_probe_experiment(**training_kwargs)
    save_results(results_path, results, training_kwargs=training_kwargs)
    make_heatmap(results_path, outfile=heatmap_path, name=exp_name)


if __name__ == "__main__":
    main(sys.argv[1:])
