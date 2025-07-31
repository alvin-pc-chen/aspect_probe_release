import argparse
import logging


def _add_global_args(parser, is_subparser=False):
    group = parser.add_argument_group("global options")
    return (
        group.add_argument(
            "--log_level",
            type=str,
            choices=logging._nameToLevel.keys(),
            default=argparse.SUPPRESS if is_subparser else "INFO",
        ),
        group.add_argument(
            "--log_file",
            type=str,
            default=argparse.SUPPRESS if is_subparser else group.argument_default,
        ),
        group.add_argument(
            "--append_log",
            dest="log_filemode",
            action="store_const",
            const="a",
            default=argparse.SUPPRESS if is_subparser else "w",
            help="Append to log file instead of overwriting.",
        ),
    )


def _add_path_args(parser, required=True):
    return (
        parser.add_argument(
            "-r",
            "--results_path",
            type=str,
            required=required,
            help=(
                f"Path to save probe result csv." f"{' Required.' if required else ''}"
            ),
        ),
        parser.add_argument(
            "--heatmap_path",
            type=str,
            required=required,
            help=(
                "Path to save heatmap png." 
                f"{' Required.' if required else ''}"
            ),
        ),
        parser.add_argument(
            "--train_dir",
            type=str,
            required=required,
            help=(
                "Path to root directory of training pkls."
                f"{' Required.' if required else ''}"
            ),
        ),
        parser.add_argument(
            "--test_dir",
            type=str,
            required=required,
            help=(
                "Path to root directory of testing pkls."
                f"{' Required.' if required else ''}"
            ),
        ),
        parser.add_argument(
            "-s",
            "--save_path",
            type=str,
            help=(
                "Path to save trained probes, doesn't save if not given."
            ),
        ),
    )


def _add_probe_args(parser):
    return (
        parser.add_argument(
            "-e",
            "--exp_name",
            type=str,
            help=("Experiment name used for heatmap."),
        ),
        parser.add_argument(
            "--device",
            type=str,
            default="auto",
            choices=["auto", "cpu", "cuda", "mps"],
            help="Target device for loading models/tokenizers. Defaults to 'auto'.",
        ),
    )


def _add_hyperparameter_args(parser):
    return (
        # TODO add loss_fn
        parser.add_argument(
            "-p",
            "--probe",
            type=str,
            default="linear",
            choices=["linear", "multi"],
            help="Type of probe to train. Defaults to 'linear'.",
        ),
        parser.add_argument(
            "--epochs",
            type=int,
            default=8,
        ),
        parser.add_argument(
            "--train_batch_size",
            type=int,
            default=16,
        ),
        parser.add_argument(
            "--train_shuffle",
            type=bool,
            default=True,
        ),
        parser.add_argument(
            "--learn_rate",
            type=float,
            default=0.01,
        ),
        parser.add_argument(
            "--test_batch_size",
            type=int,
            default=16,
        ),
    )


main_parser = argparse.ArgumentParser()
_add_global_args(main_parser)
subparsers = main_parser.add_subparsers(
    dest="subcommand",
    title="subcommands",
    required=True,
)

PROBE_HELP = "Run probing experiment."
probe_parser = subparsers.add_parser(
    "probe",
    help=PROBE_HELP,
    description=PROBE_HELP,
)
_add_global_args(probe_parser, is_subparser=True)
_add_path_args(probe_parser)
_add_probe_args(probe_parser)
_add_hyperparameter_args(probe_parser)

make_parser = subparsers.add_parser(
    "make",
)
