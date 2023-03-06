from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import typer
from spacy.cli.project.run import project_run
from wasabi import msg

from .constants import DATASETS

Arg = typer.Argument
Opt = typer.Option

NUM_TRIALS = 3
SUBCOMMAND = "spancat"


def run_spancat(
    # fmt: off
    dataset_names: Optional[List[str]] = Arg(None, help="Datasets to run the experiments on."),
    subcommand: str = Opt(SUBCOMMAND, "--subcommand", "-C", help="Workflow command to run.", show_default=True),
    num_trials: int = Opt(NUM_TRIALS, "--num-trials", "-n", help="Set the number of trials.", show_default=True),
    config: str = Opt("spancat", "--config", "-c", help="Configuration to use for training."),
    gpu_id: int = Opt(0, "--gpu-id", "-G", help="Set the GPU ID. Use -1 for CPU.", show_default=True),
    force: bool = Opt(False, "--force", "-f", help="Force run the workflow."),
    dry: bool = Opt(False, "--dry-run", "--dry", help="Print the commands, don't run them."),
    # fmt: on
):
    """Run experiment for spancat model"""
    datasets = _get_datasets(dataset_names)
    for trial_num in range(num_trials):
        msg.divider(f"Trial {trial_num}")
        general_overrides = {
            "vars.config": config,
            "vars.trial_num": trial_num,
            "vars.seed": trial_num,  # same seed as trial number
            "vars.gpu_id": gpu_id,
        }

        for dataset, cfg in datasets.items():
            overrides = deepcopy(general_overrides)
            overrides["vars.dataset"] = dataset
            overrides["vars.lang"] = cfg.get("lang")
            overrides["vars.vectors"] = cfg.get("vectors")
            project_run(
                project_dir=Path.cwd(),
                overrides=overrides,
                subcommand=subcommand,
                force=force,
                dry=dry,
            )


def _get_datasets(d: Optional[List[str]]) -> Dict:
    dataset_vectors = {k: v for k, v in DATASETS.items() if k in d} if d else DATASETS
    msg.info(f"Retrieving datasets: {', '.join(dataset_vectors.keys())}")
    return dataset_vectors


if __name__ == "__main__":
    typer.run(run_spancat)
