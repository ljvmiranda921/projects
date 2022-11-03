from typing import Optional, List, Dict
import shlex
import subprocess

import typer
from wasabi import msg

from .constants import DATASET_CONFIG, CONFIGS

Arg = typer.Argument
Opt = typer.Option


def run_main_results(
    # fmt: off
    datasets: Optional[List[str]] = Arg(None, help="Datasets to run the experiment on. If None is passed, then experiment will be run on all datasets.", show_default=True),
    config: str = Opt("spancat", help="The spaCy configuration file to use for training."),
    gpu_id: int = Opt(0, help="Set the random seed.", show_default=True),
    dry_run: bool = Opt(False, "--dry-run", help="Print the commands, don't run them."),
    batch_size: int = Opt(1000, "--batch-size", "-S", "--sz", help="Set the batch size.", show_default=True),
    seed: int = Opt(0, help="Set the random seed", show_default=True),
    # fmt: on
):
    dataset_config = _get_datasets(datasets)
    for dataset, config in dataset_config.items():
        msg.divider(dataset, char="X")
        commands = []

        # Train command
        train_command = _make_train_cmd(
            dataset=dataset,
            config=config,
            language=config.get("lang"),
            vectors=config.get("vectors"),
            gpu_id=gpu_id,
            batch_size=batch_size,
            seed=seed,
        )
        commands.append(train_command)

        # Evaluate command
        eval_command = _make_eval_cmd(
            dataset=dataset,
            config=config,
            gpu_id=gpu_id,
            seed=seed,
        )
        commands.append(eval_command)

        # Run commands
        _run(commands)


def _get_datasets(datasets: Optional[List[str]]) -> Dict:
    if datasets:
        dataset_config = {k: v for k, v in DATASET_CONFIG.items() if k in datasets}
    else:
        dataset_config = DATASET_CONFIG

    msg.info(f"Retrieving datasets: {', '.join(dataset_config.keys())}")
    return dataset_config


def _make_train_cmd(
    dataset: str,
    config: str,
    vectors: str,
    language: str,
    gpu_id: int = 0,
    seed: int = 42,
    batch_size: int = 1000,
) -> str:
    """Construct train command based from a template"""
    command = (
        f"spacy project run train .  "
        f"--vars.dataset {dataset} "
        f"--vars.config {config} "
        f"--vars.language {language} "
        f"--vars.vectors {vectors} "
        f"--vars.gpu-id {gpu_id} "
        f"--vars.seed {seed} "
        f"--vars.batch_size {batch_size} "
    )
    return command


def _make_eval_cmd(
    dataset: str,
    config: str,
    gpu_id: int = 0,
    seed: int = 0,
) -> str:
    """Construct eval command based from a template"""
    command = f"""
    spacy project run evaluate .
    --vars.ner_config {config}
    --vars.dataset {dataset}
    --vars.gpu-id {gpu_id}
    --vars.seed {seed}
    """
    return command


def _run(cmds: List[str], dry_run: bool = False):
    """Run a set of commands in order"""
    for cmd in cmds:
        _cmd = shlex.split(cmd)
        if dry_run:
            print(cmd.strip())
        else:
            subprocess.run(_cmd)
