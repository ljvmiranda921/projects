import srsly
import statistics
import typer
from wasabi import msg
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

from .constants import DATASETS

Arg = typer.Argument
Opt = typer.Option

Dict[str, List[Dict[str, Any]]]


def collate_results(
    # fmt: off
    metrics_dir: Path = Arg(..., dir_okay=True, exists=True, help="Path to the metrics directory. Must contain a folder for each trial."),
    config: str = Arg("spancat", help="Configuration to collate the results from."),
    output_path: Optional[Path] = Opt(None, "--output-path", "--output", "-o", help="Output JSON filepath to save the collated scores.")
    # fmt: on
):
    """Collate results and report their mean and stdev

    This expects a `metrics_dir` with the following structure:

        |--metrics_dir/
          |--dataset1/
          |  |--config/
          |    |--trial-0/
          |    |  |--scores.json
          |    |--trial-1/
          |    |  |--scores.json

    The datasets are set and can be found in scripts.constants. The config can either be
    `spancat` and `exclusive_spancat` (without the file format).
    """
    msg.info(f"Reporting results for directory `{metrics_dir}` using `{config}` config")
    results = {
        dataset: [
            srsly.read_json(f)
            for f in (metrics_dir / dataset / config).glob("**/*.json")
        ]
        for dataset in DATASETS.keys()
    }
    msg.text(f"Number of trials per dataset: {_format_num_trials(results)}")
    overall_results = _compute_overall(results)
    per_span_results = _compute_per_span(results)
    _report_results(overall_results, per_span_results)


def _format_num_trials(results: Dict[str, List[Dict[str, Any]]]) -> str:
    text = ""
    for dataset, res in results.items():
        text += f" {dataset} ({len(res)})"
    return text


def _compute_overall(
    results: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Compute mean and stdev for overall P/R/F results"""
    metric_names = ["spans_sc_p", "spans_sc_r", "spans_sc_f"]
    overall_results = {}
    for dataset, res in results.items():
        overall_results[dataset] = {}
        for metric in metric_names:
            scores = [r.get(metric) for r in res]
            overall_results[dataset][metric] = (
                statistics.mean(scores),
                statistics.stdev(scores),
            )
    return overall_results


def _compute_per_span(results: Dict[str, List[Dict]]):
    """Compute mean and stdev for per-span P/R/F results"""
    metric_names = ["p", "r", "f"]
    per_span_results = {}
    for dataset, res in results.items():
        per_span_results[dataset] = {}
        span_labels = res[0].get("spans_sc_per_type").keys()
        for span_label in span_labels:
            per_span_results[dataset][span_label] = {}
            span_prf = [r.get("spans_sc_per_type").get(span_label) for r in res]
            for metric in metric_names:
                scores = [r.get(metric) for r in span_prf]
                per_span_results[dataset][span_label][metric] = (
                    statistics.mean(scores),
                    statistics.stdev(scores),
                )
    return per_span_results


def _report_results(overall: Dict[str, Dict], per_span: Dict[str, Dict]):
    def _format_results(result: Tuple[float, float]) -> str:
        mean, stdev = result
        return "{:.2f} ({:.2f})".format(mean * 100, stdev * 100)

    msg.divider("Overall results")
    header = ("Dataset", "spans_sc_p", "spans_sc_r", "spans_sc_f")
    aligns = ("l", "r", "r", "r")
    table_data = []
    for dataset, results in overall.items():
        row = (
            dataset,
            _format_results(results.get("spans_sc_p")),
            _format_results(results.get("spans_sc_r")),
            _format_results(results.get("spans_sc_f")),
        )
        table_data.append(row)

    msg.table(table_data, header=header, divider=True, aligns=aligns)

    msg.divider("Per-span results")
    header = ("Span Label", "precision", "recall", "f1-score")
    for dataset, span_results in per_span.items():
        msg.divider(dataset, char="-")
        table_data = []
        for span_label, results in span_results.items():
            row = (
                span_label,
                _format_results(results.get("p")),
                _format_results(results.get("r")),
                _format_results(results.get("f")),
            )
            table_data.append(row)

        msg.table(table_data, header=header, divider=True, aligns=aligns)


if __name__ == "__main__":
    typer.run(collate_results)
