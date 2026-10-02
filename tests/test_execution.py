from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from benchforge.core.config import ModelConfig, OutputConfig, RunConfig
from benchforge.execution import run_benchmark
from benchforge.models import ModelRegistry
from benchforge.storage import read_run_artifacts


def _with_output(config: RunConfig, path: Path) -> RunConfig:
    return config.model_copy(update={"output": OutputConfig(directory=path)})


def test_example_benchmark_completes_and_aggregates(example_config: RunConfig) -> None:
    result = run_benchmark(example_config, persist=False)
    assert len(result.fold_results) == example_config.split.n_splits
    assert len(result.predictions) == 569
    assert {record.sample_index for record in result.predictions} == set(range(569))
    for name, aggregate in result.aggregate_metrics.items():
        fold_values = [fold.metrics[name] for fold in result.fold_results]
        assert aggregate.mean == np.mean(fold_values)
        assert aggregate.std == np.std(fold_values, ddof=0)


def test_identical_runs_have_equivalent_results(example_config: RunConfig) -> None:
    first = run_benchmark(example_config, persist=False)
    second = run_benchmark(example_config, persist=False)
    assert first.fingerprint == second.fingerprint
    assert first.fold_results == second.fold_results
    assert first.aggregate_metrics == second.aggregate_metrics
    assert first.predictions == second.predictions


def test_runner_constructs_and_fits_a_fresh_model_per_fold(example_config: RunConfig) -> None:
    created: list[LogisticRegression] = []

    def factory(parameters: dict[str, object], seed: int) -> LogisticRegression:
        estimator = LogisticRegression(random_state=seed, **parameters)
        created.append(estimator)
        return estimator

    registry = ModelRegistry({"observed_logistic": factory})
    config = example_config.model_copy(
        update={
            "model": ModelConfig(
                name="observed_logistic",
                parameters={"max_iter": 1000, "solver": "liblinear"},
            )
        }
    )
    run_benchmark(config, model_registry=registry, persist=False)
    assert len(created) == config.split.n_splits
    assert len({id(estimator) for estimator in created}) == config.split.n_splits
    assert all(hasattr(estimator, "coef_") for estimator in created)


def test_artifacts_are_persisted_and_readable(example_config: RunConfig, tmp_path: Path) -> None:
    config = _with_output(example_config, tmp_path / "runs")
    result = run_benchmark(config)
    assert result.artifact_directory is not None
    expected = {
        "config.json",
        "metadata.json",
        "fold_metrics.json",
        "aggregate_metrics.json",
        "predictions.csv",
    }
    assert {path.name for path in result.artifact_directory.iterdir()} == expected
    artifacts = read_run_artifacts(result.artifact_directory)
    assert artifacts["metadata"]["config_fingerprint"] == result.fingerprint
    assert len(artifacts["fold_metrics"]) == config.split.n_splits
    assert len(artifacts["predictions"]) == 569
    assert artifacts["aggregate_metrics"]["accuracy"]["mean"] == (
        result.aggregate_metrics["accuracy"].mean
    )
