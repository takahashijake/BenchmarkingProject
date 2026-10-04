from pathlib import Path

from benchforge.core.config import BenchmarkConfig, ModelConfig, OutputConfig, load_benchmark_config
from benchforge.execution import run_benchmark_suite
from benchforge.storage import read_benchmark_artifacts


def _fast_config() -> BenchmarkConfig:
    config = load_benchmark_config("configs/examples/breast_cancer_suite.yaml")
    return config.model_copy(update={"split": config.split.model_copy(update={"n_splits": 3})})


def test_all_model_families_complete_and_share_identical_folds() -> None:
    config = _fast_config()
    result = run_benchmark_suite(config, persist=False)
    assert not result.failures
    assert {candidate.model_config.name for candidate in result.candidates} == {
        "dummy_classifier",
        "logistic_regression",
        "random_forest_classifier",
        "extra_trees_classifier",
        "hist_gradient_boosting_classifier",
    }
    fold_maps = [
        {
            (prediction.sample_index, prediction.fold_id)
            for prediction in candidate.run_result.predictions
        }
        for candidate in result.candidates
    ]
    assert all(fold_map == fold_maps[0] for fold_map in fold_maps[1:])
    assert all(candidate.duration_seconds >= 0 for candidate in result.candidates)
    assert result.total_duration_seconds >= 0


def test_candidate_failure_is_recorded_and_not_ranked() -> None:
    config = _fast_config()
    models = (
        ModelConfig(name="logistic_regression", parameters={"max_iter": 500}),
        ModelConfig(name="random_forest_classifier", parameters={"not_a_parameter": 1}),
    )
    result = run_benchmark_suite(config.model_copy(update={"models": models}), persist=False)
    assert len(result.candidates) == 1
    assert len(result.failures) == 1
    assert result.failures[0].model_identifier == "random_forest_classifier"
    failed_entry = next(entry for entry in result.leaderboard if entry.status == "failed")
    assert failed_entry.rank is None
    assert failed_entry.primary_metric_mean is None


def test_benchmark_is_reproducible_and_leaderboard_matches_measurements() -> None:
    config = _fast_config()
    first = run_benchmark_suite(config, persist=False)
    second = run_benchmark_suite(config, persist=False)
    assert first.fingerprint == second.fingerprint
    assert [candidate.run_result.fold_results for candidate in first.candidates] == [
        candidate.run_result.fold_results for candidate in second.candidates
    ]
    assert [candidate.run_result.predictions for candidate in first.candidates] == [
        candidate.run_result.predictions for candidate in second.candidates
    ]
    successful = [entry for entry in first.leaderboard if entry.status == "success"]
    means = [entry.primary_metric_mean for entry in successful]
    assert means == sorted(means, reverse=True)
    assert [entry.rank for entry in successful] == list(range(1, len(successful) + 1))


def test_suite_artifacts_are_written_and_readable(tmp_path: Path) -> None:
    config = _fast_config().model_copy(
        update={"output": OutputConfig(directory=tmp_path / "artifacts")}
    )
    result = run_benchmark_suite(config)
    assert result.artifact_directory is not None
    expected = {
        "benchmark_config.json",
        "metadata.json",
        "manifest.json",
        "leaderboard.json",
        "leaderboard.csv",
        "failures.json",
        "dataset_summary.json",
        "runs",
    }
    assert {path.name for path in result.artifact_directory.iterdir()} == expected
    artifacts = read_benchmark_artifacts(result.artifact_directory)
    assert artifacts["metadata"]["benchmark_fingerprint"] == result.fingerprint
    assert len(artifacts["leaderboard"]) == len(config.models)
    assert len(list((result.artifact_directory / "runs").iterdir())) == len(result.candidates)
