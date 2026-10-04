import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from benchforge import RobustnessConfig, load_robustness_config, run_robustness
from benchforge.cli import main
from benchforge.core.config import FileDatasetConfig, ModelConfig, OutputConfig, TaskType
from benchforge.data.registry import default_dataset_registry
from benchforge.evaluation.metrics import aggregate_fold_metrics
from benchforge.models.registry import default_model_registry
from benchforge.robustness import RobustnessExecutionError
from benchforge.robustness.planner import build_robustness_plan
from benchforge.robustness.results import CandidateScore, RepetitionScores
from benchforge.robustness.statistics import compare_all_pairs, summarize_ranks
from benchforge.storage import read_robustness_artifacts


def fast_config(name: str = "breast_cancer") -> RobustnessConfig:
    config = load_robustness_config(f"configs/examples/{name}_robustness.yaml")
    return config.model_copy(
        update={
            "models": config.models[:2],
            "robustness": config.robustness.model_copy(update={"repetitions": 3}),
            "split": config.split.model_copy(update={"n_splits": 2}),
        }
    )


def semantic_result(result):
    data = asdict(replace(result, artifact_directory=None, total_duration_seconds=0))
    data["plan"]["repetitions"] = [
        {
            "repetition_id": plan.repetition_id,
            "seed": plan.seed,
            "folds": [
                (fold.fold_id, fold.train_indices.tolist(), fold.validation_indices.tolist())
                for fold in plan.folds
            ],
        }
        for plan in result.plan.repetitions
    ]
    return data


@pytest.mark.parametrize("name", ["breast_cancer", "iris_multiclass", "diabetes_regression"])
def test_end_to_end_determinism_shared_folds_metrics_and_persistence(
    name: str, tmp_path: Path
) -> None:
    config = fast_config(name).model_copy(update={"output": OutputConfig(directory=tmp_path)})
    first = run_robustness(config, persist=False)
    second = run_robustness(config)
    assert semantic_result(first) == semantic_result(second)
    assert second.artifact_directory is not None
    assert second.fingerprint == config.fingerprint_for_dataset(second.dataset.identity)
    seeds = [plan.seed for plan in first.plan.repetitions]
    assert len(set(seeds)) == 3
    fold_assignments = []
    for plan, repetition in zip(first.plan.repetitions, first.repetitions, strict=True):
        expected = {
            int(index): fold.fold_id for fold in plan.folds for index in fold.validation_indices
        }
        fold_assignments.append(expected)
        for candidate in repetition.candidates:
            assert candidate.status == "success"
            assert {row.sample_index: row.fold_id for row in candidate.predictions} == expected
            assert len(candidate.predictions) == first.dataset.row_count
            assert len(candidate.fold_results) == config.split.n_splits
            aggregate = aggregate_fold_metrics(
                {metric.name: metric.value for metric in fold.metrics}
                for fold in candidate.fold_results
            )
            assert (
                candidate.score(config.primary_metric.value)
                == aggregate[config.primary_metric.value].mean
            )
            if name != "breast_cancer":
                assert all(record.score is None for record in candidate.predictions)
            if name == "iris_multiclass":
                assert all(isinstance(record.prediction, int) for record in candidate.predictions)
    assert fold_assignments[0] != fold_assignments[1]
    for stability in first.rank_stability:
        assert sum(row.count for row in stability.rank_counts) == 3
    persisted = read_robustness_artifacts(second.artifact_directory)
    assert persisted["metadata"]["artifact_schema_version"] == 1
    assert persisted["metadata"]["robustness_fingerprint"] == second.fingerprint
    assert persisted["pairwise_comparisons"] == json.loads(
        json.dumps([asdict(pair) for pair in first.pairwise_comparisons])
    )
    assert persisted["robustness_leaderboard"] == json.loads(
        json.dumps([asdict(entry) for entry in first.leaderboard])
    )
    observations = []
    for plan, repetition in zip(second.plan.repetitions, persisted["repetitions"], strict=True):
        assert repetition["seed"] == plan.seed
        assert repetition["split_plan"] == [
            {
                "fold_id": fold.fold_id,
                "train_indices": fold.train_indices.tolist(),
                "validation_indices": fold.validation_indices.tolist(),
            }
            for fold in plan.folds
        ]
        scores = []
        for identifier, candidate in repetition["candidates"].items():
            score = next(
                metric["mean"]
                for metric in candidate["summary"]["metrics"]
                if metric["name"] == second.primary_metric
            )
            scores.append(CandidateScore(identifier, score))
            memory = next(
                item
                for item in second.repetitions[plan.repetition_id].candidates
                if item.model_identifier == identifier
            )
            assert candidate["summary"]["fold_results"] == json.loads(
                json.dumps([asdict(fold) for fold in memory.fold_results])
            )
            assert [float(row["prediction"]) for row in candidate["predictions"]] == [
                row.prediction for row in memory.predictions
            ]
        observations.append(RepetitionScores(plan.repetition_id, tuple(scores)))
    assert (
        compare_all_pairs(
            second.plan.candidate_identifiers,
            tuple(observations),
            second.primary_metric,
            second.optimization_direction,
            config.robustness,
            config.seed,
        )
        == second.pairwise_comparisons
    )
    assert (
        summarize_ranks(
            second.plan.candidate_identifiers, tuple(observations), second.optimization_direction
        )
        == second.rank_stability
    )


@pytest.mark.parametrize("primary", ["root_mean_squared_error", "mean_absolute_error"])
def test_regression_errors_stay_natural_and_comparisons_are_oriented(primary: str) -> None:
    data = fast_config("diabetes_regression").model_dump(mode="json")
    data["primary_metric"] = primary
    result = run_robustness(RobustnessConfig.model_validate(data), persist=False)
    assert result.optimization_direction == "minimize"
    assert result.leaderboard[0].model_identifier == "ridge"
    assert result.leaderboard[0].primary_metric_mean > 0
    pair = result.pairwise_comparisons[0]
    assert (pair.candidate_a, pair.candidate_b) == ("dummy", "ridge")
    assert all(value > 0 for value in pair.raw_paired_deltas)
    assert all(value < 0 for value in pair.paired_deltas)
    assert pair.b_wins == 3 and pair.interval.upper_bound < 0


def test_structurally_shared_fold_objects_and_leakage_safe_fresh_preprocessors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import benchforge.execution.runner as kernel
    import benchforge.robustness.runner as runner

    original_execute = runner.execute_fold
    original_preprocessor = kernel.build_preprocessor
    seen = []
    preprocessors = []

    def preprocessor(*args, **kwargs):
        value = original_preprocessor(*args, **kwargs)
        preprocessors.append(value)
        return value

    def execute(config, dataset, fold, **kwargs):
        seen.append((config.model.id, fold))
        output = original_execute(config, dataset, fold, **kwargs)
        scaler = preprocessors[-1].named_transformers_["numeric"].named_steps["scaler"]
        expected = dataset.features.iloc[fold.train_indices].mean().to_numpy()
        np.testing.assert_allclose(scaler.mean_, expected)
        assert not np.allclose(scaler.mean_, dataset.features.mean().to_numpy(), rtol=1e-10)
        return output

    monkeypatch.setattr(runner, "execute_fold", execute)
    monkeypatch.setattr(kernel, "build_preprocessor", preprocessor)
    config = fast_config()
    result = run_robustness(config, persist=False)
    assert len(preprocessors) == result.plan.approximate_maximum_fits
    assert len({id(item) for item in preprocessors}) == len(preprocessors)
    for index, plan in enumerate(result.plan.repetitions):
        calls = seen[index * 4 : (index + 1) * 4]
        assert calls[0][1] is calls[2][1] is plan.folds[0]
        assert calls[1][1] is calls[3][1] is plan.folds[1]


def test_failure_isolation_preserves_partial_fold_evidence_and_missing_pairs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import benchforge.robustness.runner as runner

    original = runner.execute_fold
    calls = 0

    def execute(config, dataset, fold, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise runner.FoldExecutionError("one candidate fold fails")
        return original(config, dataset, fold, **kwargs)

    monkeypatch.setattr(runner, "execute_fold", execute)
    config = fast_config().model_copy(update={"output": OutputConfig(directory=tmp_path)})
    config = config.model_copy(update={"split": config.split.model_copy(update={"n_splits": 3})})
    result = run_robustness(config)
    assert len(result.failures) == 1
    assert result.failures[0].skipped_fold_ids == (2,)
    failed = result.repetitions[0].candidates[0]
    assert failed.status == "failed" and len(failed.fold_results) == 1
    assert failed.predictions and not failed.metrics
    assert result.repetitions[0].candidates[1].status == "success"
    assert result.pairwise_comparisons[0].usable_repetitions == 2
    assert result.pairwise_comparisons[0].repetition_ids == (1, 2)
    assert result.leaderboard[0].status == "success"
    assert result.leaderboard[1].status == "partial"
    assert result.leaderboard[1].failed_repetitions == 1
    persisted = read_robustness_artifacts(result.artifact_directory)
    candidate = persisted["repetitions"][0]["candidates"][failed.model_identifier]
    assert candidate["summary"]["status"] == "failed"
    assert candidate["predictions"]
    assert len(persisted["failures"]) == 1


@pytest.mark.parametrize("all_fail", [False, True])
def test_systemic_failures_persist_even_when_all_candidates_fail(
    tmp_path: Path, all_fail: bool
) -> None:
    base = fast_config()
    models = tuple(
        ModelConfig(
            name=model.name,
            id=model.id,
            parameters={"not_a_parameter": 1} if all_fail or index == 0 else model.parameters,
        )
        for index, model in enumerate(base.models)
    )
    config = base.model_copy(update={"models": models, "output": OutputConfig(directory=tmp_path)})
    if all_fail:
        with pytest.raises(RobustnessExecutionError) as captured:
            run_robustness(config)
        result = captured.value.result
        assert all(row.rank is None for row in result.leaderboard)
    else:
        result = run_robustness(config)
        assert result.leaderboard[0].status == "success"
    assert len(result.failures) == (6 if all_fail else 3)
    assert result.pairwise_comparisons[0].usable_repetitions == 0
    assert result.pairwise_comparisons[0].mean_delta is None
    assert result.artifact_directory is not None
    assert len(read_robustness_artifacts(result.artifact_directory)["failures"]) == len(
        result.failures
    )


def test_identity_ignores_output_candidate_metric_order_and_file_relocation(tmp_path: Path) -> None:
    config = fast_config()
    first = run_robustness(config, persist=False)
    data = config.model_dump(mode="json")
    data["output"] = {"directory": str(tmp_path)}
    data["models"].reverse()
    data["metrics"].reverse()
    second = run_robustness(RobustnessConfig.model_validate(data), persist=False)
    assert semantic_result(first) == semantic_result(second)
    source = Path("configs/examples/data/mixed_customers.csv")
    relocated = tmp_path / "relocated.csv"
    relocated.write_bytes(source.read_bytes())
    file_config = config.model_copy(
        update={
            "dataset": FileDatasetConfig(
                source="csv",
                path=source,
                target_column="subscribed",
                id_columns=("customer_id",),
                task=TaskType.BINARY_CLASSIFICATION,
            )
        }
    )
    moved = file_config.model_copy(
        update={"dataset": file_config.dataset.model_copy(update={"path": relocated})}
    )
    assert semantic_result(run_robustness(file_config, persist=False)) == semantic_result(
        run_robustness(moved, persist=False)
    )


@pytest.mark.parametrize(
    "field",
    [
        "seed",
        "repetitions",
        "confidence_level",
        "bootstrap_samples",
        "primary_metric",
        "parameters",
        "split",
        "preprocessing",
    ],
)
def test_semantic_changes_affect_identity(field: str) -> None:
    config = fast_config()
    dataset = default_dataset_registry.resolve(config.dataset)
    first = build_robustness_plan(config, dataset, default_model_registry)
    data = config.model_dump(mode="json")
    if field == "seed":
        data["seed"] += 1
    elif field in {"repetitions", "bootstrap_samples"}:
        data["robustness"][field] += 1
    elif field == "confidence_level":
        data["robustness"][field] = 0.9
    elif field == "primary_metric":
        data[field] = "f1"
    elif field == "parameters":
        data["models"][0]["parameters"]["C"] = 2.0
    elif field == "split":
        data["split"]["n_splits"] = 3
    else:
        data["preprocessing"]["standardize"] = False
    changed = build_robustness_plan(
        RobustnessConfig.model_validate(data), dataset, default_model_registry
    )
    assert changed.fingerprint != first.fingerprint
    if field == "seed":
        assert [item.seed for item in changed.repetitions] != [
            item.seed for item in first.repetitions
        ]


@pytest.mark.parametrize("name", ["breast_cancer", "iris_multiclass", "diabetes_regression"])
def test_cli_smoke(name: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = fast_config(name).model_dump(mode="json")
    data["output"]["directory"] = str(tmp_path / "artifacts")
    path = tmp_path / "robustness.yaml"
    path.write_text(yaml.safe_dump(data))
    assert main(["robustness", str(path)]) == 0
    output = capsys.readouterr().out
    for expected in [
        "BenchForge robustness",
        "Robustness leaderboard",
        "Rank stability",
        "Mean paired delta",
        "robustness interval",
        "estimator fits",
        "Artifacts:",
    ]:
        assert expected in output


def test_cli_all_failures_reports_error_and_persists(tmp_path: Path, capsys) -> None:
    data = fast_config().model_dump(mode="json")
    for model in data["models"]:
        model["parameters"] = {"not_a_parameter": 1}
    data["output"]["directory"] = str(tmp_path / "artifacts")
    path = tmp_path / "robustness.yaml"
    path.write_text(yaml.safe_dump(data))
    assert main(["robustness", str(path)]) == 2
    assert "every robustness candidate failed" in capsys.readouterr().err
    assert len(list((tmp_path / "artifacts").glob("*/failures.json"))) == 1


def test_multiple_failed_candidates_do_not_remove_successful_candidate() -> None:
    base = fast_config()
    config = base.model_copy(
        update={
            "models": (
                ModelConfig(name="random_forest_classifier", id="bad_a", parameters={"bad": 1}),
                ModelConfig(name="extra_trees_classifier", id="bad_b", parameters={"bad": 1}),
                ModelConfig(name="dummy_classifier", id="good"),
            )
        }
    )
    result = run_robustness(config, persist=False)
    assert result.leaderboard[0].model_identifier == "good"
    assert result.leaderboard[0].completed_repetitions == 3
    assert len(result.failures) == 6
    assert len(result.pairwise_comparisons) == 3
    assert all(pair.usable_repetitions == 0 for pair in result.pairwise_comparisons)
    assert len(result.repetitions) == 3


def test_repetition_seed_collisions_are_resolved_deterministically(monkeypatch) -> None:
    import benchforge.robustness.planner as planner

    original = planner.derive_seed

    def colliding(seed, *parts):
        return 123 if parts[-1] == 0 else original(seed, *parts)

    monkeypatch.setattr(planner, "derive_seed", colliding)
    first = planner.build_robustness_plan(fast_config())
    second = planner.build_robustness_plan(fast_config())
    assert [item.seed for item in first.repetitions] == [item.seed for item in second.repetitions]
    assert len({item.seed for item in first.repetitions}) == 3


def test_versions_dataset_identity_and_implicit_identifiers_affect_identity_correctly(
    monkeypatch,
) -> None:
    import benchforge.robustness.planner as planner

    config = fast_config()
    original = planner.dependency_versions()
    first = planner.build_robustness_plan(config)
    monkeypatch.setattr(
        planner, "dependency_versions", lambda: {**original, "benchforge": "future"}
    )
    assert planner.build_robustness_plan(config).fingerprint != first.fingerprint
    monkeypatch.setattr(planner, "dependency_versions", lambda: original)
    assert config.fingerprint_for_dataset("different:content") != first.fingerprint
    data = config.model_dump(mode="json")
    for model in data["models"]:
        model.pop("id")
    implicit = RobustnessConfig.model_validate(data)
    for model in data["models"]:
        model["id"] = model["name"]
    explicit = RobustnessConfig.model_validate(data)
    assert implicit.canonical_json() == explicit.canonical_json()
    identity = "sklearn:breast_cancer:v1"
    assert implicit.fingerprint_for_dataset(identity) == explicit.fingerprint_for_dataset(identity)


def test_robustness_does_not_mutate_global_numpy_random_state() -> None:
    np.random.seed(999)
    before = np.random.get_state()
    run_robustness(fast_config("diabetes_regression"), persist=False)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
