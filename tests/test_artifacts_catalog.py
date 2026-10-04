"""Adversarial evidence/index tests, using inexpensive real workflow artifacts."""

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from benchforge.artifacts.discovery import classify, discover, inventory
from benchforge.artifacts.manifest import Manifest, digest, seal
from benchforge.artifacts.types import ArtifactError, ArtifactType
from benchforge.artifacts.verify import verify
from benchforge.automl.runner import run_automl
from benchforge.catalog import Catalog, CatalogError, compare
from benchforge.catalog.compare import compare_descriptors
from benchforge.cli import main
from benchforge.core.config import (
    AutoMLConfig,
    BenchmarkConfig,
    FileDatasetConfig,
    RunConfig,
    SearchConfig,
    TaskType,
)
from benchforge.execution.benchmark import run_benchmark_suite
from benchforge.execution.runner import run_benchmark
from benchforge.robustness.config import RobustnessConfig
from benchforge.robustness.runner import RobustnessExecutionError, run_robustness
from benchforge.storage import read_run_artifacts


def configuration(task="binary_classification", **updates):
    regression = task == "regression"
    multiclass = task == "multiclass_classification"
    data = {
        "schema_version": 1,
        "task": task,
        "dataset": {
            "name": "diabetes" if regression else ("iris" if multiclass else "breast_cancer")
        },
        "split": {
            "strategy": "kfold" if regression else "stratified_kfold",
            "n_splits": 2,
            "shuffle": True,
        },
        "preprocessing": {"median_imputation": True, "standardize": True},
        "models": [
            {
                "name": "ridge_regressor" if regression else "logistic_regression",
                "id": "linear",
                "parameters": {}
                if regression
                else {"max_iter": 1000, "solver": "liblinear"}
                if not multiclass
                else {"max_iter": 1000},
            },
            {
                "name": "dummy_regressor" if regression else "dummy_classifier",
                "id": "dummy",
                "parameters": {},
            },
        ],
        "metrics": ["root_mean_squared_error" if regression else "accuracy"],
        "primary_metric": "root_mean_squared_error" if regression else "accuracy",
        "seed": 42,
        "output": {"directory": "unused"},
    }
    data.update(updates)
    return data


def generate(root, kind, task="binary_classification", **updates):
    data = configuration(task, output={"directory": str(root)}, **updates)
    if kind == "run":
        data["model"] = data.pop("models")[0]
        data.pop("primary_metric")
        result = run_benchmark(RunConfig.model_validate(data))
    elif kind in {"search", "automl"}:
        data["outer_split"] = data.pop("split")
        data["inner_split"] = dict(data["outer_split"])
        if kind == "search":
            for index, model in enumerate(data["models"]):
                model["mode"] = "search" if index == 0 else "fixed"
            data["budget"] = {"trials_per_outer_fold": 1}
            data["final_search"] = {"enabled": True, "trials": 1}
            result = run_search_config(data)
        else:
            data["models"] = {"include": [m["name"] for m in data["models"]]}
            data["budget"] = {"total_trials": 2, "final_search_trials": 0}
            result = run_automl(AutoMLConfig.model_validate(data))
    elif kind == "robustness":
        data["robustness"] = {"repetitions": 2, "bootstrap_samples": 100}
        result = run_robustness(RobustnessConfig.model_validate(data))
    else:
        result = run_benchmark_suite(BenchmarkConfig.model_validate(data))
    assert result.artifact_directory is not None
    return result.artifact_directory


def run_search_config(data):
    from benchforge.search.runner import run_search

    return run_search(SearchConfig.model_validate(data))


@pytest.fixture(scope="module")
def artifact_tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("evidence")
    paths = {kind.value: generate(root, kind.value) for kind in ArtifactType}
    return root, paths


def copy_artifact(tmp_path, artifact_tree, kind="run"):
    return Path(shutil.copytree(artifact_tree[1][kind], tmp_path / kind))


def edit(path, change):
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data, sort_keys=True))


@pytest.mark.parametrize("kind", list(ArtifactType))
def test_workflow_detection_sealing_normalization_and_queries(artifact_tree, tmp_path, kind):
    path = artifact_tree[1][kind.value]
    assert classify(path) == kind
    report = verify(path, deep=True)
    assert report.passed, report
    assert report.integrity == "sealed"
    d = report.descriptor
    assert d.artifact_type == kind and d.seed == 42 and d.status == "success"
    assert d.task == "binary_classification" and d.dataset_identity == "sklearn:breast_cancer:v1"
    assert d.primary_metric == (None if kind == ArtifactType.RUN else "accuracy")
    assert d.artifact_schema_version == (1 if kind == ArtifactType.ROBUSTNESS else None)
    manifest = Manifest.model_validate_json((path / "manifest.json").read_text())
    assert tuple(m.path for m in manifest.files) == inventory(path)
    assert all(digest(path / m.path) == m.sha256 for m in manifest.files)
    with Catalog(tmp_path / "catalog.sqlite3") as catalog:
        ids = catalog.ingest(path)
        assert len(ids) == 1
        record = catalog.show(ids[0][:12])
        assert record.locations[0].descriptor == d
        assert catalog.list(artifact_type=kind.value, task=d.task, dataset=d.dataset_identity) == (
            record,
        )
        assert catalog.list(task="regression") == ()
        assert catalog.list(status="failed") == ()
        assert catalog.ingest(path) == ids
    with Catalog(tmp_path / "catalog.sqlite3") as catalog:
        assert len(catalog.list()) == 1
        assert catalog.verify(ids[0], deep=True)[0].passed


def test_discovery_deterministic_parent_ownership_symlink_and_ambiguity(artifact_tree, tmp_path):
    first = discover(artifact_tree[0])
    assert first == discover(artifact_tree[0])
    assert len(first.directories) == 5  # nested runs and AutoML search are parent-owned
    path = copy_artifact(tmp_path, artifact_tree)
    (tmp_path / "alias").symlink_to(path, target_is_directory=True)
    result = discover(tmp_path)
    assert result.directories == (path,)
    assert result.diagnostics[0].code == "symlink_skipped"
    (path / "search_config.json").write_text("{}")
    assert not verify(path).passed
    assert discover(tmp_path).diagnostics[-1].level == "FAIL"
    with pytest.raises(ArtifactError, match="ambiguous"):
        classify(path)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_json",
        "missing_config",
        "malformed_json",
        "csv",
        "metric",
        "fingerprint",
        "missing_member",
        "schema",
        "manifest_schema",
        "candidate",
        "extra_file",
        "unsafe_member",
        "json_shape",
        "symlink",
    ],
)
def test_corruption_is_explicit(artifact_tree, tmp_path, mutation):
    path = copy_artifact(tmp_path, artifact_tree)
    if mutation == "missing_json":
        (path / "aggregate_metrics.json").unlink()
    elif mutation == "missing_config":
        (path / "config.json").unlink()
    elif mutation == "malformed_json":
        (path / "config.json").write_text("{")
    elif mutation == "csv":
        with (path / "predictions.csv").open("a") as handle:
            handle.write("not,a,valid,csv,row,extra\n")
    elif mutation == "metric":
        edit(path / "aggregate_metrics.json", lambda d: d["accuracy"].update(mean=0.001))
    elif mutation == "fingerprint":
        edit(path / "metadata.json", lambda d: d.update(config_fingerprint="f" * 64))
    elif mutation == "missing_member":
        (path / "predictions.csv").unlink()
    elif mutation == "schema":
        edit(path / "config.json", lambda d: d.update(schema_version=99))
    elif mutation == "manifest_schema":
        edit(path / "manifest.json", lambda d: d.update(manifest_schema_version=99))
    elif mutation == "candidate":
        edit(path / "config.json", lambda d: d["model"].update(name="different"))
    elif mutation == "extra_file":
        (path / "unexpected.txt").write_text("unprotected")
    elif mutation == "unsafe_member":
        edit(path / "manifest.json", lambda d: d["files"][0].update(path="../outside"))
    elif mutation == "json_shape":
        (path / "metadata.json").write_text("[]")
    else:
        (path / "linked").symlink_to(artifact_tree[1]["run"] / "config.json")
    report = verify(path, deep=True)
    assert not report.passed, report
    assert any(d.level == "FAIL" for d in report.diagnostics)
    if mutation == "missing_config":
        assert any(
            d.file == "config.json" and d.code == "missing_member" for d in report.diagnostics
        )
    with Catalog(tmp_path / "db") as catalog:
        with pytest.raises((ArtifactError, OSError)):
            catalog.ingest(path)
        assert catalog.list() == ()


@pytest.mark.parametrize(
    "kind,member",
    [
        ("search", "families/linear/outer_fold_0/trials.json"),
        ("automl", "search/families/logistic_regression/summary.json"),
        ("robustness", "repetitions/repetition_001/split_plan.json"),
        ("benchmark", "runs/linear/predictions.csv"),
    ],
)
def test_broken_nested_evidence(artifact_tree, tmp_path, kind, member):
    path = copy_artifact(tmp_path, artifact_tree, kind)
    (path / member).unlink()
    report = verify(path)
    assert not report.passed
    assert any(d.file == member and d.code == "missing_member" for d in report.diagnostics)


@pytest.mark.parametrize("kind", list(ArtifactType))
def test_legacy_formats_remain_readable_and_catalogable(artifact_tree, tmp_path, kind):
    path = copy_artifact(tmp_path, artifact_tree, kind.value)
    # Remove seals and V0.8 additions; reproduce actual V0.7 fingerprint algorithms.
    from hashlib import sha256

    for metadata_path in path.rglob("metadata.json"):
        config_path = metadata_path.parent / "config.json"
        if config_path.exists():
            config = RunConfig.model_validate_json(config_path.read_text())
            edit(
                metadata_path,
                lambda d, config=config: (
                    d.update(config_fingerprint=config.fingerprint),
                    d.pop("fingerprint_method", None),
                ),
            )
    for manifest in path.rglob("manifest.json"):
        manifest.unlink()
    metadata = path / "metadata.json"
    config_path = path / "benchmark_config.json"
    if config_path.exists():
        config = BenchmarkConfig.model_validate_json(config_path.read_text())
        semantics = config.canonical_dict(include_output=False)
        semantics["dataset"].pop("path", None)
        payload = {"benchmark": semantics, "resolved_dataset_identity": "sklearn:breast_cancer:v1"}
        fingerprint = sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        edit(
            metadata,
            lambda d: (
                d.update(benchmark_fingerprint=fingerprint),
                d.pop("fingerprint_method", None),
            ),
        )
        for nested in path.glob("runs/*/metadata.json"):
            edit(nested, lambda d: d.update(benchmark_fingerprint=fingerprint))
    for nested in path.rglob("metadata.json"):
        edit(
            nested,
            lambda d: (
                d.update(benchforge_version="0.7.0"),
                d.pop("search_space_identity", None),
                d.pop("selected_model_identity", None),
                d.pop("model_identity", None),
                d.pop("fingerprint_method", None),
            ),
        )
    report = verify(path, deep=True)
    assert report.passed, report
    assert report.integrity == "legacy/unsealed"
    assert any(d.code == "legacy_unsealed" for d in report.diagnostics)
    if kind == ArtifactType.RUN:
        assert len(read_run_artifacts(path)["predictions"]) == 569
    with Catalog(tmp_path / "db") as catalog:
        assert len(catalog.ingest(path)) == 1
        assert catalog.verify(catalog.list()[0].experiment_id)[0].passed


def test_duplicate_and_relocated_copies_share_one_experiment(artifact_tree, tmp_path):
    path = copy_artifact(tmp_path, artifact_tree, "search")
    second = Path(shutil.copytree(path, tmp_path / "replica"))
    with Catalog(tmp_path / "db") as catalog:
        ids = catalog.ingest(path)
        original_counts = {
            name: catalog.connection.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
            for name in (
                "experiments",
                "artifact_locations",
                "metrics",
                "candidates",
                "verification_records",
            )
        }
        assert catalog.ingest(path) == ids
        assert all(
            catalog.connection.execute(f"SELECT count(*) FROM {name}").fetchone()[0] == count
            for name, count in original_counts.items()
        )
        assert catalog.ingest(second) == ids
        assert len(catalog.list()) == 1
        assert len(catalog.show(ids[0]).locations) == 2
        assert catalog.list() == catalog.list()
        assert compare(catalog, ids[0], ids[0]).compatibility == "compatible"
        assert catalog.connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_ingestion_transaction_rolls_back_after_partial_insert(
    artifact_tree, tmp_path, monkeypatch
):
    with Catalog(tmp_path / "db") as catalog:
        original = catalog._ingest_verified
        calls = 0

        def failing(report, evidence_digest):
            nonlocal calls
            calls += 1
            value = original(report, evidence_digest)
            if calls == 2:
                raise sqlite3.OperationalError("injected transaction failure")
            return value

        monkeypatch.setattr(catalog, "_ingest_verified", failing)
        with pytest.raises(CatalogError, match="transaction failed"):
            catalog.ingest(artifact_tree[0])
        for table in (
            "experiments",
            "artifact_locations",
            "metrics",
            "candidates",
            "verification_records",
        ):
            assert catalog.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


def test_catalog_schema_and_changed_indexed_evidence(artifact_tree, tmp_path):
    path = copy_artifact(tmp_path, artifact_tree)
    db = tmp_path / "db"
    with Catalog(db) as catalog:
        identifier = catalog.ingest(path)[0]
        # Alter and re-seal evidence: the index still remembers the original bytes.
        (path / "manifest.json").unlink()
        edit(path / "metadata.json", lambda d: d.update(created_at="2026-10-04T10:00:00+00:00"))
        seal(path)
        assert verify(path).passed
        report = catalog.verify(identifier)[0]
        assert not report.passed and report.diagnostics[-1].code == "indexed_evidence_changed"
        with pytest.raises(CatalogError, match="indexed artifact changed"):
            catalog.ingest(path)
        with pytest.raises(CatalogError, match="verification FAILED"):
            compare(catalog, identifier, identifier)
        assert catalog.show(identifier).locations[0].verification == report
        catalog.connection.execute("PRAGMA user_version=99")
    with pytest.raises(CatalogError, match="unsupported catalog schema"):
        Catalog(db)


def test_missing_partial_directory_and_cannot_seal(tmp_path):
    path = tmp_path / "partial"
    path.mkdir()
    (path / "search_config.json").write_text("{}")
    assert discover(tmp_path).directories == (path,)
    assert not verify(path).passed
    with pytest.raises(ValueError, match="incomplete artifact"):
        seal(path)
    assert not (path / "manifest.json").exists()
    assert not verify(tmp_path / "missing").passed


def test_all_protected_byte_mutations_invalidate_seal(artifact_tree, tmp_path):
    path = copy_artifact(tmp_path, artifact_tree)
    for name in inventory(path):
        member = path / name
        original = member.read_bytes()
        member.write_bytes(original + b" ")
        report = verify(path)
        assert not report.passed
        assert any(d.code == "hash_mismatch" and d.file == name for d in report.diagnostics)
        member.write_bytes(original)
    assert verify(path).passed


@pytest.mark.parametrize(
    "task", ["binary_classification", "multiclass_classification", "regression"]
)
def test_task_end_to_end_catalog_and_cli(task, tmp_path, capsys):
    path = generate(tmp_path / "evidence", "benchmark", task)
    db = str(tmp_path / "catalog.sqlite3")
    assert main(["verify", str(path), "--deep", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["integrity"] == "sealed"
    assert main(["catalog", "ingest", str(path), "--database", db, "--json"]) == 0
    identifier = json.loads(capsys.readouterr().out)["experiment_ids"][0]
    assert main(["catalog", "--database", db, "show", identifier]) == 0
    assert "top measured candidate" in capsys.readouterr().out
    assert main(["catalog", "list", "--database", db, "--task", task]) == 0
    capsys.readouterr()
    assert main(["catalog", "compare", identifier, identifier, "--database", db, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["compatibility"] == "compatible"
    with (path / "leaderboard.csv").open("a") as handle:
        handle.write("tampered")
    assert main(["verify", str(path)]) == 2
    assert "verification FAILED" in capsys.readouterr().out
    assert main(["catalog", "verify", identifier, "--database", db]) == 2
    assert "leaderboard.csv" in capsys.readouterr().out


@pytest.mark.parametrize(
    "change,expected",
    [
        ("same", "compatible"),
        ("hyperparameters", "compatible"),
        ("dataset", "incompatible"),
        ("task", "incompatible"),
        ("metric", "incompatible"),
        ("direction", "incompatible"),
        ("workflow", "incompatible"),
        ("evaluation", "partially compatible"),
        ("missing", "partially compatible"),
        ("failed", "partially compatible"),
        ("candidates", "partially compatible"),
    ],
)
def test_comparison_compatibility_is_conservative(artifact_tree, change, expected):
    a = verify(artifact_tree[1]["search"]).descriptor
    b = a
    if change == "hyperparameters":
        changed = a.candidates[0].model_copy(update={"parameters": {"new": 2}})
        b = a.model_copy(update={"candidates": (changed, *a.candidates[1:])})
    elif change == "dataset":
        b = a.model_copy(update={"dataset_identity": "different-content"})
    elif change == "task":
        b = a.model_copy(update={"task": "regression"})
    elif change == "metric":
        b = a.model_copy(update={"primary_metric": "f1"})
    elif change == "direction":
        b = a.model_copy(update={"optimization_direction": "minimize"})
    elif change == "workflow":
        b = a.model_copy(update={"artifact_type": ArtifactType.BENCHMARK})
    elif change == "evaluation":
        b = a.model_copy(update={"evaluation": {"seed": 12}})
    elif change == "missing":
        b = a.model_copy(
            update={"candidates": tuple(c.model_copy(update={"metrics": {}}) for c in a.candidates)}
        )
    elif change == "failed":
        b = a.model_copy(update={"status": "failed", "failure_count": 2})
    elif change == "candidates":
        b = a.model_copy(update={"candidates": a.candidates[:1]})
    comparison = compare_descriptors(a, b, "a", "b")
    assert comparison.compatibility == expected
    if expected == "incompatible":
        assert comparison.differences == ()
    if change == "hyperparameters":
        assert comparison.differences[0].parameters_changed


def test_minimized_metric_comparison_retains_natural_scores(tmp_path):
    path = generate(tmp_path, "benchmark", "regression")
    a = verify(path).descriptor
    b = a.model_copy(
        update={
            "candidates": tuple(
                c.model_copy(
                    update={
                        "metrics": {
                            "root_mean_squared_error": c.metrics["root_mean_squared_error"] - 1
                        }
                    }
                )
                for c in a.candidates
            )
        }
    )
    result = compare_descriptors(a, b, "a", "b")
    assert all(
        d.raw_delta_b_minus_a == -1 and d.oriented_delta_b_minus_a == 1 for d in result.differences
    )


def test_resolved_run_identity_uses_content_and_ignores_source_relocation(tmp_path):
    source = Path("configs/examples/data/mixed_customers.csv")
    copied = tmp_path / "copy.csv"
    copied.write_bytes(source.read_bytes())
    data = configuration()
    data["model"] = data.pop("models")[1]
    data.pop("primary_metric")
    data["dataset"] = FileDatasetConfig(
        source="csv",
        path=source,
        target_column="subscribed",
        id_columns=("customer_id",),
        task=TaskType.BINARY_CLASSIFICATION,
    )
    first_config = RunConfig.model_validate(data)
    moved = first_config.model_copy(
        update={"dataset": first_config.dataset.model_copy(update={"path": copied})}
    )
    first = run_benchmark(first_config, persist=False)
    second = run_benchmark(moved, persist=False)
    assert first.fingerprint == second.fingerprint and first.predictions == second.predictions
    copied.write_text(copied.read_text().replace("29,", "28,"))
    changed = run_benchmark(moved, persist=False)
    assert first.dataset_identity != changed.dataset_identity
    assert first.fingerprint != changed.fingerprint


def test_metrics_and_candidates_order_do_not_change_identity(tmp_path):
    data = configuration(metrics=["accuracy", "f1"])
    first = BenchmarkConfig.model_validate(data)
    data["metrics"].reverse()
    data["models"].reverse()
    second = BenchmarkConfig.model_validate(data)
    assert first.fingerprint_for_dataset("content") == second.fingerprint_for_dataset("content")


def test_partial_and_all_failed_robustness_evidence_is_verifiable(tmp_path):
    data = configuration(
        output={"directory": str(tmp_path)}, robustness={"repetitions": 2, "bootstrap_samples": 100}
    )
    for model in data["models"]:
        model["parameters"] = {"not_a_parameter": 1}
    with pytest.raises(RobustnessExecutionError) as captured:
        run_robustness(RobustnessConfig.model_validate(data))
    path = captured.value.result.artifact_directory
    report = verify(path, deep=True)
    assert report.passed and report.descriptor.status == "failed"
    with Catalog(tmp_path / "db") as catalog:
        ids = catalog.ingest(path)
        assert catalog.list(status="failed")[0].experiment_id == ids[0]


@pytest.mark.parametrize(
    "kind,member",
    [
        ("search", "families/linear/summary.json"),
        ("robustness", "repetitions/repetition_000/candidates/linear/summary.json"),
        ("automl", "search/tuned_leaderboard.json"),
    ],
)
def test_tampered_candidate_summaries(artifact_tree, tmp_path, kind, member):
    path = copy_artifact(tmp_path, artifact_tree, kind)
    target = path / member
    data = json.loads(target.read_text())
    if isinstance(data, dict):
        data["model_identifier"] = "unknown-candidate"
    else:
        data[0]["model_identifier"] = "unknown-candidate"
    target.write_text(json.dumps(data))
    report = verify(path, deep=True)
    assert not report.passed
    assert any(d.file == member and d.code == "hash_mismatch" for d in report.diagnostics)
    assert any(d.code == "invalid_evidence" for d in report.diagnostics)


@pytest.mark.parametrize(
    "kind,member,field,value",
    [
        ("search", "families/linear/outer_fold_0/outer_result.json", "completed_trials", 99),
        ("search", "final_candidate.json", "model_identifier", "missing"),
        ("search", "tuned_leaderboard.json", "mode", "unknown"),
        ("robustness", "pairwise_comparisons.json", "mean_delta", 99),
        ("robustness", "rank_stability.json", "ranked_repetitions", 99),
    ],
)
def test_semantic_verification_independent_of_seal(
    artifact_tree, tmp_path, kind, member, field, value
):
    path = copy_artifact(tmp_path, artifact_tree, kind)
    (path / "manifest.json").unlink()
    data = json.loads((path / member).read_text())
    if isinstance(data, list):
        data[0][field] = value
    else:
        data[field] = value
    (path / member).write_text(json.dumps(data))
    report = verify(path, deep=True, _allow_unsealed=True)
    assert not report.passed and any(d.code == "invalid_evidence" for d in report.diagnostics)


def test_legacy_fingerprint_collision_does_not_merge_different_data(artifact_tree, tmp_path):
    path = copy_artifact(tmp_path, artifact_tree)
    config = RunConfig.model_validate_json((path / "config.json").read_text())
    (path / "manifest.json").unlink()
    edit(
        path / "metadata.json",
        lambda d: (
            d.update(config_fingerprint=config.fingerprint, benchforge_version="0.7.0"),
            d.pop("fingerprint_method"),
        ),
    )
    second = Path(shutil.copytree(path, tmp_path / "replica"))
    edit(
        second / "metadata.json",
        lambda d: (
            d.update(dataset_identity="other:content"),
            d["dataset_summary"].update(identity="other:content"),
        ),
    )
    assert verify(path, deep=True).passed and verify(second, deep=True).passed
    with Catalog(tmp_path / "db") as catalog:
        catalog.ingest(path)
        with pytest.raises(CatalogError, match="fingerprint collision"):
            catalog.ingest(second)
        assert len(catalog.list()) == len(catalog.list()[0].locations) == 1


def test_missing_v08_seal_is_failure_and_staging_can_seal(artifact_tree, tmp_path):
    path = copy_artifact(tmp_path, artifact_tree)
    (path / "manifest.json").unlink()
    edit(path / "metadata.json", lambda d: d.update(benchforge_version="0.8.0"))
    report = verify(path)
    assert not report.passed and any("required manifest" in d.message for d in report.diagnostics)
    seal(path)
    assert verify(path).passed
    with pytest.raises(ValueError, match="already sealed"):
        seal(path)


def test_duplicate_json_keys_and_nonfinite_values_fail(artifact_tree, tmp_path):
    path = copy_artifact(tmp_path, artifact_tree)
    metadata = path / "metadata.json"
    original = metadata.read_bytes()
    metadata.write_text('{"seed": 1, "seed": 2}')
    assert any("duplicate JSON key" in d.message for d in verify(path).diagnostics)
    metadata.write_text('{"value": NaN}')
    assert any("non-finite" in d.message for d in verify(path).diagnostics)
    metadata.write_bytes(original)
    assert verify(path).passed


def test_database_failures_have_context(artifact_tree, tmp_path):
    with Catalog(tmp_path / "db") as catalog:
        identifier = catalog.ingest(artifact_tree[1]["run"])[0]
        catalog.connection.close()
        with pytest.raises(CatalogError, match="catalog show failed"):
            catalog.show(identifier)


def test_discovery_reports_corruption_and_missing_signatures(artifact_tree, tmp_path):
    path = copy_artifact(tmp_path, artifact_tree)
    (path / "config.json").unlink()
    discovery = discover(tmp_path)
    assert discovery.directories == ()
    assert any(d.level == "FAIL" and "unsupported" in d.message for d in discovery.diagnostics)


@pytest.mark.parametrize("kind", ["benchmark", "search", "automl", "robustness"])
def test_manifest_mutation_invariant_for_composed_artifacts(artifact_tree, tmp_path, kind):
    path = copy_artifact(tmp_path, artifact_tree, kind)
    manifest = Manifest.model_validate_json((path / "manifest.json").read_text())
    for member in manifest.files:
        target = path / member.path
        original = target.read_bytes()
        target.write_bytes(original + b" ")
        report = verify(path)
        assert not report.passed
        assert any(d.file == member.path and d.code == "hash_mismatch" for d in report.diagnostics)
        target.write_bytes(original)
    assert verify(path).passed


def test_catalog_stores_summaries_without_trial_histories(artifact_tree, tmp_path):
    with Catalog(tmp_path / "db") as catalog:
        for kind in ("search", "automl"):
            identifier = catalog.ingest(artifact_tree[1][kind])[0]
            summary = catalog.show(identifier).locations[0].descriptor.summary_evidence
            final = summary["final_candidate"]
            if final is not None:
                assert "trials" not in final


def test_catalog_records_missing_or_symlink_replaced_locations(artifact_tree, tmp_path):
    path = copy_artifact(tmp_path, artifact_tree)
    with Catalog(tmp_path / "db") as catalog:
        identifier = catalog.ingest(path)[0]
        moved = tmp_path / "moved"
        path.rename(moved)
        report = catalog.verify(identifier)[0]
        assert not report.passed
        assert catalog.show(identifier).locations[0].verification == report
        path.symlink_to(moved, target_is_directory=True)
        report = catalog.verify(identifier)[0]
        assert not report.passed
        assert catalog.show(identifier).locations[0].verification == report
        path.unlink()
        assert catalog.ingest(moved) == (identifier,)
        assert len(catalog.show(identifier).locations) == 2


def test_offline_verification_does_not_resolve_source_data(tmp_path):
    source = tmp_path / "source.csv"
    source.write_bytes(Path("configs/examples/data/mixed_customers.csv").read_bytes())
    data = configuration(
        dataset={
            "source": "csv",
            "path": str(source),
            "target_column": "subscribed",
            "id_columns": ["customer_id"],
            "task": "binary_classification",
        },
        output={"directory": str(tmp_path / "evidence")},
    )
    data["model"] = data.pop("models")[1]
    data.pop("primary_metric")
    result = run_benchmark(RunConfig.model_validate(data))
    source.unlink()
    report = verify(result.artifact_directory, deep=True)
    assert report.passed and report.integrity == "sealed"
    with Catalog(tmp_path / "db") as catalog:
        assert len(catalog.ingest(result.artifact_directory)) == 1


def test_reproductions_and_transient_failures_remain_distinct_observations(tmp_path, monkeypatch):
    import benchforge.execution.benchmark as benchmark

    first = generate(tmp_path, "benchmark")
    original = benchmark.run_benchmark

    def transient(config, **kwargs):
        if config.model.id == "linear":
            raise RuntimeError("temporary estimator failure")
        return original(config, **kwargs)

    monkeypatch.setattr(benchmark, "run_benchmark", transient)
    second = generate(tmp_path, "benchmark")
    assert verify(second, deep=True).descriptor.status == "partial"
    with Catalog(tmp_path / "db") as catalog:
        identifier = catalog.ingest(first)[0]
        assert catalog.ingest(second) == (identifier,)
        locations = catalog.show(identifier).locations
        assert len(locations) == 2
        assert {location.descriptor.status for location in locations} == {"success", "partial"}
        assert locations[0].evidence_digest != locations[1].evidence_digest
        result = compare(catalog, identifier, identifier)
        assert result.compatibility == "partially compatible"
        assert any("replicas have different" in reason for reason in result.reasons)


def test_invalid_creation_metadata_cannot_be_sealed(artifact_tree, tmp_path):
    path = copy_artifact(tmp_path, artifact_tree)
    (path / "manifest.json").unlink()
    edit(path / "metadata.json", lambda d: d.update(created_at="not-a-timestamp"))
    with pytest.raises(ValueError, match="invalid creation timestamp"):
        seal(path)
    assert not (path / "manifest.json").exists()


def test_empty_prediction_files_require_the_real_header(tmp_path):
    data = configuration(
        output={"directory": str(tmp_path)}, robustness={"repetitions": 2, "bootstrap_samples": 100}
    )
    for model in data["models"]:
        model["parameters"] = {"not_a_parameter": 1}
    with pytest.raises(RobustnessExecutionError) as captured:
        run_robustness(RobustnessConfig.model_validate(data))
    path = captured.value.result.artifact_directory
    member = next(path.glob("repetitions/*/candidates/*/predictions.csv"))
    member.write_text("wrong,header\n")
    report = verify(path, deep=True)
    assert not report.passed and any(
        "required CSV columns" in d.message for d in report.diagnostics
    )


def test_historical_run_with_unavailable_dataset_summary(artifact_tree, tmp_path):
    path = copy_artifact(tmp_path, artifact_tree)
    (path / "manifest.json").unlink()
    config = RunConfig.model_validate_json((path / "config.json").read_text())
    edit(
        path / "metadata.json",
        lambda d: (
            d.update(config_fingerprint=config.fingerprint, benchforge_version="0.7.0"),
            d.pop("fingerprint_method"),
            d.pop("dataset_summary"),
        ),
    )
    report = verify(path, deep=True)
    assert report.passed and report.integrity == "legacy/unsealed"
    assert any(d.code == "historical_metadata_unavailable" for d in report.diagnostics)
    with Catalog(tmp_path / "db") as catalog:
        assert len(catalog.ingest(path)) == 1
