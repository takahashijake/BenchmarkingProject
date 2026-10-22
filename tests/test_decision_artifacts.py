"""Rehashed semantic adversaries, staged publication and offline integration."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from benchforge.artifacts.discovery import classify, discover, inventory
from benchforge.artifacts.manifest import digest
from benchforge.artifacts.types import ArtifactType
from benchforge.artifacts.verify import verify
from benchforge.catalog import Catalog
from benchforge.catalog.compare import compare_descriptors
from benchforge.cli import main
from benchforge.decision import DecisionConfig, load_decision_config, run_decision_analysis
from benchforge.reporting import (
    VerificationError,
    generate_report,
    load_comparison,
    load_report,
    render_report,
)
from benchforge.storage.decision import DecisionArtifactStore
from tests.test_reporting import check_html


@pytest.fixture(scope="module")
def decision_artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("decision-evidence")
    config = DecisionConfig.model_validate(
        {
            "dataset": {"name": "breast_cancer"},
            "outer_split": {"n_splits": 2},
            "inner_split": {"n_splits": 2},
            "models": [
                {"name": "logistic_regression", "id": "logistic", "parameters": {"max_iter": 1000}},
                {"name": "dummy_classifier", "id": "dummy"},
            ],
            "calibration": {"method": "sigmoid", "n_splits": 2},
            "threshold": {
                "policy": "optimize",
                "objective": "misclassification_cost",
                "false_positive_cost": 7,
                "false_negative_cost": 2,
            },
            "primary_metric": "selected.misclassification_cost",
            "output": {"directory": str(root)},
        }
    )
    result = run_decision_analysis(config)
    return config, result


def copy_artifact(decision_artifact, tmp_path):
    return Path(shutil.copytree(decision_artifact[1].artifact_directory, tmp_path / "copy"))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rewrite(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def refresh_hashes(directory):
    manifest = read(directory / "manifest.json")
    manifest["files"] = [
        {"path": name, "sha256": digest(directory / name)} for name in inventory(directory)
    ]
    rewrite(directory / "manifest.json", manifest)


def test_manifest_deep_verification_discovery_and_catalog_idempotency(decision_artifact, tmp_path):
    path = decision_artifact[1].artifact_directory
    report = verify(path, deep=True)
    assert report.passed and report.integrity == "sealed", report
    assert classify(path) == ArtifactType.DECISION
    assert discover(path.parent).directories == (path,)
    manifest = read(path / "manifest.json")
    assert manifest["experiment_fingerprint"] == report.descriptor.fingerprint
    assert [m["path"] for m in manifest["files"]] == list(inventory(path))
    assert all(m["sha256"] == digest(path / m["path"]) for m in manifest["files"])
    second = copy_artifact(decision_artifact, tmp_path)
    with Catalog(tmp_path / "catalog.sqlite") as catalog:
        first_ids = catalog.ingest(path)
        assert catalog.ingest(path) == first_ids
        assert catalog.ingest(second) == first_ids
        assert len(catalog.list(artifact_type="decision")) == 1
        experiment = catalog.show(first_ids[0])
        assert len(experiment.locations) == 2
        assert all(v.passed for v in catalog.verify(first_ids[0], deep=True))


def mutate_result(result, mutation):
    c = result["candidates"][0]
    f = c["folds"][0]
    plan = result["plan"]["outer"][0]
    if mutation == "probability":
        f["predictions"][0]["probability"] = 0.333333
    elif mutation == "range":
        f["predictions"][0]["probability"] = 1.1
    elif mutation == "prediction":
        f["predictions"][0]["prediction"] ^= 1
    elif mutation == "reference_prediction":
        f["predictions"][0]["reference_prediction"] ^= 1
    elif mutation == "dataset_truth":
        result["dataset"]["truth"][0] ^= 1
    elif mutation == "truth":
        f["predictions"][0]["truth"] ^= 1
    elif mutation == "outer_fold":
        plan["final"]["fold"]["train_indices"][0] = plan["final"]["fold"]["validation_indices"][0]
    elif mutation == "inner_fold":
        plan["inner"][0]["fold"]["train_indices"][0] = plan["final"]["fold"]["validation_indices"][
            0
        ]
    elif mutation == "calibration_fold":
        plan["inner"][0]["calibration"]["folds"][0]["train_indices"][0] = plan["inner"][0]["fold"][
            "validation_indices"
        ][0]
    elif mutation == "missing_calibration":
        del plan["inner"][0]["calibration"]
    elif mutation == "null_calibration":
        plan["inner"][0]["calibration"] = None
    elif mutation == "calibration_seed":
        plan["final"]["calibration"]["seed"] ^= 1
    elif mutation == "estimator_seed":
        f["inner"][0]["estimator_seed"] ^= 1
    elif mutation == "threshold":
        f["selection"]["threshold"] = 0.123456
    elif mutation == "objective":
        f["selection"]["objective_value"] += 0.1
    elif mutation == "selection_curve":
        f["selection"]["trials"][0]["confusion"]["fp"] += 1
    elif mutation == "confusion":
        f["selected"]["confusion"]["fp"] += 1
    elif mutation == "cost":
        next(m for m in f["selected"]["metrics"] if m["name"] == "misclassification_cost")[
            "value"
        ] += 1
    elif mutation == "bins":
        f["probability_quality"]["bins"][0]["count"] += 1
    elif mutation == "brier":
        f["probability_quality"]["brier_score"] += 0.01
    elif mutation == "ece":
        f["probability_quality"]["ece"] += 0.01
    elif mutation == "log_loss":
        f["probability_quality"]["log_loss"] += 0.01
    elif mutation == "aggregate":
        c["metrics"][0]["mean"] += 0.1
    elif mutation == "rank":
        c["rank"] = 999
    elif mutation == "status":
        c["status"] = "partial"
    elif mutation == "missing_inner":
        f["inner"].pop()
    elif mutation == "inner_probability":
        f["inner"][0]["probabilities"][0]["probability"] = 0.333
    elif mutation == "sample_assignment":
        f["inner"][0]["probabilities"][0]["sample_index"] = plan["final"]["fold"][
            "validation_indices"
        ][0]
    elif mutation == "duplicate_prediction":
        f["predictions"].append(f["predictions"][0])
    elif mutation == "schema":
        result["artifact_schema_version"] = 2
    elif mutation == "extra_field":
        f["probabilities"] = []
    elif mutation == "runtime_path":
        result["artifact_directory"] = "fake"
    else:
        raise AssertionError(mutation)


@pytest.mark.parametrize(
    "mutation",
    [
        "dataset_truth",
        "probability",
        "range",
        "prediction",
        "reference_prediction",
        "truth",
        "outer_fold",
        "inner_fold",
        "calibration_fold",
        "missing_calibration",
        "null_calibration",
        "calibration_seed",
        "estimator_seed",
        "threshold",
        "objective",
        "selection_curve",
        "confusion",
        "cost",
        "bins",
        "brier",
        "ece",
        "log_loss",
        "aggregate",
        "rank",
        "status",
        "missing_inner",
        "inner_probability",
        "sample_assignment",
        "duplicate_prediction",
        "schema",
        "extra_field",
        "runtime_path",
    ],
)
def test_rehashed_semantic_corruption_is_rejected(decision_artifact, tmp_path, mutation):
    path = copy_artifact(decision_artifact, tmp_path)
    member = path / "decision_result.json"
    result = read(member)
    mutate_result(result, mutation)
    rewrite(member, result)
    refresh_hashes(path)
    for deep in (False, True):
        report = verify(path, deep=deep)
        assert not report.passed, mutation
        assert not any(d.code == "hash_mismatch" for d in report.diagnostics)
        assert any(d.code in {"invalid_evidence", "unsupported_schema"} for d in report.diagnostics)
    output = tmp_path / "rejected.html"
    with pytest.raises(VerificationError):
        generate_report(path, output, deep=True)
    assert not output.exists()


def test_byte_corruption_and_missing_evidence_rejected(decision_artifact, tmp_path):
    path = copy_artifact(decision_artifact, tmp_path)
    member = path / "decision_result.json"
    result = read(member)
    mutate_result(result, "prediction")
    rewrite(member, result)
    report = verify(path, deep=True)
    assert not report.passed and any(d.code == "hash_mismatch" for d in report.diagnostics)
    member.unlink()
    refresh_hashes(path)
    assert not verify(path).passed


@pytest.mark.parametrize(
    "member", ["decision_config.json", "decision_result.json", "metadata.json"]
)
def test_new_contract_never_allows_missing_seal(decision_artifact, tmp_path, member):
    path = copy_artifact(decision_artifact, tmp_path)
    (path / "manifest.json").unlink()
    assert not verify(path).passed
    with pytest.raises(VerificationError):
        generate_report(path, tmp_path / "report.html", allow_legacy=True)
    (path / member).unlink()
    assert not verify(path).passed


@pytest.mark.parametrize(
    "change", ["config_version", "metadata_version", "methodology", "versions"]
)
def test_rehashed_identity_and_version_changes_rejected(decision_artifact, tmp_path, change):
    path = copy_artifact(decision_artifact, tmp_path)
    filename = "decision_config.json" if change == "config_version" else "metadata.json"
    value = read(path / filename)
    if change.endswith("version"):
        value["schema_version" if change == "config_version" else "artifact_schema_version"] = 2
    elif change == "methodology":
        value["methodology_version"] = "unknown-v9"
    else:
        value["versions"].pop("scikit_learn")
    rewrite(path / filename, value)
    refresh_hashes(path)
    assert not verify(path, deep=True).passed


def test_reports_verification_catalog_are_offline_without_models_or_providers(
    monkeypatch,
    decision_artifact,
    tmp_path,
):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline

    from benchforge.data.registry import default_dataset_registry
    from benchforge.models.registry import default_model_registry
    from benchforge.providers import runtime

    def forbidden(*a, **kw):
        pytest.fail(
            "offline path attempted training, model construction, data loading or activation"
        )

    monkeypatch.setattr(Pipeline, "fit", forbidden)
    monkeypatch.setattr(CalibratedClassifierCV, "fit", forbidden)
    monkeypatch.setattr(default_dataset_registry, "resolve", forbidden)
    monkeypatch.setattr(default_model_registry, "create", forbidden)
    monkeypatch.setattr(runtime, "activate", forbidden)
    path = decision_artifact[1].artifact_directory
    original = {name: digest(path / name) for name in inventory(path)}
    assert verify(path, deep=True).passed
    model = load_report(path, deep=True)
    content = render_report(model)
    assert content == render_report(load_report(path, deep=True))
    check_html(content)
    assert "Selected policy versus fixed 0.5 reference" in content
    assert "Held-out probability quality" in content
    assert "inner threshold selection" in content
    assert "distribution shift" in content and "not a validated deployment threshold" in content
    assert "sample_index" not in content
    first = generate_report(path, tmp_path / "first.html", deep=True)
    second = generate_report(path, tmp_path / "second.html", deep=True)
    assert first.read_bytes() == second.read_bytes()
    with Catalog(tmp_path / "catalog.sqlite") as catalog:
        assert catalog.ingest(path)
    comparison = load_comparison(path, path, deep=True)
    assert comparison.compatibility == "compatible"
    assert {name: digest(path / name) for name in inventory(path)} == original


def test_decision_policy_changes_are_conservatively_incompatible(decision_artifact, tmp_path):
    config, result = decision_artifact
    a = verify(result.artifact_directory).descriptor
    raw = config.model_dump(mode="json")
    raw["threshold"]["false_positive_cost"] = 3
    raw["output"]["directory"] = str(tmp_path)
    b_result = run_decision_analysis(DecisionConfig.model_validate(raw))
    b = verify(b_result.artifact_directory).descriptor
    assert a.fingerprint != b.fingerprint
    assert compare_descriptors(a, b, "A", "B").compatibility == "partially compatible"
    comparison = load_comparison(a.directory, b.directory)
    assert comparison.compatibility == "partially compatible"
    assert all(c.delta_b_minus_a is None for c in comparison.changes)


@pytest.mark.parametrize("stage", ["first_json", "second_json", "seal", "rename"])
def test_atomic_failure_cleanup_preserves_no_published_experiment(
    monkeypatch,
    decision_artifact,
    tmp_path,
    stage,
):
    from benchforge.storage import decision as storage

    config, result = decision_artifact
    calls = 0
    original_write = storage._write_atomic
    original_rename = os.rename

    def write(path, value):
        nonlocal calls
        calls += 1
        if calls == (1 if stage == "first_json" else 2) and stage in {"first_json", "second_json"}:
            raise OSError("injected write failure")
        original_write(path, value)

    def seal(path):
        raise OSError("injected sealing failure")

    def rename(source, target):
        if Path(source).is_dir():
            assert verify(source, deep=True).passed
            assert not Path(target).exists()
            raise OSError("injected publication failure")
        return original_rename(source, target)

    monkeypatch.setattr(storage, "_write_atomic", write)
    if stage == "seal":
        monkeypatch.setattr(storage, "seal", seal)
    if stage == "rename":
        monkeypatch.setattr(storage.os, "rename", rename)
    root = tmp_path / "published"
    with pytest.raises(OSError, match="injected"):
        DecisionArtifactStore(root).write(config, result)
    assert root.is_dir() and not [p for p in root.iterdir() if p.is_dir()]


def test_successful_publication_moves_only_complete_sealed_evidence(
    monkeypatch,
    decision_artifact,
    tmp_path,
):
    from benchforge.storage import decision as storage

    original = os.rename
    moved = []

    def observed(source, target):
        if Path(source).is_dir():
            assert verify(source, deep=True).passed
            assert not (Path(target) / "decision_config.json").exists()
            moved.append(Path(target))
        return original(source, target)

    monkeypatch.setattr(storage.os, "rename", observed)
    path = DecisionArtifactStore(tmp_path).write(*decision_artifact)
    assert moved == [path] and verify(path, deep=True).passed
    assert not list(tmp_path.glob(".decision-stage-*"))


def test_output_cannot_be_inside_sealed_evidence_or_workspace(decision_artifact, tmp_path):
    config, result = decision_artifact
    with pytest.raises(ValueError, match="outside existing"):
        DecisionArtifactStore(result.artifact_directory / "nested").write(config, result)
    (tmp_path / "workspace.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="outside existing"):
        DecisionArtifactStore(tmp_path / "nested").write(config, result)


def test_file_dataset_label_mapping_and_reporting_after_original_removed(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text(
        "x,category,target\n"
        + "\n".join(
            f"{i % 7},{'a' if i % 3 else 'b'},{'disease' if i % 2 else 'control'}"
            for i in range(80)
        ),
        encoding="utf-8",
    )
    config = DecisionConfig.model_validate(
        {
            "dataset": {
                "source": "csv",
                "path": str(source),
                "target_column": "target",
                "task": "binary_classification",
            },
            "models": [{"name": "logistic_regression"}],
            "outer_split": {"n_splits": 2},
            "inner_split": {"n_splits": 2},
            "calibration": {"method": "sigmoid", "n_splits": 2},
            "output": {"directory": str(tmp_path / "artifacts")},
        }
    )
    result = run_decision_analysis(config)
    assert result.dataset.target_labels == ("control", "disease")
    source.unlink()
    assert verify(result.artifact_directory, deep=True).passed
    assert generate_report(result.artifact_directory, tmp_path / "report.html", deep=True).is_file()


def test_cli_subprocess_repeated_run_verify_catalog_report(tmp_path):
    config = tmp_path / "decision.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "dataset": {"name": "breast_cancer"},
                "models": [{"name": "dummy_classifier"}],
                "outer_split": {"n_splits": 2},
                "inner_split": {"n_splits": 2},
                "threshold": {"policy": "optimize"},
                "output": {"directory": str(tmp_path / "artifacts")},
            }
        ),
        encoding="utf-8",
    )

    def command(*args):
        return subprocess.run(
            [sys.executable, "-m", "benchforge", *map(str, args)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )

    first = command("decision", config)
    second = command("decision", config)
    paths = sorted(p for p in (tmp_path / "artifacts").iterdir() if p.is_dir())
    assert len(paths) == 2
    one, two = [read(p / "decision_result.json") for p in paths]
    assert one == two
    assert "per-fold thresholds" in first.stdout and "reference 0.5" in second.stdout
    assert command("verify", paths[0], "--deep", "--json").returncode == 0
    command("report", paths[0], "--deep", "--output", tmp_path / "report.html")
    command("catalog", "ingest", paths[0], "--database", tmp_path / "db")
    with pytest.raises(SystemExit) as exc:
        main(["decision", str(config), "--workers", "2"])
    assert exc.value.code == 2


def test_load_config_errors_and_schema_roundtrip(tmp_path):
    for value in ("[]", "{broken:"):
        path = tmp_path / "invalid.yaml"
        path.write_text(value, encoding="utf-8")
        with pytest.raises(ValueError):
            load_decision_config(path)
    with pytest.raises(ValueError):
        load_decision_config(tmp_path / "missing.yaml")


def test_catalog_rejects_different_resolved_content_with_same_source_name(
    monkeypatch,
    decision_artifact,
    tmp_path,
):
    from dataclasses import replace

    from benchforge.data.registry import default_dataset_registry

    config, old = decision_artifact
    dataset = default_dataset_registry.resolve(config.dataset)
    features = dataset.features.copy()
    features.iloc[0, 0] += 1
    monkeypatch.setattr(
        default_dataset_registry, "resolve", lambda *a: replace(dataset, features=features)
    )
    config = DecisionConfig.model_validate(
        {**config.model_dump(mode="json"), "output": {"directory": str(tmp_path)}}
    )
    new = run_decision_analysis(config)
    assert new.dataset.identity == old.dataset.identity
    assert new.dataset.content_digest != old.dataset.content_digest
    assert new.fingerprint != old.fingerprint
    a, b = verify(old.artifact_directory).descriptor, verify(new.artifact_directory).descriptor
    assert a.dataset_identity != b.dataset_identity
    assert compare_descriptors(a, b, "A", "B").compatibility == "incompatible"


def test_staging_is_not_discovered_and_collision_cannot_overwrite(
    monkeypatch,
    decision_artifact,
    tmp_path,
):
    from benchforge.storage import decision as storage

    original = os.rename
    observations = []

    def publish(source, target):
        assert discover(tmp_path).directories == ()
        assert verify(source, deep=True).passed
        observations.append(True)
        return original(source, target)

    monkeypatch.setattr(storage.os, "rename", publish)
    path = DecisionArtifactStore(tmp_path).write(*decision_artifact)
    assert observations == [True] and discover(tmp_path).directories == (path,)
    # Force a deterministic destination collision without changing existing bytes.
    name = path.name
    monkeypatch.setattr(storage, "uuid4", lambda: type("UUID", (), {"hex": name.split("_")[-1]})())
    created = read(path / "metadata.json")["created_at"]
    from datetime import datetime

    class FrozenDatetime:
        @staticmethod
        def now(zone):
            return datetime.fromisoformat(created)

    monkeypatch.setattr(storage, "datetime", FrozenDatetime)
    before = {n: digest(path / n) for n in inventory(path)}
    with pytest.raises(FileExistsError):
        DecisionArtifactStore(tmp_path).write(*decision_artifact)
    assert before == {n: digest(path / n) for n in inventory(path)}
    assert not list(tmp_path.glob(".decision-stage-*"))


def test_all_failed_candidates_persist_verify_and_report(tmp_path):
    from benchforge.decision import DecisionAnalysisError

    config = DecisionConfig.model_validate(
        {
            "dataset": {"name": "breast_cancer"},
            "models": [{"name": "dummy_classifier", "parameters": {"strategy": "invalid"}}],
            "outer_split": {"n_splits": 2},
            "inner_split": {"n_splits": 2},
            "output": {"directory": str(tmp_path / "evidence")},
        }
    )
    with pytest.raises(DecisionAnalysisError) as exc:
        run_decision_analysis(config)
    result = exc.value.result
    assert verify(result.artifact_directory, deep=True).passed
    report = load_report(result.artifact_directory, deep=True)
    assert report.status == "failed" and all(c.rank is None for c in report.candidates)
    assert generate_report(result.artifact_directory, tmp_path / "failed.html").exists()


def test_outer_costs_and_aggregate_means_independently_reconstructed(decision_artifact):
    result = decision_artifact[1]
    for c in result.candidates:
        costs = []
        for fold in c.folds:
            p = fold.predictions
            fp = sum(row.truth == 0 and row.prediction == 1 for row in p)
            fn = sum(row.truth == 1 and row.prediction == 0 for row in p)
            cost = (7 * fp + 2 * fn) / len(p)
            actual = next(
                m.value for m in fold.selected.metrics if m.name == "misclassification_cost"
            )
            assert actual == cost
            costs.append(cost)
        mean = sum(costs) / len(costs)
        variance = sum((cost - mean) ** 2 for cost in costs) / len(costs)
        metric = next(m for m in c.metrics if m.name == "selected.misclassification_cost")
        assert metric.mean == pytest.approx(mean) and metric.std == pytest.approx(variance**0.5)


def test_html_target_label_injection_is_text_and_escaped(tmp_path):
    source = tmp_path / "injection.csv"
    source.write_text(
        "x,target\n"
        + "\n".join(
            f'{i},"<script>alert(1)</script>"' if i % 2 else f'{i},"benign"' for i in range(40)
        ),
        encoding="utf-8",
    )
    config = DecisionConfig.model_validate(
        {
            "dataset": {
                "source": "csv",
                "path": str(source),
                "target_column": "target",
                "task": "binary_classification",
            },
            "models": [{"name": "dummy_classifier"}],
            "outer_split": {"n_splits": 2},
            "inner_split": {"n_splits": 2},
            "output": {"directory": str(tmp_path / "evidence")},
        }
    )
    result = run_decision_analysis(config)
    content = render_report(load_report(result.artifact_directory, deep=True))
    check_html(content)
    assert "&lt;script&gt;" in content and "<script>" not in content
    assert "Calibration is disabled" in content
    assert "Calibration cross-fits" not in content


def test_new_contract_schema_files_match_public_types():
    from benchforge.decision.results import DecisionResult

    root = Path(__file__).resolve().parents[1] / "schemas"
    assert read(root / "decision-config-v1.schema.json") == DecisionConfig.model_json_schema()
    assert read(root / "decision-result-v1.schema.json") == DecisionResult.model_json_schema(
        mode="serialization"
    )


def test_original_label_meanings_are_part_of_comparison_identity(
    monkeypatch,
    decision_artifact,
    tmp_path,
):
    from dataclasses import replace

    from benchforge.data.registry import default_dataset_registry

    config, old = decision_artifact
    dataset = default_dataset_registry.resolve(config.dataset)
    monkeypatch.setattr(
        default_dataset_registry,
        "resolve",
        lambda *a: replace(dataset, target_labels=("different-negative", "different-positive")),
    )
    config = DecisionConfig.model_validate(
        {**config.model_dump(mode="json"), "output": {"directory": str(tmp_path)}}
    )
    new = run_decision_analysis(config)
    assert old.dataset.content_digest == new.dataset.content_digest
    assert old.dataset.target_labels != new.dataset.target_labels
    a, b = verify(old.artifact_directory).descriptor, verify(new.artifact_directory).descriptor
    assert a.dataset_identity != b.dataset_identity
    assert compare_descriptors(a, b, "A", "B").compatibility == "incompatible"
