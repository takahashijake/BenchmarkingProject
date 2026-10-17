"""Provider contract and installed-package integration regression coverage."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression

from benchforge.artifacts import verify
from benchforge.core.config import (
    AutoMLConfig,
    ExternalMetric,
    ModelConfig,
    OutputConfig,
    RunConfig,
    TaskType,
    load_benchmark_config,
    load_run_config,
    load_search_config,
    parse_metric,
)
from benchforge.core.provider_config import (
    Component,
    EstimatorDeclaration,
    MetricDeclaration,
    ProviderManifest,
    ProviderRef,
    validate_identifier,
)
from benchforge.data import default_dataset_registry
from benchforge.evaluation.metrics import compute_metrics, metric_spec
from benchforge.execution import ExecutionOptions, run_benchmark, run_benchmark_suite
from benchforge.models import ModelSpec, default_model_registry
from benchforge.providers import runtime
from benchforge.robustness import RobustnessConfig, run_robustness
from benchforge.search import run_search
from benchforge.search.spaces import SearchSpace


@pytest.fixture
def installed():
    # CI installs the separately packaged example; this fixture never edits registries.
    assert metadata.version("benchforge-example-provider") == "1.0.0"
    from benchforge_example import provider

    return provider()


@pytest.fixture
def config(tmp_path: Path):
    return load_benchmark_config("configs/providers/benchmark.yaml").model_copy(
        update={"output": OutputConfig(directory=tmp_path / "published")}
    )


class Entry:
    dist = None
    name = "example"
    value = "benchforge_example:provider"

    def __init__(self, factory):
        self.factory = factory
        self.loaded = 0

    def load(self):
        self.loaded += 1
        return self.factory


def select(monkeypatch, provider):
    entry = Entry(lambda: provider)
    monkeypatch.setattr(runtime, "_entries", lambda: (entry,))
    return entry


def test_discovery_does_not_import_or_activate(monkeypatch):
    entry = Entry(lambda: (_ for _ in ()).throw(AssertionError("must not execute")))
    monkeypatch.setattr(runtime, "_entries", lambda: (entry,))
    assert runtime.discover()[0].id == "example"
    assert entry.loaded == 0


@pytest.mark.parametrize(
    "value", ["..", ".", "../a", "x/y", "x\\y", "x y", "", "a\x00", "A", "a" * 129]
)
def test_unsafe_provider_identifiers(value):
    with pytest.raises(ValueError):
        validate_identifier(value)
    with pytest.raises(ValidationError):
        ProviderRef(id=value, version="1.0.0")


@pytest.mark.parametrize("value", ["1", "", "latest", "../../x", "1.0.0\n"])
def test_explicit_version_required(value):
    with pytest.raises(ValidationError):
        ProviderRef(id="example", version=value)


@pytest.mark.parametrize("api", [0, 2, True, "1"])
def test_incompatible_api_version(api):
    with pytest.raises(ValidationError):
        ProviderManifest(
            id="example",
            version="1.0.0",
            api_version=api,
            datasets=(Component(id="example.toy", tasks=("regression",)),),
        )


def test_duplicate_refs_and_discovery(monkeypatch, config):
    with pytest.raises(ValidationError, match="duplicate provider"):
        type(config).model_validate({**config.model_dump(), "providers": config.providers * 2})
    monkeypatch.setattr(runtime, "_entries", lambda: (Entry(None), Entry(None)))
    with pytest.raises(ValueError, match="duplicate installed"):
        runtime.discover()
    with pytest.raises(ValueError, match="found 2"):
        runtime.resolve(config.providers)


def test_malformed_manifest(installed):
    raw = installed.manifest.model_dump()
    raw["datasets"] = (Component(id="other.toy", tasks=("regression",)),)
    with pytest.raises(ValidationError, match="namespaced"):
        ProviderManifest.model_validate(raw)
    raw = installed.manifest.model_dump()
    raw["metrics"] *= 2
    with pytest.raises(ValidationError, match="duplicate component"):
        ProviderManifest.model_validate(raw)


def test_initialization_errors_are_actionable(monkeypatch, config):
    def broken():
        raise ImportError("install optional-extra")

    monkeypatch.setattr(runtime, "_entries", lambda: (Entry(broken),))
    with pytest.raises(ValueError, match="example.*install optional-extra"):
        runtime.resolve(config.providers)
    select(monkeypatch, object())
    with pytest.raises(ValueError, match="must return"):
        runtime.resolve(config.providers)


def test_callback_mismatch_cannot_corrupt_builtins(monkeypatch, installed, config):
    names = default_model_registry.names
    select(monkeypatch, replace(installed, estimators={}))
    with pytest.raises(ValueError, match="callbacks disagree"):
        runtime.resolve(config.providers)
    assert default_model_registry.names == names
    assert "example.logistic" not in names


def test_declared_capabilities_must_match(monkeypatch, installed, config):
    select(
        monkeypatch,
        replace(
            installed, estimators={"example.logistic": ModelSpec(lambda p, s: LogisticRegression())}
        ),
    )
    with pytest.raises(ValueError, match="capabilities disagree"):
        runtime.resolve(config.providers)


def test_version_pin_and_missing_provider(monkeypatch, installed, config):
    select(monkeypatch, installed)
    with pytest.raises(ValueError, match="version mismatch"):
        runtime.resolve((ProviderRef(id="example", version="2.0.0"),))
    monkeypatch.setattr(runtime, "_entries", lambda: ())
    with pytest.raises(ValueError, match="found 0"):
        runtime.resolve(config.providers)


def test_builtin_execution_never_discovers(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_entries", lambda: (_ for _ in ()).throw(AssertionError()))
    result = run_benchmark(
        load_run_config("configs/examples/iris_multiclass_run.yaml"), persist=False
    )
    assert result.predictions


def test_installed_benchmark_and_offline_verification(installed, config, monkeypatch):
    result = run_benchmark_suite(config)
    assert not result.failures
    assert config.provider_manifests == ()  # activation never mutates caller config
    for candidate in result.candidates:
        metrics = candidate.run_result.aggregate_metrics
        assert metrics["accuracy"] == metrics["example.agreement"]
    monkeypatch.setattr(
        runtime, "_entries", lambda: (_ for _ in ()).throw(AssertionError("offline"))
    )
    report = verify(result.artifact_directory, deep=True)
    assert report.passed, report.diagnostics
    assert any(d.code == "external_metric_not_recomputed" for d in report.diagnostics)
    stored = json.loads((result.artifact_directory / "metadata.json").read_text())
    assert stored["provider_schema_version"] == 1
    assert stored["provider_distributions"][0]["distribution_version"] == "1.0.0"


def test_installed_cli(config, tmp_path):
    source = tmp_path / "config.json"
    source.write_text(config.model_dump_json())
    for args in [
        ["plugins", "list"],
        ["plugins", "inspect", "example", "--version", "1.0.0"],
        ["benchmark", str(source)],
    ]:
        completed = subprocess.run(
            [sys.executable, "-m", "benchforge", *args], capture_output=True, text=True, check=False
        )
        assert completed.returncode == 0, completed.stderr
        assert "example" in completed.stdout


def test_installed_nested_search_and_fixed_parameters(installed, tmp_path):
    config = load_search_config("configs/providers/search.yaml").model_copy(
        update={"output": OutputConfig(directory=tmp_path)}
    )
    result = run_search(config)
    assert not result.failures
    assert result.leaderboard[0].completed_trials == 4
    assert verify(result.artifact_directory, deep=True).passed
    fixed = config.models[0].model_copy(update={"parameters": {"C": 0.7}})
    again = run_search(config.model_copy(update={"models": (fixed,)}), persist=False)
    assert all(f.selected_parameters["C"] == 0.7 for f in again.families[0].outer_folds)


def test_automl_and_robustness_use_existing_kernels(installed, config, tmp_path):
    from benchforge.automl import build_automl_plan, run_automl

    search = load_search_config("configs/providers/search.yaml")
    raw = search.model_dump()
    raw.pop("final_search")
    raw["models"] = {"include": ["example.logistic"], "include_fixed_baselines": False}
    raw["budget"] = {"total_trials": 4}
    raw["output"] = {"directory": tmp_path / "automl"}
    automl = AutoMLConfig.model_validate(raw)
    plan = build_automl_plan(automl)
    result = run_automl(automl)
    assert result.plan.fingerprint == plan.fingerprint
    assert verify(result.artifact_directory, deep=True).passed
    robust = RobustnessConfig.model_validate(
        {
            **config.model_dump(),
            "robustness": {
                "repetitions": 2,
                "bootstrap_samples": 100,
            },
        }
    )
    robustness = run_robustness(robust)
    assert robust.fingerprint_for_dataset(robustness.dataset.identity) == robustness.fingerprint
    assert verify(robustness.artifact_directory, deep=True).passed


def test_serial_checkpoint_reuse_and_version_incompatibility(
    installed, config, tmp_path, monkeypatch
):
    workspace = tmp_path / "work"
    first = run_benchmark_suite(config, checkpoint_dir=workspace, persist=False)
    assert not first.failures

    def forbidden(*args, **kwargs):
        raise AssertionError("checkpoint should not fit")

    monkeypatch.setattr("benchforge.execution.benchmark._execute_candidate", forbidden)
    second = run_benchmark_suite(config, checkpoint_dir=workspace, resume=True, persist=False)
    assert second.checkpoint_reused == ("plugin", "baseline")
    assert second.candidates == first.candidates
    newer = replace(installed, manifest=installed.manifest.model_copy(update={"version": "2.0.0"}))
    select(monkeypatch, newer)
    changed = config.model_copy(update={"providers": (ProviderRef(id="example", version="2.0.0"),)})
    with pytest.raises(ValueError, match="incompatible"):
        run_benchmark_suite(changed, checkpoint_dir=workspace, resume=True, persist=False)


@pytest.mark.parametrize("mode", ["workers", "candidate_timeout", "run_timeout"])
def test_external_supervision_rejected_before_import(config, monkeypatch, mode):
    monkeypatch.setattr(runtime, "_entries", lambda: (_ for _ in ()).throw(AssertionError()))
    kwargs = (
        {"workers": 2}
        if mode == "workers"
        else {
            "execution_options": ExecutionOptions(**{mode: 1}),
        }
    )
    with pytest.raises(ValueError, match="serial execution without hard budgets"):
        run_benchmark_suite(config, **kwargs)


def test_identity_canonicalization_and_semantic_changes(installed, config, monkeypatch):
    bound, _ = runtime.activate(config)
    fingerprint = bound.fingerprint_for_dataset("data-v1")
    reordered = bound.provider_manifests[0].model_copy(
        update={
            "estimators": tuple(reversed(bound.provider_manifests[0].estimators)),
            "metrics": tuple(reversed(bound.provider_manifests[0].metrics)),
        }
    )
    assert (
        bound.model_copy(update={"provider_manifests": (reordered,)}).fingerprint_for_dataset(
            "data-v1"
        )
        == fingerprint
    )
    changed = reordered.model_copy(update={"version": "2.0.0"})
    assert (
        bound.model_copy(update={"provider_manifests": (changed,)}).fingerprint_for_dataset(
            "data-v1"
        )
        != fingerprint
    )
    assert bound.fingerprint_for_dataset("data-v2") != fingerprint
    assert (
        bound.model_copy(
            update={"output": OutputConfig(directory=Path("elsewhere"))}
        ).fingerprint_for_dataset("data-v1")
        == fingerprint
    )
    with pytest.raises(ValueError, match="declarations differ"):
        runtime.activate(bound.model_copy(update={"provider_manifests": (changed,)}))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, [0.1], complex(1, 2)])
def test_nonfinite_and_invalid_metric_outputs(installed, value):
    metric = parse_metric("example.agreement")
    with pytest.raises((ValueError, TypeError), match="finite scalar"):
        compute_metrics(
            (metric,),
            np.array([0, 1]),
            np.array([0, 1]),
            None,
            TaskType.BINARY_CLASSIFICATION,
            manifests=(installed.manifest,),
            functions={str(metric): lambda t, p, s: value},
        )


def test_metric_task_direction_and_score_contract(installed):
    assert (
        metric_spec(parse_metric("example.agreement"), (installed.manifest,)).direction
        == "maximize"
    )
    with pytest.raises(ValidationError, match="binary"):
        MetricDeclaration(
            id="example.score",
            tasks=("regression",),
            direction="minimize",
            requires_score=True,
            display_name="score",
            semantics_version="1.0.0",
        )
    with pytest.raises(ValueError, match="unresolved external"):
        metric_spec(ExternalMetric("missing.metric"))


def test_factories_are_cloned_and_wrong_task_is_rejected():
    singleton = LogisticRegression()
    registry = default_model_registry.extended({"test.model": ModelSpec(lambda p, s: singleton)})
    first = registry.create(ModelConfig(name="test.model"), 5, TaskType.BINARY_CLASSIFICATION)
    second = registry.create(ModelConfig(name="test.model"), 5, TaskType.BINARY_CLASSIFICATION)
    assert first is not second and first is not singleton
    assert first.random_state == 5
    bad = default_model_registry.extended({"test.bad": ModelSpec(lambda p, s: DummyRegressor())})
    with pytest.raises(ValueError, match="wrong estimator task"):
        bad.create(ModelConfig(name="test.bad"), 5, TaskType.BINARY_CLASSIFICATION)


def test_fixed_parameter_mutation_cannot_escape_search_boundary():
    def mutate(trial, fixed):
        fixed["C"] = 123
        return {"C": 234}

    space = SearchSpace("test.model", 1, {}, mutate)
    fixed = {"C": 1.0}
    with pytest.raises(ValueError, match="overwrite"):
        space.propose(None, fixed)
    assert fixed == {"C": 1.0}


def test_dataset_deep_copy_and_validation(installed, config, monkeypatch):
    bound, resolved = runtime.activate(config)
    first = resolved.datasets.resolve("example.toy")
    first.features.iloc[0, 0] = 999
    assert resolved.datasets.resolve("example.toy").features.iloc[0, 0] != 999
    bad = installed.datasets["example.toy"]()
    bad.target.iloc[0] = 4
    select(monkeypatch, replace(installed, datasets={"example.toy": lambda: bad}))
    _, resolved = runtime.activate(config)
    with pytest.raises(ValueError, match="consecutive"):
        resolved.datasets.resolve("example.toy")


def test_missing_or_corrupt_provider_metadata_fails_offline(installed, config, tmp_path):
    result = run_benchmark_suite(config)
    source = result.artifact_directory / "benchmark_config.json"
    raw = json.loads(source.read_text())
    raw.pop("provider_manifests")
    source.write_text(json.dumps(raw))
    report = verify(result.artifact_directory, deep=True)
    assert not report.passed
    assert any("missing resolved provider" in d.message for d in report.diagnostics)


def test_historical_builtin_fingerprint_is_unchanged():
    config = load_run_config("configs/examples/breast_cancer_logreg.yaml")
    assert config.fingerprint == "d41548be73cbb3b91126f9f4404b7d0fdaab128609c818958f7af65c17cfadf2"
    assert "providers" not in config.canonical_dict()
    assert "provider_manifests" not in config.canonical_dict()
    assert default_dataset_registry.names == ("breast_cancer", "diabetes", "iris")


@pytest.mark.parametrize(
    "example,model,metric,task",
    [
        ("iris_multiclass_run", "logistic", "accuracy", TaskType.MULTICLASS_CLASSIFICATION),
        ("diabetes_regression_run", "ridge", "mean_absolute_error", TaskType.REGRESSION),
    ],
)
def test_external_models_and_metrics_across_tasks(monkeypatch, example, model, metric, task):
    from sklearn.linear_model import Ridge

    manifest = ProviderManifest(
        id="example",
        version="1.0.0",
        estimators=(EstimatorDeclaration(id=f"example.{model}", tasks=(task.value,)),),
        metrics=(
            MetricDeclaration(
                id="example.measure",
                tasks=(task.value,),
                direction="minimize" if task == TaskType.REGRESSION else "maximize",
                display_name="Measure",
                semantics_version="1.0.0",
            ),
        ),
    )
    function = (
        (lambda t, p, s: float(np.mean(np.abs(t - p))))
        if task == TaskType.REGRESSION
        else (lambda t, p, s: float(np.mean(t == p)))
    )
    factory = (
        (lambda p, s: Ridge(**p))
        if model == "ridge"
        else (lambda p, s: LogisticRegression(random_state=s, **p))
    )
    plugin = runtime.Provider(
        manifest,
        {},
        {f"example.{model}": ModelSpec(factory, frozenset({task}))},
        {"example.measure": function},
        {},
    )
    select(monkeypatch, plugin)
    original = load_run_config(f"configs/examples/{example}.yaml")
    configured = RunConfig.model_validate(
        {
            **original.model_dump(),
            "providers": [{"id": "example", "version": "1.0.0"}],
            "model": {
                "name": f"example.{model}",
                "parameters": {"max_iter": 1000} if model == "logistic" else {},
            },
            "metrics": [metric, "example.measure"],
        }
    )
    result = run_benchmark(configured, persist=False)
    assert result.aggregate_metrics[metric] == result.aggregate_metrics["example.measure"]


def test_binary_score_metric_and_preflight_task_validation(monkeypatch, installed, config):
    declaration = MetricDeclaration(
        id="example.score",
        tasks=("binary_classification",),
        direction="maximize",
        requires_score=True,
        display_name="Score",
        semantics_version="1.0.0",
    )
    manifest = installed.manifest.model_copy(update={"metrics": (declaration,)})
    plugin = replace(
        installed, manifest=manifest, metrics={"example.score": lambda t, p, s: float(np.mean(s))}
    )
    select(monkeypatch, plugin)
    raw = config.model_dump()
    raw.update(metrics=["example.score"], primary_metric="example.score")
    result = run_benchmark_suite(type(config).model_validate(raw), persist=False)
    assert not result.failures
    assert all(p.score is not None for c in result.candidates for p in c.run_result.predictions)
    raw = load_run_config("configs/examples/diabetes_regression_run.yaml").model_dump()
    raw.update(providers=[{"id": "example", "version": "1.0.0"}], metrics=["example.score"])
    with pytest.raises(ValueError, match="incompatible"):
        run_benchmark(RunConfig.model_validate(raw), persist=False)


def test_metric_callback_cannot_modify_prediction_evidence(installed):
    def mutate(truth, prediction, score):
        truth[:] = 99
        prediction[:] = 99
        return 1.0

    truth = np.array([0, 1])
    prediction = np.array([1, 0])
    compute_metrics(
        (parse_metric("example.agreement"),),
        truth,
        prediction,
        None,
        manifests=(installed.manifest,),
        functions={"example.agreement": mutate},
    )
    assert truth.tolist() == [0, 1]
    assert prediction.tolist() == [1, 0]


def test_provider_runtime_mappings_and_search_definitions_are_isolated(installed, config):
    _, resolved = runtime.activate(config)
    with pytest.raises(TypeError):
        resolved.metrics["another"] = lambda t, p, s: 1.0
    space = resolved.spaces.resolve("example.logistic")
    space.definition["C"]["high"] = 999
    assert resolved.spaces.resolve("example.logistic").definition["C"]["high"] == 2.0
    with pytest.raises(ValueError, match="duplicate"):
        default_model_registry.extended({"dummy_classifier": ModelSpec(lambda p, s: None)})
    with pytest.raises(ValueError, match="duplicate"):
        default_dataset_registry.extended({"iris": lambda: None})


def test_missing_checkpoint_provider_metadata_is_recomputed(installed, config, tmp_path):
    from benchforge.execution.checkpoints import _sha

    workspace = tmp_path / "work"
    first = run_benchmark_suite(config, checkpoint_dir=workspace, persist=False)
    for source in (workspace / "tasks").glob("*.json"):
        data = json.loads(source.read_text())
        data["payload"]["run_result"].pop("provider_manifests")
        data["sha256"] = _sha(data["payload"])
        source.write_text(json.dumps(data))
    second = run_benchmark_suite(config, checkpoint_dir=workspace, resume=True, persist=False)
    assert not second.checkpoint_reused
    assert len(second.checkpoint_invalid) == 2
    assert [c.run_result for c in second.candidates] == [c.run_result for c in first.candidates]


def test_checkpoint_duplicate_keys_rejected(installed, config, tmp_path):
    workspace = tmp_path / "work"
    run_benchmark_suite(config, checkpoint_dir=workspace, persist=False)
    manifest = workspace / "workspace.json"
    manifest.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        run_benchmark_suite(config, checkpoint_dir=workspace, resume=True, persist=False)


def test_bad_dataset_metadata_and_identity_change(monkeypatch, installed, config, tmp_path):
    workspace = tmp_path / "work"
    run_benchmark_suite(config, checkpoint_dir=workspace, persist=False)
    original = installed.datasets["example.toy"]()
    changed = replace(original, identity="example:toy:v2")
    select(monkeypatch, replace(installed, datasets={"example.toy": lambda: changed}))
    with pytest.raises(ValueError, match="incompatible"):
        run_benchmark_suite(config, checkpoint_dir=workspace, resume=True, persist=False)
    bad = replace(original, missing_values={"value": 123, "group": 0})
    select(monkeypatch, replace(installed, datasets={"example.toy": lambda: bad}))
    with pytest.raises(ValueError, match="missing-value metadata"):
        run_benchmark_suite(config, persist=False)


def test_unrelated_broken_entry_point_is_not_loaded(config, monkeypatch, installed):
    selected = Entry(lambda: installed)
    unrelated = Entry(lambda: (_ for _ in ()).throw(AssertionError("unrelated")))
    unrelated.name = "unrelated"
    monkeypatch.setattr(runtime, "_entries", lambda: (unrelated, selected))
    bound, _ = runtime.activate(config)
    assert bound.provider_manifests
    assert unrelated.loaded == 0


def test_changed_search_definition_and_defaults_rejected(config, monkeypatch, installed):
    space = installed.search_spaces["example.logistic"]
    select(
        monkeypatch,
        replace(installed, search_spaces={"example.logistic": replace(space, defaults={"C": 5})}),
    )
    with pytest.raises(ValueError, match="identity disagrees"):
        runtime.resolve(config.providers)


def test_untrusted_manifest_api_and_metadata_consistency(installed, config):
    result = run_benchmark_suite(config)
    metadata_path = result.artifact_directory / "metadata.json"
    raw = json.loads(metadata_path.read_text())
    raw["provider_manifests"][0]["version"] = "9.0.0"
    metadata_path.write_text(json.dumps(raw))
    report = verify(result.artifact_directory)
    assert not report.passed
    assert any("provider metadata disagrees" in d.message for d in report.diagnostics)


@pytest.mark.parametrize("example", ["breast_cancer", "iris_multiclass", "diabetes_regression"])
def test_v011_scientific_evidence_is_preserved_with_unselected_plugins_installed(example):
    """Fixture captured from the isolated v0.11 baseline, before provider changes."""
    import hashlib

    from benchforge.core.config import BenchmarkConfig
    from benchforge.splits.stratified import build_folds

    expected = json.loads(Path("tests/fixtures/v011_scientific_baseline.json").read_text())[example]
    config = BenchmarkConfig.model_validate(expected["config"])
    dataset = default_dataset_registry.resolve(config.dataset)
    folds = build_folds(dataset.target, config.split, config.seed, config.task)
    plan = [
        {"train": f.train_indices.tolist(), "validation": f.validation_indices.tolist()}
        for f in folds
    ]
    assert (
        hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        == expected["folds_sha256"]
    )
    result = run_benchmark_suite(config, persist=False)
    assert result.fingerprint == expected["fingerprint"]
    assert [r.model_identifier for r in result.leaderboard] == expected["ranking"]
    for actual, saved in zip(result.candidates, expected["candidates"], strict=True):
        assert actual.run_result.seed == config.seed
        assert actual.run_result.fingerprint == saved["fingerprint"]
        for metric, aggregate in actual.run_result.aggregate_metrics.items():
            assert aggregate.mean == pytest.approx(saved["metrics"][metric]["mean"], abs=1e-12)
            assert aggregate.std == pytest.approx(saved["metrics"][metric]["std"], abs=1e-12)
        assert [p.sample_index for p in actual.run_result.predictions] == list(
            range(dataset.row_count)
        )
        for prediction in actual.run_result.predictions:
            assert prediction.truth == pytest.approx(dataset.target.iloc[prediction.sample_index])
            assert prediction.prediction == pytest.approx(
                saved["prediction_by_fold"][str(prediction.fold_id)]
            )
            score = saved["score_by_fold"][str(prediction.fold_id)]
            assert (
                prediction.score is None
                if score is None
                else prediction.score == pytest.approx(score)
            )


def test_verification_and_catalog_never_import_provider_modules(installed, config, tmp_path):
    result = run_benchmark_suite(config)
    code = r"""
import importlib.abc, sys
from pathlib import Path
class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("benchforge_example"):
            raise AssertionError("artifact verification tried to import provider code")
sys.meta_path.insert(0, Refuse())
from benchforge.artifacts import verify
from benchforge.catalog import Catalog
report = verify(Path(sys.argv[1]), deep=True)
assert report.passed, report.diagnostics
with Catalog(Path(sys.argv[2])) as catalog:
    assert catalog.ingest(Path(sys.argv[1]))
assert "benchforge_example" not in sys.modules
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(result.artifact_directory),
            str(tmp_path / "offline.sqlite"),
        ],
        check=True,
    )


def test_legacy_builtin_checkpoint_without_extension_fields_resumes(tmp_path):
    from benchforge.core.config import BenchmarkConfig
    from benchforge.execution.checkpoints import _sha

    baseline = json.loads(Path("tests/fixtures/v011_scientific_baseline.json").read_text())
    config = BenchmarkConfig.model_validate(baseline["iris_multiclass"]["config"])
    workspace = tmp_path / "work"
    run_benchmark_suite(config, checkpoint_dir=workspace, persist=False)
    for source in (workspace / "tasks").glob("*.json"):
        raw = json.loads(source.read_text())
        raw["payload"]["run_result"].pop("provider_manifests")
        raw["sha256"] = _sha(raw["payload"])
        source.write_text(json.dumps(raw))
    resumed = run_benchmark_suite(config, checkpoint_dir=workspace, resume=True, persist=False)
    assert resumed.checkpoint_reused == ("dummy_0", "dummy_1")


@pytest.mark.parametrize("workflow", ["run", "search", "robustness", "automl"])
def test_all_provider_cli_workflows(installed, config, workflow, tmp_path):
    if workflow in {"search", "automl"}:
        raw = load_search_config("configs/providers/search.yaml").model_dump(mode="json")
        if workflow == "automl":
            raw.pop("final_search")
            raw["models"] = {"include": ["example.logistic"], "include_fixed_baselines": False}
            raw["budget"] = {"total_trials": 4}
    else:
        raw = config.model_dump(mode="json")
        if workflow == "run":
            raw["model"] = raw.pop("models")[0]
            raw.pop("primary_metric")
        else:
            raw["robustness"] = {"repetitions": 2, "bootstrap_samples": 100}
    raw["output"] = {"directory": str(tmp_path / "published")}
    source = tmp_path / "config.json"
    source.write_text(json.dumps(raw))
    completed = subprocess.run(
        [sys.executable, "-m", "benchforge", workflow, str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Fingerprint" in completed.stdout or "fingerprint" in completed.stdout
    if workflow == "automl":
        planned = subprocess.run(
            [sys.executable, "-m", "benchforge", workflow, str(source), "--plan-only"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert planned.returncode == 0, planned.stderr


def test_numpy_real_metric_scalar_is_supported(installed):
    values = compute_metrics(
        (parse_metric("example.agreement"),),
        np.array([0, 1]),
        np.array([0, 1]),
        None,
        manifests=(installed.manifest,),
        functions={"example.agreement": lambda t, p, s: np.float32(0.5)},
    )
    assert values == {"example.agreement": 0.5}


def test_deferred_dataset_dependency_error_is_actionable(monkeypatch, installed, config):
    def missing():
        raise ImportError("install dataset-extra")

    select(monkeypatch, replace(installed, datasets={"example.toy": missing}))
    with pytest.raises(ValueError, match="example.toy.*install dataset-extra"):
        run_benchmark_suite(config, persist=False)


def test_activation_snapshots_revalidated_immutable_manifest(monkeypatch, installed, config):
    mutable = list(installed.manifest.datasets)
    unsafe_copy = installed.manifest.model_copy(update={"datasets": mutable})
    select(monkeypatch, replace(installed, manifest=unsafe_copy))
    bound, _ = runtime.activate(config)
    mutable.clear()
    assert len(bound.provider_manifests[0].datasets) == 1
    assert isinstance(bound.provider_manifests[0].datasets, tuple)
