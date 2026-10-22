"""Scientific and adversarial contracts for complete outer-fold recovery."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from benchforge.artifacts.verify import verify
from benchforge.automl.runner import run_automl
from benchforge.catalog import Catalog
from benchforge.cli import main
from benchforge.core.config import (
    OutputConfig,
    SearchModelConfig,
    load_automl_config,
    load_search_config,
)
from benchforge.data.registry import default_dataset_registry
from benchforge.execution.checkpoint_io import _sha
from benchforge.search import runner
from tests.test_search_execution import _semantic_result


def config_for(tmp_path, example="breast_cancer"):
    original = load_search_config(f"configs/examples/{example}_search.yaml")
    name = "ridge_regressor" if example == "diabetes_regression" else "logistic_regression"
    baseline = "dummy_regressor" if example == "diabetes_regression" else "dummy_classifier"
    return original.model_copy(
        update={
            "models": (
                SearchModelConfig(name=name, id="tuned", mode="search"),
                SearchModelConfig(name=baseline, id="fixed", mode="fixed"),
            ),
            "outer_split": original.outer_split.model_copy(update={"n_splits": 2}),
            "inner_split": original.inner_split.model_copy(update={"n_splits": 2}),
            "budget": original.budget.model_copy(update={"trials_per_outer_fold": 2}),
            "final_search": original.final_search.model_copy(update={"enabled": True, "trials": 2}),
            "output": OutputConfig(directory=tmp_path / "published"),
        }
    )


@pytest.mark.parametrize("example", ["breast_cancer", "iris_multiclass", "diabetes_regression"])
def test_complete_resume_has_no_fits_and_identical_science(tmp_path, monkeypatch, example):
    config = config_for(tmp_path, example)
    reference = runner.run_search(config, persist=False)
    work = tmp_path / "work"
    fresh = runner.run_search(config, checkpoint_dir=work)
    assert len(fresh.checkpoint_executed) == 5
    assert _semantic_result(reference) == _semantic_result(fresh)
    assert verify(fresh.artifact_directory, deep=True).passed

    def forbidden(*args, **kwargs):
        raise AssertionError("a reusable task must not fit")

    monkeypatch.setattr(runner, "execute_fold", forbidden)
    monkeypatch.setattr(runner, "_run_study", forbidden)
    resumed = runner.run_search(config, checkpoint_dir=work, resume=True)
    assert len(resumed.checkpoint_reused) == 5
    assert not resumed.checkpoint_executed
    assert not resumed.checkpoint_invalid
    assert _semantic_result(reference) == _semantic_result(resumed)
    assert resumed.final_candidate.parameters == reference.final_candidate.parameters
    assert verify(resumed.artifact_directory, deep=True).passed
    with Catalog(tmp_path / "catalog.db") as catalog:
        assert len(catalog.ingest(resumed.artifact_directory)) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "truncate",
        "checksum",
        "schema",
        "contract",
        "indices",
        "duplicate",
        "truth",
        "metrics",
        "count",
        "selection",
        "trial_parameters",
        "inner",
        "sampler",
        "candidate",
    ],
)
def test_invalid_task_recomputed_even_with_rehashed_payload(tmp_path, mutation):
    config = config_for(tmp_path)
    work = tmp_path / "work"
    fresh = runner.run_search(config, checkpoint_dir=work, persist=False)
    path = next(
        p
        for p in (work / "tasks").glob("*.json")
        if "outer_result" in json.loads(p.read_text())["payload"]
        and json.loads(p.read_text())["payload"]["trials"]
    )
    data = json.loads(path.read_text())
    payload = data["payload"]
    if mutation == "truncate":
        path.write_text("{")
    else:
        if mutation == "checksum":
            data["sha256"] = "0" * 64
        elif mutation == "schema":
            data["schema"] = 999
        elif mutation == "contract":
            data["contract"] = "other"
        elif mutation == "indices":
            payload["outer_predictions"][0]["sample_index"] = -1
        elif mutation == "duplicate":
            payload["outer_predictions"][1] = payload["outer_predictions"][0]
        elif mutation == "truth":
            payload["outer_predictions"][0]["truth"] = 9
        elif mutation == "metrics":
            payload["outer_result"]["metrics"]["roc_auc"] = 0
        elif mutation == "count":
            payload["completed_trials"] += 1
        elif mutation == "selection":
            payload["best_inner_score"] = -3
        elif mutation == "trial_parameters":
            payload["trials"][0]["parameters"]["C"] = -1
        elif mutation == "inner":
            payload["inner_plan"][0]["train"] = []
        elif mutation == "sampler":
            payload["sampler_seed"] += 1
        elif mutation == "candidate":
            payload["selected_parameters"]["C"] = -1
        if mutation != "checksum":
            data["sha256"] = _sha(payload)
        path.write_text(json.dumps(data))
    resumed = runner.run_search(config, checkpoint_dir=work, resume=True, persist=False)
    assert len(resumed.checkpoint_invalid) == 1
    assert len(resumed.checkpoint_executed) == 1
    assert len(resumed.checkpoint_reused) == 4
    assert _semantic_result(fresh) == _semantic_result(resumed)


@pytest.mark.parametrize("mutation", ["seed", "model", "metrics", "folds", "data", "space"])
def test_scientific_identity_drift_rejected_before_fit(tmp_path, monkeypatch, mutation):
    config = config_for(tmp_path)
    work = tmp_path / "work"
    runner.run_search(config, checkpoint_dir=work, persist=False)
    dataset = default_dataset_registry.resolve(config.dataset)
    if mutation == "seed":
        config = config.model_copy(update={"seed": config.seed + 1})
    elif mutation == "model":
        model = config.models[0].model_copy(update={"parameters": {"C": 3}})
        config = config.model_copy(update={"models": (model, *config.models[1:])})
    elif mutation == "metrics":
        config = config.model_copy(update={"metrics": config.metrics[:1]})
    elif mutation == "folds":
        original = runner.build_folds

        def changed(*args):
            folds = original(*args)
            return tuple(replace(f, train_indices=f.train_indices[::-1]) for f in folds)

        monkeypatch.setattr(runner, "build_folds", changed)
    elif mutation == "space":
        import benchforge.search.recovery as recovery

        monkeypatch.setattr(recovery, "CONTRACT", "changed-contract")
    else:
        dataset.features.iloc[0, 0] += 1

    def forbidden(*args, **kwargs):
        raise AssertionError("incompatibility must be detected before fitting")

    monkeypatch.setattr(runner, "_run_study", forbidden)
    with pytest.raises(ValueError, match="incompatible"):
        runner.run_search(config, dataset=dataset, checkpoint_dir=work, resume=True, persist=False)


@pytest.mark.parametrize("member", ["root", "task", "temporary", "lock", "ancestor"])
def test_unexpected_members_and_symlinks_rejected(tmp_path, member):
    config = config_for(tmp_path)
    work = tmp_path / "work"
    runner.run_search(config, checkpoint_dir=work, persist=False)
    if member == "root":
        (work / "surprise").write_text("x")
    elif member == "task":
        (work / "tasks" / "surprise.json").write_text("{}")
    elif member == "temporary":
        (work / "tasks" / ".partial.tmp").symlink_to(work / "workspace.json")
    elif member == "lock":
        lock = Path(str(work) + ".lock")
        lock.unlink()
        lock.symlink_to(work / "workspace.json")
    else:
        (tmp_path / "alias").symlink_to(tmp_path, target_is_directory=True)
        work = tmp_path / "alias" / "work"
    with pytest.raises(ValueError, match="unexpected|unsafe|symlink"):
        runner.run_search(config, checkpoint_dir=work, resume=True, persist=False)


def test_lifecycle_and_concurrent_process_lock(tmp_path):
    config = config_for(tmp_path)
    work = tmp_path / "work"
    with pytest.raises(ValueError, match="does not exist"):
        runner.run_search(config, checkpoint_dir=work, resume=True, persist=False)
    runner.run_search(config, checkpoint_dir=work, persist=False)
    with pytest.raises(ValueError, match="already exists"):
        runner.run_search(config, checkpoint_dir=work, persist=False)
    script = """
from filelock import FileLock
import sys
with FileLock(sys.argv[1], timeout=0):
    print('locked', flush=True)
    sys.stdin.readline()
"""
    with subprocess.Popen(
        [sys.executable, "-c", script, str(work) + ".lock"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as child:
        assert child.stdout.readline().strip() == "locked"
        try:
            with pytest.raises(ValueError, match="already in use"):
                runner.run_search(config, checkpoint_dir=work, resume=True, persist=False)
        finally:
            child.communicate("release\n", timeout=10)


@pytest.mark.parametrize("stage", ["mid-study", "publication", "sealing"])
@pytest.mark.parametrize("automl", [False, True])
def test_real_sigkill_resume_matches_uninterrupted(tmp_path, stage, automl, monkeypatch):
    """Pipe handshake blocks at a precise boundary; parent kills the real process."""
    config = automl_config(tmp_path) if automl else config_for(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(config.model_dump_json())
    work = tmp_path / "work"
    script = """
import sys
from pathlib import Path
from benchforge.core.config import SearchConfig, AutoMLConfig
from benchforge.search import runner
from benchforge.automl.runner import run_automl
from benchforge.search.recovery import SearchWorkspace
from benchforge.artifacts import manifest
stage, automl = sys.argv[3], sys.argv[4] == 'True'
cls = AutoMLConfig if automl else SearchConfig
config = cls.model_validate_json(Path(sys.argv[1]).read_text())
def pause():
    print('boundary', flush=True)
    sys.stdin.readline()
if stage == 'publication':
    import benchforge.execution.checkpoint_io as io
    original = io.os.replace
    calls = 0
    def replace(src, dst):
        global calls
        if Path(dst).parent.name == 'tasks':
            calls += 1
            if calls == 2:
                pause()
        return original(src, dst)
    io.os.replace = replace
elif stage == 'mid-study':
    original = runner.run_benchmark
    def benchmark(*args, **kwargs):
        if len(list((Path(sys.argv[2]) / 'tasks').glob('*.json'))) == 1:
            pause()
        return original(*args, **kwargs)
    runner.run_benchmark = benchmark
else:
    original = manifest.seal
    def seal(*args, **kwargs):
        pause()
        return original(*args, **kwargs)
    manifest.seal = seal
    import benchforge.storage.search as storage
    storage.seal = seal
(run_automl if automl else runner.run_search)(config, checkpoint_dir=Path(sys.argv[2]))
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(config_path), str(work), stage, str(automl)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # select provides a bounded handshake without timing-based synchronization.
    import select

    try:
        assert select.select([child.stdout], [], [], 45)[0], "child never reached boundary"
        assert child.stdout.readline().strip() == "boundary"
        child.kill()
        child.communicate(timeout=10)
        assert child.returncode == -9
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=10)
    completed = len(list((work / "tasks").glob("*.json")))
    assert completed >= 1
    run = run_automl if automl else runner.run_search
    reference = run(config, persist=False)
    resumed = run(config, checkpoint_dir=work, resume=True)
    if automl:
        assert resumed.fingerprint == reference.fingerprint
        assert resumed.plan.canonical_dict() == reference.plan.canonical_dict()
        assert verify(resumed.artifact_directory, deep=True).passed
        resumed, reference = resumed.search_result, reference.search_result
    assert len(resumed.checkpoint_reused) == completed
    assert _semantic_result(resumed) == _semantic_result(reference)
    assert verify(resumed.artifact_directory, deep=True).passed


def automl_config(tmp_path):
    config = load_automl_config("configs/examples/breast_cancer_automl.yaml")
    return config.model_copy(
        update={
            "models": config.models.model_copy(
                update={"include": ("logistic_regression", "dummy_classifier")}
            ),
            "outer_split": config.outer_split.model_copy(update={"n_splits": 2}),
            "inner_split": config.inner_split.model_copy(update={"n_splits": 2}),
            "budget": config.budget.model_copy(
                update={"total_trials": 6, "final_search_trials": 2}
            ),
            "output": OutputConfig(directory=tmp_path / "published"),
        }
    )


def test_automl_plan_change_rejected_and_complete_reuse(tmp_path, monkeypatch):
    config = automl_config(tmp_path)
    work = tmp_path / "work"
    first = run_automl(config, checkpoint_dir=work)

    def forbidden(*args, **kwargs):
        raise AssertionError("completed AutoML must avoid fits")

    monkeypatch.setattr(runner, "execute_fold", forbidden)
    monkeypatch.setattr(runner, "_run_study", forbidden)
    resumed = run_automl(config, checkpoint_dir=work, resume=True)
    assert len(resumed.search_result.checkpoint_reused) == 5
    assert first.fingerprint == resumed.fingerprint
    assert verify(resumed.artifact_directory, deep=True).passed
    with Catalog(tmp_path / "catalog.db") as catalog:
        assert len(catalog.ingest(resumed.artifact_directory)) == 1
    changed = config.model_copy(
        update={"budget": config.budget.model_copy(update={"total_trials": 7})}
    )
    with pytest.raises(ValueError, match="incompatible"):
        run_automl(changed, checkpoint_dir=work, resume=True, persist=False)


def test_timeout_custom_registry_and_resume_without_path_rejected(tmp_path):
    config = config_for(tmp_path)
    config = config.model_copy(
        update={"budget": config.budget.model_copy(update={"timeout_seconds_per_outer_fold": 10})}
    )
    with pytest.raises(ValueError, match="wall-clock"):
        runner.run_search(config, checkpoint_dir=tmp_path / "work", persist=False)
    with pytest.raises(ValueError, match="requires checkpoint_dir"):
        runner.run_search(config, resume=True, persist=False)


def test_cli_reports_reuse(tmp_path, capsys):
    config = config_for(tmp_path)
    path = tmp_path / "search.yaml"
    path.write_text(config.model_dump_json())
    work = tmp_path / "work"
    assert main(["search", str(path), "--workspace", str(work)]) == 0
    assert "Executed tasks: 5" in capsys.readouterr().out
    assert main(["search", str(path), "--resume", str(work)]) == 0
    assert "Reused tasks: 5" in capsys.readouterr().out


def test_provider_recovery_and_manifest_drift(tmp_path, monkeypatch):
    from benchforge.providers import runtime

    config = load_search_config("configs/providers/search.yaml").model_copy(
        update={"output": OutputConfig(directory=tmp_path / "published")}
    )
    work = tmp_path / "providers"
    first = runner.run_search(config, checkpoint_dir=work)
    assert not first.failures

    def forbidden(*args, **kwargs):
        raise AssertionError("valid provider checkpoint must not fit")

    monkeypatch.setattr(runner, "_run_study", forbidden)
    second = runner.run_search(config, checkpoint_dir=work, resume=True)
    assert len(second.checkpoint_reused) == 2
    assert verify(second.artifact_directory, deep=True).passed
    original = runtime._load

    def changed(*args):
        provider, provenance = original(*args)
        manifest = provider.manifest
        component = manifest.metrics[0].model_copy(update={"display_name": "changed metric"})
        manifest = manifest.model_copy(update={"metrics": (component, *manifest.metrics[1:])})
        return replace(provider, manifest=manifest), provenance

    monkeypatch.setattr(runtime, "_load", changed)
    with pytest.raises(ValueError, match="incompatible"):
        runner.run_search(config, checkpoint_dir=work, resume=True, persist=False)


def test_unversioned_registries_rejected(tmp_path):
    from benchforge.models.registry import ModelRegistry
    from benchforge.search.registry import SearchSpaceRegistry

    config = config_for(tmp_path)
    with pytest.raises(ValueError, match="version-pinned"):
        runner.run_search(
            config,
            model_registry=ModelRegistry({}),
            checkpoint_dir=tmp_path / "models",
            persist=False,
        )
    with pytest.raises(ValueError, match="version-pinned"):
        runner.run_search(
            config,
            search_space_registry=SearchSpaceRegistry({}),
            checkpoint_dir=tmp_path / "spaces",
            persist=False,
        )


def test_mixed_family_failure_recovery_preserves_ranking(tmp_path):
    config = config_for(tmp_path)
    bad = SearchModelConfig(
        name="logistic_regression", id="bad", mode="fixed", parameters={"C": -1}
    )
    config = config.model_copy(update={"models": (*config.models, bad)})
    work = tmp_path / "work"
    reference = runner.run_search(config, persist=False)
    fresh = runner.run_search(config, checkpoint_dir=work, persist=False)
    resumed = runner.run_search(config, checkpoint_dir=work, resume=True, persist=False)
    assert len(resumed.failures) == 1
    assert len(resumed.checkpoint_reused) == 5
    assert resumed.failures[0].model_identifier == "bad"
    assert _semantic_result(resumed) == _semantic_result(reference) == _semantic_result(fresh)


def test_all_failed_search_remains_explicit(tmp_path):
    config = config_for(tmp_path)
    bad = SearchModelConfig(
        name="logistic_regression", id="bad", mode="fixed", parameters={"C": -1}
    )
    config = config.model_copy(update={"models": (bad,)})
    work = tmp_path / "work"
    for resume in (False, True):
        with pytest.raises(runner.FamilySearchError, match="every search family failed"):
            runner.run_search(config, checkpoint_dir=work, resume=resume, persist=False)
    assert not list((work / "tasks").glob("*.json"))


def test_output_path_changes_do_not_invalidate_recovery(tmp_path):
    config = config_for(tmp_path)
    work = tmp_path / "work"
    runner.run_search(config, checkpoint_dir=work, persist=False)
    changed = config.model_copy(update={"output": OutputConfig(directory=tmp_path / "elsewhere")})
    result = runner.run_search(changed, checkpoint_dir=work, resume=True)
    assert len(result.checkpoint_reused) == 5
    assert result.artifact_directory.parent == tmp_path / "elsewhere"


def test_invalid_final_task_is_recomputed(tmp_path):
    config = config_for(tmp_path)
    work = tmp_path / "work"
    first = runner.run_search(config, checkpoint_dir=work, persist=False)
    path = next(
        p
        for p in (work / "tasks").glob("*.json")
        if "result" in json.loads(p.read_text())["payload"]
    )
    data = json.loads(path.read_text())
    data["payload"]["result"]["cv_selection_score"] = -5
    data["sha256"] = _sha(data["payload"])
    path.write_text(json.dumps(data))
    second = runner.run_search(config, checkpoint_dir=work, resume=True, persist=False)
    assert len(second.checkpoint_invalid) == 1
    assert len(second.checkpoint_executed) == 1
    assert len(second.checkpoint_reused) == 4
    assert second.final_candidate.parameters == first.final_candidate.parameters


def test_checkpoint_publication_failure_is_not_a_family_failure(tmp_path, monkeypatch):
    from benchforge.search import recovery

    config = config_for(tmp_path)

    def broken(*args, **kwargs):
        raise ValueError("serialization error")

    monkeypatch.setattr(recovery.SearchWorkspace, "_save", broken)
    with pytest.raises(recovery.CheckpointPublicationError):
        runner.run_search(config, checkpoint_dir=tmp_path / "work", persist=False)


def test_multiple_search_families_and_fixed_final_recovery(tmp_path, monkeypatch):
    config = config_for(tmp_path)
    tree = SearchModelConfig(
        name="random_forest_classifier",
        id="forest",
        mode="search",
        parameters={"n_estimators": 5, "n_jobs": 1},
    )
    config = config.model_copy(update={"models": (*config.models, tree)})
    work = tmp_path / "work"
    reference = runner.run_search(config, persist=False)
    first = runner.run_search(config, checkpoint_dir=work, persist=False)
    second = runner.run_search(config, checkpoint_dir=work, resume=True, persist=False)
    assert len(second.checkpoint_reused) == 7
    assert _semantic_result(reference) == _semantic_result(first) == _semantic_result(second)
    # A fixed-only search also has a final full-data CV selection stage.
    fixed = config.model_copy(update={"models": (config.models[1],)})
    fixed_work = tmp_path / "fixed"
    runner.run_search(fixed, checkpoint_dir=fixed_work, persist=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("completed fixed final selection must not refit")

    monkeypatch.setattr(runner, "run_benchmark", forbidden)
    monkeypatch.setattr(runner, "execute_fold", forbidden)
    result = runner.run_search(fixed, checkpoint_dir=fixed_work, resume=True, persist=False)
    assert len(result.checkpoint_reused) == 3


def test_failed_trials_are_preserved_and_reused(tmp_path, monkeypatch):
    from benchforge.execution.runner import FoldExecutionError

    config = config_for(tmp_path)
    original = runner.run_benchmark
    calls = 0

    def intermittent(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls % 2:
            try:
                raise ValueError("deterministic numeric trial failure")
            except ValueError as exc:
                raise FoldExecutionError("failed trial") from exc
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "run_benchmark", intermittent)
    work = tmp_path / "work"
    first = runner.run_search(config, checkpoint_dir=work, persist=False)
    assert first.families[0].failed_trials == 2
    before = calls
    second = runner.run_search(config, checkpoint_dir=work, resume=True, persist=False)
    assert calls == before
    assert len(second.checkpoint_reused) == 5
    assert not second.checkpoint_invalid
    assert _semantic_result(first) == _semantic_result(second)


@pytest.mark.parametrize("field", ["workspace_schema", "execution_contract", "experiment_identity"])
def test_manifest_version_or_identity_mismatch_is_fatal(tmp_path, field):
    config = config_for(tmp_path)
    work = tmp_path / "work"
    runner.run_search(config, checkpoint_dir=work, persist=False)
    path = work / "workspace.json"
    manifest = json.loads(path.read_text())
    manifest[field] = "unknown"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incompatible"):
        runner.run_search(config, checkpoint_dir=work, resume=True, persist=False)


def test_automl_cli_resume_and_plan_only_restriction(tmp_path, capsys):
    config = automl_config(tmp_path)
    path = tmp_path / "automl.yaml"
    path.write_text(config.model_dump_json())
    work = tmp_path / "work"
    assert main(["automl", str(path), "--workspace", str(work)]) == 0
    capsys.readouterr()
    assert main(["automl", str(path), "--resume", str(work)]) == 0
    assert "Reused tasks: 5" in capsys.readouterr().out
    assert main(["automl", str(path), "--resume", str(work), "--plan-only"]) == 2


def test_successful_fold_of_failed_family_survives_and_leaderboard_is_rebuilt(
    tmp_path, monkeypatch
):
    from benchforge.execution.runner import FoldExecutionError

    config = config_for(tmp_path)
    work = tmp_path / "work"
    reference = runner.run_search(config, persist=False)
    execute = runner.execute_fold

    def fail_second(config, dataset, fold, **kwargs):
        if config.model.id == "tuned" and fold.fold_id == 1:
            raise FoldExecutionError("interrupted family evaluation")
        return execute(config, dataset, fold, **kwargs)

    monkeypatch.setattr(runner, "execute_fold", fail_second)
    partial = runner.run_search(config, checkpoint_dir=work, persist=False)
    assert [f.model_identifier for f in partial.failures] == ["tuned"]
    assert "outer:tuned:0" in partial.checkpoint_executed
    executed = []

    def counted(config, dataset, fold, **kwargs):
        executed.append((config.model.id, fold.fold_id))
        return execute(config, dataset, fold, **kwargs)

    monkeypatch.setattr(runner, "execute_fold", counted)
    resumed = runner.run_search(config, checkpoint_dir=work, resume=True, persist=False)
    assert executed == [("tuned", 1)]
    assert resumed.checkpoint_reused == ("outer:tuned:0", "outer:fixed:0", "outer:fixed:1")
    assert _semantic_result(resumed) == _semantic_result(reference)


@pytest.mark.parametrize(
    "mutation", ["space_version", "space_defaults", "final_folds", "target_labels"]
)
def test_resolved_identity_inputs_are_bound(tmp_path, monkeypatch, mutation):
    from benchforge.search import recovery
    from benchforge.search.registry import default_search_space_registry

    config = config_for(tmp_path)
    work = tmp_path / "work"
    runner.run_search(config, checkpoint_dir=work, persist=False)
    dataset = default_dataset_registry.resolve(config.dataset)
    if mutation in {"space_version", "space_defaults"}:
        original = default_search_space_registry.resolve

        def resolve(name):
            space = original(name)
            if mutation == "space_version":
                return replace(space, version=space.version + 1)
            return replace(space, defaults={**space.defaults, "max_iter": 123})

        monkeypatch.setattr(default_search_space_registry, "resolve", resolve)
    elif mutation == "final_folds":
        original = recovery.build_folds

        def folds(*args):
            return tuple(replace(f, train_indices=f.train_indices[::-1]) for f in original(*args))

        monkeypatch.setattr(recovery, "build_folds", folds)
    else:
        dataset = replace(dataset, target_labels=("changed", "labels"))
    with pytest.raises(ValueError, match="incompatible"):
        runner.run_search(config, dataset=dataset, checkpoint_dir=work, resume=True, persist=False)


@pytest.mark.parametrize("mutation", ["duplicate_json", "null", "unexpected_field", "nonfinite"])
def test_malformed_structured_payload_recomputed(tmp_path, mutation):
    config = config_for(tmp_path)
    work = tmp_path / "work"
    runner.run_search(config, checkpoint_dir=work, persist=False)
    path = next((work / "tasks").glob("*.json"))
    data = json.loads(path.read_text())
    if mutation == "duplicate_json":
        path.write_text('{"schema":1,' + path.read_text()[1:])
    elif mutation == "nonfinite":
        data["payload"]["duration_seconds"] = float("nan")
        path.write_text(json.dumps(data))
    else:
        if mutation == "null":
            data["payload"] = None
        else:
            data["payload"]["unrecognized"] = True
        data["sha256"] = _sha(data["payload"])
        path.write_text(json.dumps(data))
    result = runner.run_search(config, checkpoint_dir=work, resume=True, persist=False)
    assert len(result.checkpoint_invalid) == 1
    assert len(result.checkpoint_executed) == 1


def test_boolean_schema_is_not_integer_version_one(tmp_path):
    config = config_for(tmp_path)
    work = tmp_path / "work"
    runner.run_search(config, checkpoint_dir=work, persist=False)
    path = next((work / "tasks").glob("*.json"))
    data = json.loads(path.read_text())
    data["schema"] = True
    path.write_text(json.dumps(data))
    result = runner.run_search(config, checkpoint_dir=work, resume=True, persist=False)
    assert len(result.checkpoint_invalid) == 1
    manifest_path = work / "workspace.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["workspace_schema"] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incompatible"):
        runner.run_search(config, checkpoint_dir=work, resume=True, persist=False)
