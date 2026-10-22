"""Reporting is an immutable presentation of verified scientific evidence."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest

from benchforge.analysis.robustness import interpret_pairwise
from benchforge.artifacts.manifest import digest, seal
from benchforge.artifacts.verify import verify
from benchforge.cli import main
from benchforge.core.config import BenchmarkConfig, RunConfig
from benchforge.reporting import (
    ReportingError,
    VerificationError,
    generate_report,
    load_comparison,
    load_report,
    publication,
    render_comparison,
    render_report,
)
from benchforge.reporting.loading import ROBUSTNESS_LIMITATIONS
from benchforge.reporting.models import Evidence
from benchforge.robustness.config import RobustnessConfig
from benchforge.robustness.runner import RobustnessExecutionError, run_robustness
from tests.test_artifacts_catalog import configuration, edit, generate

KINDS = ("run", "benchmark", "search", "automl", "robustness")
TASKS = ("binary_classification", "multiclass_classification", "regression")


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp("report-evidence")
    return {(k, t): generate(root / k / t, k, t) for k in KINDS for t in TASKS}


def hashes(directory):
    return {
        p.relative_to(directory).as_posix(): digest(p) for p in directory.rglob("*") if p.is_file()
    }


def clone(evidence, tmp_path, kind="run", task=TASKS[0]):
    return Path(shutil.copytree(evidence[kind, task], tmp_path / "copy"))


class OfflineHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.ids = []
        self.references = []
        self.texts = []
        self.values = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        attributes = dict(attrs)
        assert tag not in {"script", "iframe", "object", "embed", "link", "img", "form"}
        assert not any(k.startswith("on") for k in attributes)
        for key in ("src", "href", "action"):
            if key in attributes:
                assert attributes[key].startswith("#"), attributes
                self.references.append(attributes[key][1:])
        if "id" in attributes:
            self.ids.append(attributes["id"])
        if tag == "data":
            self.values.append(attributes["value"])

    def handle_data(self, data):
        self.texts.append(data)


def check_html(content):
    parser = OfflineHTML()
    parser.feed(content)
    assert len(parser.ids) == len(set(parser.ids))
    assert set(parser.references).issubset(parser.ids)
    assert {"html", "head", "title", "style", "main", "table", "caption", "th", "nav"}.issubset(
        parser.tags
    )
    assert "@import" not in content and "url(" not in content
    assert "Content-Security-Policy" in content
    return parser


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("task", TASKS)
def test_real_workflow_values_determinism_and_source_immutability(evidence, tmp_path, kind, task):
    path = evidence[kind, task]
    before = hashes(path)
    report = load_report(path, deep=True)
    assert report.workflow == kind and report.task == task
    assert (
        report.evaluation.decode()["preprocessing"]
        == report.configuration.decode()["preprocessing"]
    )
    assert report.verification.integrity == "sealed" and report.verification.deep
    base = path / "search" if kind == "automl" else path
    if kind == "run":
        board = [
            {
                "model_identifier": report.candidates[0].identifier,
                "metrics": json.loads(
                    (path / "aggregate_metrics.json").read_text(encoding="utf-8")
                ),
            }
        ]
        assert report.primary_metric is None and report.candidates[0].rank is None
    else:
        name = {
            "benchmark": "leaderboard",
            "search": "tuned_leaderboard",
            "automl": "tuned_leaderboard",
            "robustness": "robustness_leaderboard",
        }[kind]
        board = json.loads((base / f"{name}.json").read_text(encoding="utf-8"))
        assert report.primary_metric == (
            "root_mean_squared_error" if task == "regression" else "accuracy"
        )
    html = render_report(report)
    parsed = check_html(html)
    for candidate in report.candidates:
        original = next(r for r in board if r["model_identifier"] == candidate.identifier)
        metrics = original["metrics"]
        metrics = {m["name"]: m for m in metrics} if isinstance(metrics, list) else metrics
        for score in candidate.scores:
            assert score.mean == metrics[score.name]["mean"]
            assert score.std == metrics[score.name]["std"]
            assert str(score.mean) in parsed.values
            fold_values = [
                dict(f.metrics)[score.name]
                for f in report.folds
                if f.candidate == candidate.identifier
            ]
            # Complete repeated folds allow an independent mean here.
            assert score.mean == pytest.approx(float(np.mean(fold_values)), abs=1e-12)
        assert candidate.completed == candidate.expected == 2
        if kind in {"search", "automl"}:
            assert candidate.trials is not None
            assert candidate.trials.completed == original["completed_trials"]
            assert candidate.trials.configured == original["configured_trials"]
    assert all(
        m.direction == ("minimize" if task == "regression" else "maximize") for m in report.metrics
    )
    first = generate_report(path, tmp_path / "first.html", deep=True)
    second = generate_report(path, tmp_path / "second.html", deep=True)
    assert first.read_bytes() == second.read_bytes() == html.encode()
    assert before == hashes(path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.status = "changed"
    private = report.configuration.decode()
    private["seed"] = 999
    assert report.configuration.decode()["seed"] == 42


@pytest.mark.parametrize("kind", ("search", "automl"))
def test_nested_selection_is_distinct_from_unbiased_outer_scores(evidence, kind):
    report = load_report(evidence[kind, TASKS[0]], deep=True)
    for candidate in report.candidates:
        folds = [f for f in report.folds if f.candidate == candidate.identifier]
        assert candidate.scores[0].mean == np.mean([dict(f.metrics)["accuracy"] for f in folds])
        for f in folds:
            assert f.selection is not None
            data = f.selection.decode()
            assert "best_inner_score" in data and "selected_parameters" in data
    html = render_report(report)
    assert "outer evaluation only" in html
    assert "not unbiased performance estimates" in html
    if kind == "search":
        original = json.loads(
            (report.directory / "final_candidate.json").read_text(encoding="utf-8")
        )
        assert report.final_selection.decode()["parameters"] == original["parameters"]
        assert (
            report.final_selection.decode()["cv_selection_score"] == original["cv_selection_score"]
        )
        assert "trials" not in report.final_selection.decode()
    else:
        assert report.plan.decode() == json.loads(
            (report.directory / "plan.json").read_text(encoding="utf-8")
        )


@pytest.mark.parametrize("task", TASKS)
def test_robustness_exact_statistics_and_existing_interpretations(evidence, task):
    path = evidence["robustness", task]
    report = load_report(path, deep=True)
    original = json.loads((path / "pairwise_comparisons.json").read_text(encoding="utf-8"))
    assert json.loads(json.dumps([dataclasses.asdict(p) for p in report.pairs])) == original
    html = render_report(report)
    parser = check_html(html)
    for pair in report.pairs:
        assert interpret_pairwise(pair) in "".join(parser.texts)
        assert pair.mean_delta == np.mean(pair.paired_deltas)
        assert pair.paired_deltas == tuple(
            v if task != "regression" else -v for v in pair.raw_paired_deltas
        )
        assert str(pair.interval.lower_bound) in parser.values
    assert all(w in report.limitations for w in ROBUSTNESS_LIMITATIONS)
    stored_ranks = json.loads((path / "rank_stability.json").read_text(encoding="utf-8"))
    for rank in report.ranks:
        row = next(r for r in stored_ranks if r["model_identifier"] == rank.model_identifier)
        assert rank.first_place_rate == row["first_place_rate"]
        assert {str(f.rank): f.count for f in rank.rank_counts} == row["rank_counts"]


def test_reporting_cannot_fit_load_data_activate_providers_or_execute_optuna(
    evidence, monkeypatch, tmp_path
):
    import optuna
    import sklearn.pipeline

    import benchforge.data.registry as datasets
    import benchforge.execution.runner as kernel
    import benchforge.providers.runtime as providers

    def forbidden(*args, **kwargs):
        raise AssertionError("reporting executed scientific or provider code")

    monkeypatch.setattr(sklearn.pipeline.Pipeline, "fit", forbidden)
    monkeypatch.setattr(optuna, "create_study", forbidden)
    monkeypatch.setattr(optuna.study.Study, "optimize", forbidden)
    monkeypatch.setattr(datasets.DatasetRegistry, "resolve", forbidden)
    monkeypatch.setattr(kernel, "run_benchmark", forbidden)
    monkeypatch.setattr(providers, "activate", forbidden)
    monkeypatch.setattr(providers, "discover", forbidden)
    for kind in KINDS:
        generate_report(evidence[kind, TASKS[0]], tmp_path / f"{kind}.html", deep=True)


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("hash", "hash_mismatch"),
        ("missing", "missing_member"),
        ("manifest", "invalid_manifest_or_structure"),
        ("schema", "unsupported_schema"),
        ("json", "invalid_evidence"),
        ("csv", "invalid_evidence"),
        ("nan", "invalid_evidence"),
        ("duplicate", "invalid_evidence"),
        ("unicode", "invalid_evidence"),
        ("metrics", "invalid_evidence"),
        ("unexpected", "unexpected_member"),
        ("unsafe_manifest", "invalid_manifest_or_structure"),
    ],
)
def test_corrupt_evidence_rejected_before_publication(evidence, tmp_path, mutation, code):
    path = clone(evidence, tmp_path)
    if mutation == "hash":
        (path / "metadata.json").write_text(
            (path / "metadata.json").read_text(encoding="utf-8") + " "
        )
    elif mutation == "missing":
        (path / "predictions.csv").unlink()
    elif mutation == "manifest":
        (path / "manifest.json").write_text("{")
    elif mutation == "schema":
        edit(path / "config.json", lambda d: d.update(schema_version=999))
    elif mutation == "json":
        (path / "aggregate_metrics.json").write_text("{")
    elif mutation == "csv":
        (path / "predictions.csv").write_text("truth,prediction\n1,2,3\n")
    elif mutation == "nan":
        (path / "aggregate_metrics.json").write_text('{"accuracy": {"mean": NaN, "std": 0}}')
    elif mutation == "duplicate":
        (path / "aggregate_metrics.json").write_text('{"accuracy": {}, "accuracy": {}}')
    elif mutation == "unicode":
        (path / "metadata.json").write_bytes(b"\xff")
    elif mutation == "metrics":
        edit(path / "aggregate_metrics.json", lambda d: d["accuracy"].update(mean=0.12345))
    elif mutation == "unexpected":
        (path / "extra.txt").write_text("unprotected")
    else:
        edit(path / "manifest.json", lambda d: d["files"][0].update(path="../../escape"))
    before = hashes(path)
    with pytest.raises(VerificationError, match=code):
        generate_report(path, tmp_path / "out.html", deep=True, allow_legacy=True)
    assert not (tmp_path / "out.html").exists()
    assert hashes(path) == before


@pytest.mark.parametrize("kind", KINDS)
def test_allow_legacy_cannot_excuse_missing_modern_seal(evidence, tmp_path, kind):
    path = clone(evidence, tmp_path, kind)
    (path / "manifest.json").unlink()
    with pytest.raises(VerificationError, match="required manifest.json is missing"):
        load_report(path, allow_legacy=True)


def test_legacy_opt_in_missing_metadata_and_cli(evidence, tmp_path, capsys):
    path = clone(evidence, tmp_path)
    (path / "manifest.json").unlink()
    config = RunConfig.model_validate_json((path / "config.json").read_text(encoding="utf-8"))
    edit(
        path / "metadata.json",
        lambda d: (
            d.update(config_fingerprint=config.fingerprint, benchforge_version="0.7.0"),
            d.pop("fingerprint_method"),
            d.pop("dataset_summary"),
            d.pop("created_at"),
        ),
    )
    with pytest.raises(VerificationError, match="--allow-legacy"):
        load_report(path)
    report = load_report(path, deep=True, allow_legacy=True)
    assert report.dataset is None and report.created_at is None
    assert report.verification.integrity == "legacy/unsealed"
    assert {v[1] for v in report.verification.diagnostics} >= {
        "legacy_unsealed",
        "historical_metadata_unavailable",
    }
    assert "modern byte-integrity guarantees are unavailable" in render_report(report)
    assert (
        main(
            [
                "report",
                str(path),
                "--output",
                str(tmp_path / "legacy.html"),
                "--allow-legacy",
                "--deep",
            ]
        )
        == 0
    )
    assert "Report:" in capsys.readouterr().out


@pytest.mark.parametrize("kind", ("benchmark", "search", "robustness"))
def test_partial_and_failed_candidates_remain_explicit(tmp_path, kind):
    models = configuration()["models"]
    models[0]["parameters"] = {"C": -1}
    path = generate(tmp_path / "source", kind, models=models)
    report = load_report(path, deep=True)
    assert report.status == "partial"
    failed = next(c for c in report.candidates if c.status == "failed")
    assert failed.rank is None and failed.scores == ()
    assert failed.failures > 0
    assert report.failures.decode()
    assert "This experiment is incomplete" in render_report(report)
    assert next(c for c in report.candidates if c.status == "success").scores


def test_all_failed_robustness_and_partial_repetition_folds(tmp_path, monkeypatch):
    import benchforge.robustness.runner as runner

    original = runner.execute_fold
    calls = 0

    def failing(config, dataset, fold, **kwargs):
        nonlocal calls
        calls += 1
        if fold.fold_id == 1:
            raise ValueError("controlled failure after one valid fold")
        return original(config, dataset, fold, **kwargs)

    monkeypatch.setattr(runner, "execute_fold", failing)
    data = configuration(
        output={"directory": str(tmp_path / "source")},
        robustness={"repetitions": 2, "bootstrap_samples": 100},
    )
    with pytest.raises(RobustnessExecutionError) as error:
        run_robustness(RobustnessConfig.model_validate(data))
    report = load_report(error.value.result.artifact_directory, deep=True)
    assert report.status == "failed" and len(report.folds) == 4
    assert all(c.completed == 0 and c.failures == 2 for c in report.candidates)
    assert all(p.usable_repetitions == 0 and p.interval.lower_bound is None for p in report.pairs)
    assert "unavailable" in render_report(report)


def test_exact_ties_retain_stored_policy_and_rank_stability(tmp_path):
    models = [{"name": "dummy_classifier", "id": ident, "parameters": {}} for ident in ("a", "b")]
    report = load_report(generate(tmp_path / "source", "robustness", models=models), deep=True)
    assert report.candidates[0].scores == report.candidates[1].scores
    assert [c.rank for c in report.candidates] == [1, 2]
    assert [r.mean_rank for r in report.ranks] == [1.0, 1.0]
    assert report.pairs[0].ties == 2
    assert "no measured separation" in render_report(report)


@pytest.mark.parametrize("task", TASKS)
def test_compatible_comparison_without_database(evidence, tmp_path, task):
    a = evidence["benchmark", task]
    b = generate(
        tmp_path / "b",
        "benchmark",
        task,
        models=[
            {
                **m,
                "parameters": (
                    {"alpha": 2} if task == "regression" else {"max_iter": 1000, "C": 0.25}
                ),
            }
            if m["id"] == "linear"
            else m
            for m in configuration(task)["models"]
        ],
    )
    report = load_comparison(a, b, deep=True)
    assert report.compatibility == "compatible" and not report.reasons
    for c in report.changes:
        assert c.delta_b_minus_a == c.score_b - c.score_a
        assert c.oriented_delta_b_minus_a == (
            (-1 if task == "regression" else 1) * c.delta_b_minus_a
        )
    assert any(c.parameters_changed for c in report.changes)
    output = generate_report(a, tmp_path / "comparison.html", compare=b, deep=True)
    assert output.read_text(encoding="utf-8") == render_comparison(report)
    check_html(output.read_text(encoding="utf-8"))
    assert not list(tmp_path.rglob("*.sqlite*"))


@pytest.mark.parametrize(
    "change", ("seed", "preprocessing", "candidates", "dataset", "metric", "workflow")
)
def test_incompatible_or_partial_comparisons_withhold_all_deltas(evidence, tmp_path, change):
    updates = {}
    kind = "benchmark"
    if change == "seed":
        updates["seed"] = 43
    elif change == "preprocessing":
        updates["preprocessing"] = {"standardize": False, "median_imputation": True}
    elif change == "candidates":
        updates["models"] = configuration()["models"][:1]
    elif change == "dataset":
        updates["dataset"] = {"name": "iris"}
        updates["task"] = "multiclass_classification"
        updates["models"] = [{"name": "dummy_classifier", "id": "dummy"}]
    elif change == "metric":
        updates["metrics"] = ["balanced_accuracy"]
        updates["primary_metric"] = "balanced_accuracy"
    else:
        kind = "search"
    # task must be supplied positionally to the shared workflow fixture.
    task = updates.pop("task", TASKS[0])
    b = generate(tmp_path / "b", kind, task, **updates)
    report = load_comparison(evidence["benchmark", TASKS[0]], b, deep=True)
    assert report.compatibility == (
        "partially compatible"
        if change in {"seed", "preprocessing", "candidates"}
        else "incompatible"
    )
    assert report.reasons
    assert all(
        c.delta_b_minus_a is None and c.oriented_delta_b_minus_a is None for c in report.changes
    )
    assert "differences are withheld" in render_comparison(report)
    if change == "candidates":
        assert report.only_a == ("dummy",)


def test_robustness_comparison_reuses_existing_statistics(evidence, tmp_path):
    source = evidence["robustness", TASKS[0]]
    report = load_comparison(source, source, deep=True)
    assert report.compatibility == "compatible"
    assert report.a.pairs == report.b.pairs == load_report(source, deep=True).pairs
    generate_report(source, tmp_path / "robustness.html", compare=source, deep=True)


@pytest.mark.parametrize(
    "destination", ("source", "nested", "workspace", "other_seal", "missing_parent", "directory")
)
def test_unsafe_output_destinations(evidence, tmp_path, destination):
    source = evidence["run", TASKS[0]]
    if destination == "source":
        output = source / "report.html"
    elif destination == "nested":
        output = evidence["benchmark", TASKS[0]] / "runs" / "report.html"
    elif destination in {"workspace", "other_seal"}:
        root = tmp_path / "restricted"
        root.mkdir()
        (root / ("workspace.json" if destination == "workspace" else "manifest.json")).write_text(
            "{}"
        )
        output = root / "report.html"
    elif destination == "missing_parent":
        output = tmp_path / "missing" / "report.html"
    else:
        output = tmp_path
    before = hashes(source)
    with pytest.raises(ReportingError):
        generate_report(source, output, overwrite=True)
    assert before == hashes(source)


@pytest.mark.parametrize(
    "where", ("input", "input_ancestor", "output", "output_ancestor", "member")
)
def test_symlinks_fail_closed(evidence, tmp_path, where):
    source = clone(evidence, tmp_path)
    output = tmp_path / "report.html"
    link = tmp_path / "alias"
    try:
        if where == "input":
            link.symlink_to(source, target_is_directory=True)
            source = link
        elif where == "input_ancestor":
            link.symlink_to(tmp_path, target_is_directory=True)
            source = link / "copy"
        elif where == "output":
            output.symlink_to(tmp_path / "not-created")
        elif where == "output_ancestor":
            link.symlink_to(source, target_is_directory=True)
            output = link / "report.html"
        else:
            (source / "bad").symlink_to(source / "config.json")
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(ReportingError, match="symlink"):
        generate_report(source, output, overwrite=True)


def test_output_overwrite_is_explicit_and_atomic(evidence, tmp_path, monkeypatch):
    source = evidence["run", TASKS[0]]
    output = tmp_path / "report.html"
    output.write_text("previous report")
    with pytest.raises(ReportingError, match="already exists"):
        generate_report(source, output)
    assert output.read_text(encoding="utf-8") == "previous report"
    original_replace = publication.os.replace

    def inspect_staged(staged, target):
        assert Path(staged).read_text(encoding="utf-8").endswith("</html>\n")
        assert output.read_text(encoding="utf-8") == "previous report"
        return original_replace(staged, target)

    monkeypatch.setattr(publication.os, "replace", inspect_staged)
    generate_report(source, output, overwrite=True)
    check_html(output.read_text(encoding="utf-8"))
    assert not list(tmp_path.glob(".benchforge-report-*"))


@pytest.mark.parametrize("failure", ("link", "replace", "fsync", "permission", "interrupt", "race"))
def test_failed_publication_keeps_original_and_cleans_staging(
    evidence, tmp_path, monkeypatch, failure
):
    source = evidence["run", TASKS[0]]
    output = tmp_path / "report.html"
    overwrite = failure == "replace"
    if overwrite:
        output.write_text("previous")
    before = hashes(source)

    def fail(*args, **kwargs):
        if failure == "interrupt":
            raise KeyboardInterrupt
        if failure == "race":
            output.write_text("concurrent publisher")
            raise FileExistsError("concurrent output")
        raise PermissionError("injected publication failure")

    if failure == "permission":
        monkeypatch.setattr(publication.tempfile, "NamedTemporaryFile", fail)
    else:
        monkeypatch.setattr(
            publication.os, {"interrupt": "fsync", "race": "link"}.get(failure, failure), fail
        )
    with pytest.raises(KeyboardInterrupt if failure == "interrupt" else ReportingError):
        generate_report(source, output, overwrite=overwrite)
    assert before == hashes(source)
    assert not list(tmp_path.glob(".benchforge-report-*"))
    assert (
        output.read_text(encoding="utf-8") == "previous"
        if overwrite
        else (
            output.read_text(encoding="utf-8") == "concurrent publisher"
            if failure == "race"
            else not output.exists()
        )
    )


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="POSIX permission enforcement")
def test_read_only_output_directory(evidence, tmp_path):
    root = tmp_path / "readonly"
    root.mkdir()
    root.chmod(0o555)
    try:
        with pytest.raises(ReportingError, match="Permission denied"):
            generate_report(evidence["run", TASKS[0]], root / "report.html")
    finally:
        root.chmod(0o755)


def test_malicious_metadata_and_unicode_are_always_text(evidence, tmp_path):
    source = clone(evidence, tmp_path)
    evil = (
        '</pre><script>alert(1)</script><img src=x onerror="boom"> '
        "javascript:alert(2)\ud800\x00\u202e"
    )
    (source / "manifest.json").unlink()
    edit(source / "metadata.json", lambda d: d.update(untrusted=evil))
    seal(source)
    report = load_report(source, deep=True)
    html = render_report(report)
    parser = check_html(html)
    assert "&lt;script&gt;" in html and "alert(1)" in html and "\ufffd" in html
    assert "script" not in parser.tags and "img" not in parser.tags
    # Identifiers and values never become HTML IDs, URLs, attributes or CSS.
    candidate = dataclasses.replace(
        report.candidates[0],
        identifier=evil,
        parameters=Evidence.capture("malicious source", {evil: evil}),
    )
    check_html(render_report(dataclasses.replace(report, candidates=(candidate,))))
    generate_report(source, tmp_path / "evil.html")
    (tmp_path / "evil.html").read_bytes().decode("utf-8", errors="strict")


def test_cli_success_comparison_failures_and_real_subprocess(evidence, tmp_path, capsys):
    source = evidence["benchmark", TASKS[0]]
    output = tmp_path / "cli.html"
    assert (
        main(["report", str(source), "--compare", str(source), "--deep", "--output", str(output)])
        == 0
    )
    assert "Report:" in capsys.readouterr().out
    assert main(["report", str(source), "--output", str(output)]) == 2
    assert "already exists" in capsys.readouterr().err
    assert main(["report", str(tmp_path / "missing"), "--output", str(tmp_path / "bad.html")]) == 2
    assert "verification FAILED" in capsys.readouterr().err
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from benchforge.cli import main; import sys; sys.exit(main())",
            "report",
            str(source),
            "--output",
            str(output),
            "--overwrite",
        ],
        check=True,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    check_html(output.read_text(encoding="utf-8"))


def test_unavailable_provider_and_custom_metric_never_imported(tmp_path):
    from benchforge.core.config import load_benchmark_config
    from benchforge.execution.benchmark import run_benchmark_suite

    config = load_benchmark_config("configs/providers/benchmark.yaml").model_copy(
        update={
            "output": BenchmarkConfig.model_validate(
                configuration(output={"directory": str(tmp_path / "source")})
            ).output
        }
    )
    source = run_benchmark_suite(config).artifact_directory
    script = r"""
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("benchforge_example"):
            raise AssertionError("provider imported during offline reporting")
sys.meta_path.insert(0, Block())
from benchforge.reporting import load_report, generate_report
r = load_report(sys.argv[1], deep=True)
assert r.primary_metric == "example.agreement"
assert any(d[1] == "external_metric_not_recomputed" for d in r.verification.diagnostics)
assert any(m.name == "example.agreement" and m.direction == "maximize" for m in r.metrics)
generate_report(sys.argv[1], sys.argv[2], deep=True)
"""
    output = tmp_path / "provider.html"
    subprocess.run(
        [sys.executable, "-c", script, str(source), str(output)],
        check=True,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    check_html(output.read_text(encoding="utf-8"))


def test_semantically_invalid_rehashed_evidence_is_not_reportable(evidence, tmp_path):
    source = clone(evidence, tmp_path)
    edit(source / "aggregate_metrics.json", lambda d: d["accuracy"].update(mean=0.12345))
    # Refresh only public hashes, bypassing seal()'s semantic guard deliberately.
    edit(
        source / "manifest.json",
        lambda d: [r.update(sha256=digest(source / r["path"])) for r in d["files"]],
    )
    with pytest.raises(VerificationError, match="invalid_evidence"):
        load_report(source)
    assert not any(d.code == "hash_mismatch" for d in verify(source).diagnostics)


def test_source_data_files_are_not_required(tmp_path):
    from benchforge.core.config import FileDatasetConfig, OutputConfig
    from benchforge.data.registry import default_dataset_registry
    from benchforge.execution.runner import run_benchmark

    dataset = default_dataset_registry.resolve("breast_cancer")
    frame = dataset.features.copy()
    frame["target"] = dataset.target
    csv = tmp_path / "original.csv"
    frame.to_csv(csv, index=False)
    from benchforge.core.config import load_run_config

    config = load_run_config("configs/examples/breast_cancer_logreg.yaml").model_copy(
        update={
            "dataset": FileDatasetConfig(
                source="csv", path=csv, target_column="target", task=dataset.task
            ),
            "output": OutputConfig(directory=tmp_path / "source"),
        }
    )
    source = run_benchmark(config).artifact_directory
    csv.unlink()
    generate_report(source, tmp_path / "report.html", deep=True)
    assert not csv.exists()


def test_missing_or_incomplete_artifacts_never_create_output(tmp_path):
    for path in (tmp_path / "missing", tmp_path / "partial"):
        if path.name == "partial":
            path.mkdir()
            (path / "search_config.json").write_text("{}")
        with pytest.raises(VerificationError, match="FAILED"):
            generate_report(path, tmp_path / "out.html")
        assert not (tmp_path / "out.html").exists()


def test_secondary_metrics_keep_individual_directions_and_variability(tmp_path):
    source = generate(
        tmp_path / "source",
        "benchmark",
        "regression",
        metrics=["r2", "root_mean_squared_error"],
        primary_metric="root_mean_squared_error",
    )
    report = load_report(source, deep=True)
    assert {m.name: m.direction for m in report.metrics} == {
        "r2": "maximize",
        "root_mean_squared_error": "minimize",
    }
    assert {m.name for m in report.metrics if m.primary} == {"root_mean_squared_error"}
    rows = json.loads((source / "leaderboard.json").read_text(encoding="utf-8"))
    for c in report.candidates:
        stored = next(r for r in rows if r["model_identifier"] == c.identifier)
        for score in c.scores:
            assert (score.mean, score.std) == (
                stored["metrics"][score.name]["mean"],
                stored["metrics"][score.name]["std"],
            )
    check_html(render_report(report))


@pytest.mark.parametrize("member", ("metadata.json", "aggregate_metrics.json"))
@pytest.mark.parametrize("value", ("1e9999", "-1e9999", "Infinity", "NaN"))
def test_nonfinite_json_is_rejected_even_in_rehashed_metadata(evidence, tmp_path, member, value):
    source = clone(evidence, tmp_path)
    target = source / member
    data = target.read_text(encoding="utf-8").rstrip()
    target.write_text(data[:-1] + ', "nonfinite": ' + value + "}")
    edit(
        source / "manifest.json",
        lambda d: [r.update(sha256=digest(source / r["path"])) for r in d["files"]],
    )
    with pytest.raises(VerificationError, match="non-finite JSON"):
        generate_report(source, tmp_path / "bad.html")
    assert not (tmp_path / "bad.html").exists()


def test_deep_mode_detects_rehashed_robustness_summary_corruption(evidence, tmp_path):
    source = clone(evidence, tmp_path, "robustness")
    edit(source / "pairwise_comparisons.json", lambda pairs: pairs[0].update(mean_delta=123.456))
    edit(
        source / "manifest.json",
        lambda d: [r.update(sha256=digest(source / r["path"])) for r in d["files"]],
    )
    assert verify(source).passed  # Existing standard/deep verification distinction is preserved.
    with pytest.raises(VerificationError, match="pairwise/bootstrap"):
        generate_report(source, tmp_path / "bad.html", deep=True)


def test_partial_comparison_withholds_deltas_and_exposes_failures(evidence, tmp_path):
    models = configuration()["models"]
    models[0]["parameters"] = {"C": -1}
    partial = generate(tmp_path / "source", "benchmark", models=models)
    report = load_comparison(evidence["benchmark", TASKS[0]], partial)
    assert report.compatibility == "partially compatible"
    assert all(c.delta_b_minus_a is None for c in report.changes)
    assert report.b.failures.decode()
    assert "failed/partial evidence" in render_comparison(report)


def test_input_file_instead_of_artifact_and_comparison_corruption(evidence, tmp_path):
    source = clone(evidence, tmp_path)
    (source / "metadata.json").write_text("{}")
    with pytest.raises(VerificationError):
        generate_report(evidence["run", TASKS[0]], tmp_path / "bad.html", compare=source)
    assert not (tmp_path / "bad.html").exists()
    with pytest.raises(VerificationError):
        load_report(source / "metadata.json")


def test_actual_read_only_source_remains_reportable(evidence, tmp_path):
    source = clone(evidence, tmp_path, "benchmark")
    entries = list(source.rglob("*"))
    before = hashes(source)
    for entry in entries:
        entry.chmod(0o555 if entry.is_dir() else 0o444)
    source.chmod(0o555)
    try:
        generate_report(source, tmp_path / "read-only-source.html", deep=True)
        assert hashes(source) == before
    finally:
        source.chmod(0o755)
        for entry in entries:
            entry.chmod(0o755 if entry.is_dir() else 0o644)


def test_rendering_loaded_model_does_not_access_files(evidence, monkeypatch):
    report = load_report(evidence["robustness", TASKS[0]], deep=True)
    expected = render_report(report)

    def forbidden(*args, **kwargs):
        raise AssertionError("renderer accessed filesystem")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    assert render_report(report) == expected
