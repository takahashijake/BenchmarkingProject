"""Deterministic, offline, script-free HTML. Evidence is always text, never markup."""

from __future__ import annotations

import html
import json
from collections.abc import Iterable
from dataclasses import asdict
from typing import Any

from benchforge.analysis.robustness import interpret_pairwise
from benchforge.reporting.models import ComparisonReport, Evidence, ExperimentReport

CSS = """
:root { color-scheme: light; font-family: system-ui, sans-serif; color: #18283a;
 background: #f3f5f8; line-height: 1.6; }
* { box-sizing: border-box; }
body { margin: 0; } header, main, footer { max-width: 1240px; margin: auto; padding: 1.5rem; }
header { padding-top: 3rem; } h1 { font-size: clamp(2rem, 5vw, 3.2rem); line-height: 1.15; }
h2 { margin-top: 0; } h3 { margin-bottom: .5rem; } p { max-width: 95ch; }
.eyebrow { letter-spacing: .12em; text-transform: uppercase; color: #465870; font-weight: 700; }
section { background: white; border: 1px solid #d6dfe8; border-radius: .8rem;
 padding: 1.4rem; margin-bottom: 1.5rem; box-shadow: 0 2px 6px #132d4910; }
nav { display: flex; flex-wrap: wrap; gap: 1rem; } a { color: #164d83; }
.notice { border-left: .35rem solid #a65700; background: #fff5e7; padding: 1rem; }
.status { display: inline-block; font-weight: 700; padding: .15rem .6rem;
 border: 1px solid #69809a; border-radius: .3rem; }
dl { display: grid; grid-template-columns: minmax(8rem, 1fr) 4fr; gap: .4rem 1rem; }
dt { font-weight: 700; } dd { margin: 0; overflow-wrap: anywhere; }
.scroll { overflow-x: auto; } table { border-collapse: collapse; width: 100%; min-width: 52rem;
 font-size: .94rem; }
caption { text-align: left; font-weight: 600; margin: .6rem 0; }
th, td { text-align: left; vertical-align: top; padding: .65rem;
 border-bottom: 1px solid #d6dfe8; overflow-wrap: anywhere; min-width: 7rem; }
thead { background: #edf2f8; } tbody th { font-weight: 600; }
.num { font-variant-numeric: tabular-nums; white-space: nowrap; }
pre { white-space: pre-wrap; overflow-wrap: anywhere;
 padding: 1rem; background: #f4f6f9; border: 1px solid #d6dfe8; font-size: .88rem; }
details { margin: .8rem 0; } summary { cursor: pointer; font-weight: 600; }
small { display: block; color: #42536b; } meter { width: 6rem; display: block; }
article + article { border-top: 3px solid #69809a; padding-top: 1rem; }
:focus-visible { outline: 3px solid #ba5e00; outline-offset: 3px; }
@media (max-width: 600px) { header, main, footer { padding: .8rem; }
 section { padding: .8rem; } dl { grid-template-columns: 1fr; } dd { margin-bottom: .6rem; } }
@media print { :root { background: white; } section { box-shadow: none; }
 .scroll { overflow: visible; } nav { display: none; } }
"""


def text(value: Any) -> str:
    """Handle lone surrogates, control characters and bidi overrides as visible replacements."""
    raw = "unavailable" if value is None else str(value)
    clean = "".join(
        c
        if (c in "\n\t" or ord(c) >= 32)
        and not (
            0xD800 <= ord(c) <= 0xDFFF
            or 0x202A <= ord(c) <= 0x202E
            or 0x2066 <= ord(c) <= 0x2069
            or ord(c) == 127
        )
        else "\ufffd"
        for c in raw
    )
    return html.escape(clean, quote=True)


def number(value: float | int | None) -> str:
    if value is None:
        return "unavailable"
    return f'<data class="num" value="{text(value)}">{text(value)}</data>'


def table(caption: str, headings: Iterable[str], rows: Iterable[Iterable[str]]) -> str:
    body = "".join(
        "<tr>"
        + "".join(
            f'<th scope="row">{cell}</th>' if i == 0 else f"<td>{cell}</td>"
            for i, cell in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    return (
        '<div class="scroll" tabindex="0" role="region" aria-label="' + text(caption) + '">'
        "<table><caption>"
        + text(caption)
        + "</caption><thead><tr>"
        + "".join(f'<th scope="col">{text(h)}</th>' for h in headings)
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table></div>"
    )


def details(label: str, evidence: Evidence | None) -> str:
    if evidence is None:
        return f"<p>{text(label)}: unavailable</p>"
    pretty = json.dumps(evidence.decode(), indent=2, sort_keys=True, ensure_ascii=False)
    return (
        f"<details><summary>{text(label)}</summary><small>Source: {text(evidence.source)}</small>"
        f"<pre>{text(pretty)}</pre></details>"
    )


def facts(rows: Iterable[tuple[str, Any]]) -> str:
    return "<dl>" + "".join(f"<dt>{text(k)}</dt><dd>{text(v)}</dd>" for k, v in rows) + "</dl>"


def section(identifier: str, heading: str, body: str) -> str:
    # identifier comes only from fixed renderer strings, never from experiment metadata.
    return f'<section id="{identifier}"><h2>{text(heading)}</h2>{body}</section>'


def _overview(r: ExperimentReport) -> str:
    return (
        f'<p><span class="status">{text(r.status)}</span> '
        f"{text(r.verification.integrity)} · "
        f"{'deep' if r.verification.deep else 'standard'} verification passed</p>"
        + facts(
            (
                ("Workflow", r.workflow),
                ("Task", r.task),
                ("Dataset", r.dataset_name),
                ("Dataset identity", r.dataset_identity),
                ("Fingerprint", r.fingerprint),
                ("Primary metric", r.primary_metric),
                ("Verified artifact location", r.directory),
                ("Artifact schema", r.schema_version),
                ("Evaluator version", r.evaluator_version),
                ("Experiment created", r.created_at),
            )
        )
    )


def _leaderboard(r: ExperimentReport) -> str:
    rows = []
    for c in r.candidates:
        by_name = {s.name: s for s in c.scores}
        scores = []
        for metric in r.metrics:
            score = by_name.get(metric.name)
            scores.append(
                "unavailable"
                if score is None
                else (number(score.mean) + "<small>std: " + number(score.std) + "</small>")
            )
        coverage = f"{text(c.completed)} / {text(c.expected)} {text(c.evaluation_unit)}"
        if c.completed is not None and c.expected is not None and c.expected > 0:
            coverage += (
                f'<meter min="0" max="{text(c.expected)}" value="{text(c.completed)}" '
                'aria-label="Completed evaluations"></meter>'
            )
        trials = (
            "unavailable"
            if c.trials is None
            else (
                f"{c.trials.completed} complete / {c.trials.configured} budget"
                f"<small>{c.trials.failed} failed; {c.trials.pruned} pruned</small>"
            )
        )
        rows.append(
            [
                text(c.identifier) + f"<small>{text(c.model_name)}; {text(c.mode)}</small>",
                number(c.rank),
                text(c.status),
                *scores,
                coverage,
                number(c.failures),
                trials,
            ]
        )
    variability = (
        "Standard deviation is across repetition-level fold means (sample ddof=1; a single "
        "observation has descriptive std=0)."
        if r.workflow == "robustness"
        else "Standard deviation is stored fold-score variability, not an uncertainty interval."
    )
    ranking = (
        "Positions reproduce stored leaderboard order, including its tie policy."
        if r.primary_metric
        else "No primary metric or candidate ranking was declared."
    )
    return (
        f"<p>{text(variability)} {text(ranking)}</p>"
        + table(
            "Recorded candidate results"
            + (" — outer evaluation only" if r.workflow in {"search", "automl"} else ""),
            [
                "Candidate",
                "Stored rank",
                "Status",
                *(
                    f"{m.label} ({m.name}; {m.direction})" + (" — primary" if m.primary else "")
                    for m in r.metrics
                ),
                "Completed / expected",
                "Failures",
                "Outer-study trials",
            ],
            rows,
        )
        + "".join(details(f"Parameters: {c.identifier}", c.parameters) for c in r.candidates)
    )


def _evaluation(r: ExperimentReport) -> str:
    return (
        "<p>Fresh preprocessing and estimators are fitted on training folds only under the "
        "recorded methodology. This report reads persisted evidence and performs no training.</p>"
        + details("Evaluation settings, seed and fold digest", r.evaluation)
        + details("Dataset and target metadata", r.dataset)
        + table(
            "Completed fold evidence (partial folds retained when available)",
            [
                "Candidate",
                "Repetition",
                "Fold",
                "Training rows",
                "Validation rows",
                "Metrics",
                "Estimator seed",
            ],
            (
                [
                    text(f.candidate),
                    number(f.repetition_id),
                    number(f.fold_id),
                    number(f.train_size),
                    number(f.validation_size),
                    "<br>".join(f"{text(k)}: {number(v)}" for k, v in f.metrics),
                    number(f.estimator_seed),
                ]
                for f in r.folds
            ),
        )
    )


def _search(r: ExperimentReport) -> str:
    body = (
        '<p class="notice">Inner-validation and final full-data CV scores are selection-only. '
        "The leaderboard uses untouched outer evaluations of each tuning procedure. Choosing "
        "a family from those results adds model-selection uncertainty.</p>"
        + details("AutoML allocation and family policy", r.plan)
    )
    if r.final_selection:
        value = r.final_selection.decode()
        body += facts(
            (
                ("Final selected family", value.get("model_identifier")),
                ("Mode", value.get("mode")),
                (
                    "Full-data CV selection score (not unbiased evaluation)",
                    value.get("cv_selection_score"),
                ),
            )
        )
    else:
        body += "<p>Final candidate selection: unavailable or disabled in persisted evidence.</p>"
    body += details(
        "Final candidate parameters and trial counts (selection only)", r.final_selection
    )
    body += "<h3>Outer-fold parameter selection</h3>"
    body += "".join(
        details(f"{f.candidate}, outer fold {f.fold_id}: inner selection only", f.selection)
        for f in r.folds
        if f.selection is not None
    )
    return body


def _robustness(r: ExperimentReport) -> str:
    body = (
        '<p class="notice">'
        + " ".join(
            text(v)
            for v in r.limitations
            if ("Intervals describe" in v or "Repetitions reuse" in v or "Ranks compare" in v)
        )
        + "</p>"
    )
    body += table(
        "Paired repetition effects — positive oriented delta means A better than B",
        [
            "A / B",
            "Usable pairs",
            "Mean oriented delta",
            "Robustness interval",
            "A wins / B wins / ties",
            "Interpretation",
        ],
        (
            [
                f"{text(p.candidate_a)} / {text(p.candidate_b)}",
                number(p.usable_repetitions),
                number(p.mean_delta),
                f"{number(p.interval.confidence_level)}: "
                f"[{number(p.interval.lower_bound)}, {number(p.interval.upper_bound)}]",
                f"{p.a_wins} / {p.b_wins} / {p.ties}",
                text(interpret_pairwise(p)),
            ]
            for p in r.pairs
        ),
    )
    body += table(
        "Rank stability — rates include all configured repetitions",
        [
            "Candidate",
            "Mean / median rank",
            "Rank std",
            "First-place rate",
            "Top-two rate",
            "Ranked / missing",
        ],
        (
            [
                text(s.model_identifier),
                f"{number(s.mean_rank)} / {number(s.median_rank)}",
                number(s.rank_std),
                number(s.first_place_rate),
                number(s.top_2_rate),
                f"{s.ranked_repetitions} / {s.missing_repetitions}",
            ]
            for s in r.ranks
        ),
    )
    body += details("Repetition seeds and plan", r.repetition_plan)
    body += "".join(
        details(
            f"Paired evidence: {p.candidate_a} / {p.candidate_b}",
            Evidence.capture("pairwise_comparisons.json", asdict(p)),
        )
        for p in r.pairs
    )
    body += details(
        "Complete rank frequencies",
        Evidence.capture("rank_stability.json", [asdict(s) for s in r.ranks]),
    )
    body += "".join(details(f"Repetition evidence: {i}", e) for i, e in enumerate(r.repetitions))
    return body


def _decision(r: ExperimentReport) -> str:
    assert r.decision is not None
    evidence = r.decision.decode()
    labels = r.dataset.decode().get("target_labels") if r.dataset else None
    body = facts(
        (
            ("Class 0 label", labels[0] if labels else None),
            ("Positive class 1 label", labels[1] if labels else None),
            ("Calibration", evidence["calibration"]["method"]),
            (
                "Calibration CV splits",
                evidence["calibration"]["n_splits"]
                if evidence["calibration"]["method"] != "none"
                else None,
            ),
            ("Threshold policy", evidence["threshold"]["policy"]),
            ("Selection objective", evidence["threshold"]["objective"]),
            ("False-positive cost", evidence["threshold"]["false_positive_cost"]),
            ("False-negative cost", evidence["threshold"]["false_negative_cost"]),
        )
    )
    body += '<p class="notice">Inner selection objectives are selection-only. '
    body += "The following comparisons use matched untouched outer probabilities.</p>"
    body += "<p>Positive decisions use P(encoded class 1) &gt;= threshold. "
    body += "FP means class 0 predicted as 1; FN means class 1 predicted as 0. "
    if evidence["calibration"]["method"] == "none":
        body += "Calibration is disabled; probabilities come directly from predict_proba.</p>"
    else:
        body += "Calibration cross-fits the full preprocessing/model pipeline within training "
        body += "boundaries, with ensemble=False, then refits the base model on that boundary.</p>"
    rows = []
    for candidate in evidence["candidates"]:
        for fold in candidate["folds"]:
            for name in ("selected", "reference"):
                policy = fold[name]
                counts = policy["confusion"]
                metrics = {m["name"]: m["value"] for m in policy["metrics"]}
                rows.append(
                    [
                        text(candidate["identifier"]),
                        number(fold["fold_id"]),
                        text(name),
                        number(policy["threshold"]),
                        *(number(counts[k]) for k in ("tn", "fp", "fn", "tp")),
                        *(
                            number(metrics[k])
                            for k in (
                                "precision",
                                "recall",
                                "f1",
                                "balanced_accuracy",
                                "misclassification_cost",
                            )
                        ),
                    ]
                )
    body += table(
        "Selected policy versus fixed 0.5 reference on each outer fold",
        [
            "Candidate",
            "Outer fold",
            "Policy",
            "Threshold",
            "TN",
            "FP",
            "FN",
            "TP",
            "Precision",
            "Recall",
            "F1",
            "Balanced accuracy",
            "Cost per sample",
        ],
        rows,
    )
    body += "<h3>Held-out probability quality and reliability bins</h3>"
    for candidate in evidence["candidates"]:
        for fold in candidate["folds"]:
            q = fold["probability_quality"]
            body += "<h4>" + text(candidate["identifier"]) + ", outer fold "
            body += number(fold["fold_id"]) + "</h4>"
            body += facts((k, q[k]) for k in ("brier_score", "log_loss", "roc_auc", "ece"))
            body += table(
                "Reliability bins: [lower, upper), final upper inclusive",
                ["Lower", "Upper", "Count", "Mean probability", "Positive fraction"],
                (
                    [
                        number(b[k])
                        for k in (
                            "lower",
                            "upper",
                            "count",
                            "mean_probability",
                            "positive_fraction",
                        )
                    ]
                    for b in q["bins"]
                ),
            )
    body += "<h3>Inner threshold selection (selection evidence only)</h3>"
    body += "".join(details(f"{f.candidate}, outer fold {f.fold_id}", f.selection) for f in r.folds)
    return body


def _experiment(r: ExperimentReport, prefix: str) -> str:
    body = section(prefix + "overview", "Overview", _overview(r))
    body += section(prefix + "leaderboard", "Leaderboard", _leaderboard(r))
    body += section(prefix + "evaluation", "Evaluation", _evaluation(r))
    if r.workflow in {"search", "automl"}:
        body += section(prefix + "search", "Search and AutoML", _search(r))
    if r.workflow == "decision":
        body += section(prefix + "decision", "Calibration and decision analysis", _decision(r))
    if r.workflow == "robustness":
        body += section(prefix + "robustness", "Robustness", _robustness(r))
    body += section(
        prefix + "failures",
        "Failures and incomplete evaluations",
        details("Recorded failures", r.failures),
    )
    diagnostics = (
        table(
            "Verification diagnostics",
            ["Level", "Code", "Message"],
            ([text(v) for v in d] for d in r.verification.diagnostics),
        )
        if r.verification.diagnostics
        else "<p>No verification warnings.</p>"
    )
    body += section(
        prefix + "reproducibility",
        "Reproducibility",
        "<p>Standard verification checks hashes, structure and redundant semantic evidence. "
        "Deep verification additionally recomputes built-in prediction metrics and robustness "
        "statistics. Custom metric implementations are never executed.</p>"
        + diagnostics
        + details("Scientific configuration", r.configuration)
        + details("Dependency and provider provenance", r.provenance),
    )
    body += section(
        prefix + "limitations",
        "Limitations",
        "<ul>" + "".join(f"<li>{text(w)}</li>" for w in r.limitations) + "</ul>",
    )
    return body


def _document(title: str, body: str, nav: str) -> str:
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        "style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'\">"
        f"<title>{text(title)}</title><style>{CSS}</style></head><body>"
        '<header><p class="eyebrow">BenchForge · verified experiment evidence</p>'
        f'<h1>{text(title)}</h1><nav aria-label="Report sections">{nav}</nav></header>'
        f"<main>{body}</main><footer>Standalone offline report · presentation format 1 · "
        "No generated timestamp. Source locations and recorded experiment timestamps are retained."
        "</footer></body></html>\n"
    )


def render_report(report: ExperimentReport) -> str:
    """Render a loaded model without filesystem access or execution."""
    return _document(
        f"{report.workflow.title()} experiment report",
        _experiment(report, ""),
        '<a href="#overview">Overview</a><a href="#leaderboard">Results</a>'
        '<a href="#evaluation">Evaluation</a><a href="#limitations">Limitations</a>',
    )


def render_comparison(report: ComparisonReport) -> str:
    r = report
    body = f'<p class="notice"><strong>{text(r.compatibility.upper())}</strong></p>'
    body += "<ul>" + "".join(f"<li>{text(v)}</li>" for v in r.reasons) + "</ul>"
    body += facts(
        (
            ("Experiment A", r.a.directory),
            ("Experiment B", r.b.directory),
            ("Only in A", ", ".join(r.only_a) or "none"),
            ("Only in B", ", ".join(r.only_b) or "none"),
        )
    )
    body += f"<p>{text(r.interpretation)}</p>"
    if r.compatibility != "compatible":
        body += (
            "<p>Numerical differences are withheld because scientific "
            "compatibility is insufficient.</p>"
        )
    body += table(
        "Descriptive primary scores — differences B minus A",
        [
            "Candidate",
            "A",
            "B",
            "Raw difference",
            "Oriented difference (positive B better)",
            "Parameters changed",
        ],
        (
            [
                text(c.identifier),
                number(c.score_a),
                number(c.score_b),
                number(c.delta_b_minus_a),
                number(c.oriented_delta_b_minus_a),
                text(c.parameters_changed),
            ]
            for c in r.changes
        ),
    )
    for c in r.changes:
        body += details(f"{c.identifier}: parameters A", c.parameters_a)
        body += details(f"{c.identifier}: parameters B", c.parameters_b)
    body += table(
        "Dataset and evaluation differences",
        ["Field", "A", "B"],
        (
            ["Dataset identity", text(r.a.dataset_identity), text(r.b.dataset_identity)],
            ["Task", text(r.a.task), text(r.b.task)],
            ["Primary metric", text(r.a.primary_metric), text(r.b.primary_metric)],
            [
                "Optimization direction",
                text(next((m.direction for m in r.a.metrics if m.primary), None)),
                text(next((m.direction for m in r.b.metrics if m.primary), None)),
            ],
            ["Status", text(r.a.status), text(r.b.status)],
        ),
    )
    body += details("Evaluation A", r.a.evaluation) + details("Evaluation B", r.b.evaluation)
    body = section("comparison", "Comparison", body)
    body += (
        '<article aria-label="Experiment A"><h2>Experiment A</h2>'
        + _experiment(r.a, "a-")
        + "</article>"
    )
    body += (
        '<article aria-label="Experiment B"><h2>Experiment B</h2>'
        + _experiment(r.b, "b-")
        + "</article>"
    )
    return _document(
        "Experiment comparison",
        body,
        '<a href="#comparison">Compatibility</a><a href="#a-overview">Experiment A</a>'
        '<a href="#b-overview">Experiment B</a>',
    )
