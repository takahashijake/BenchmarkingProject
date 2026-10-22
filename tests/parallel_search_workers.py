"""Importable real-process fixtures for nested-search supervision contracts."""

from __future__ import annotations

import os
from dataclasses import replace
from multiprocessing.connection import Client

from benchforge.search.parallel import TaskOutcome, TaskRequest, _outer_process_entry


def crash_first(request: TaskRequest) -> TaskOutcome:
    if request.task.candidate_index == 0 and request.task.outer.fold_id == 0:
        os._exit(73)
    return _outer_process_entry(request)


def corrupt_result(request: TaskRequest) -> TaskOutcome:
    outcome = _outer_process_entry(request)
    assert outcome.result is not None
    return replace(outcome, result=replace(outcome.result, sampler_seed=123))


def bad_ipc(request: TaskRequest) -> str:
    return "invalid protocol"


def unexpected(request: TaskRequest) -> TaskOutcome:
    from benchforge.search import runner

    def broken(*args: object, **kwargs: object) -> object:
        raise LookupError("unexpected kernel defect")

    runner._search_outer = broken
    return _outer_process_entry(request)


def controlled(request: TaskRequest) -> TaskOutcome:
    """Test parent authorizes return, death, or pause inside study/refit over a socket."""
    from benchforge.search import runner

    host, port = os.environ["BENCHFORGE_TEST_ADDRESS"].rsplit(":", 1)
    with Client((host, int(port)), authkey=b"benchforge-test") as connection:
        connection.send((request.task.task_id, os.getpid()))
        command = connection.recv()
        if command in {"study", "refit"}:
            original = runner.run_benchmark if command == "study" else runner.execute_fold

            def blocked(*args: object, **kwargs: object) -> object:
                connection.send(("boundary", os.getpid()))
                action = connection.recv()
                if action == "crash":
                    os._exit(74)
                return original(*args, **kwargs)

            if command == "study":
                runner.run_benchmark = blocked
            else:
                runner.execute_fold = blocked
        elif command == "crash":
            os._exit(73)
        outcome = _outer_process_entry(request)
        connection.send(("complete", os.getpid()))
        connection.recv()
        return outcome


def audited(request: TaskRequest) -> TaskOutcome:
    """Assert actual worker studies, preprocessing fits and private-state ownership."""
    import numpy as np
    from sklearn.compose import ColumnTransformer

    from benchforge.search import runner

    original_study = runner._run_study
    original_execute = runner.execute_fold
    original_fit = ColumnTransformer.fit_transform
    expected = None
    outer_train = set(request.task.outer.train)
    held_out = set(request.task.outer.validation)
    assert not request.task.outer.to_fold().train_indices.flags.writeable
    assert outer_train.isdisjoint(held_out)

    def study(config, candidate, dataset, folds, **kwargs):
        assert all(set(f.train_indices) | set(f.validation_indices) == outer_train for f in folds)
        assert all(set(f.train_indices).isdisjoint(held_out) for f in folds)
        return original_study(config, candidate, dataset, folds, **kwargs)

    def execute(config, dataset, fold, **kwargs):
        nonlocal expected
        expected = dataset.features.iloc[fold.train_indices].copy(deep=True)
        return original_execute(config, dataset, fold, **kwargs)

    def fit(self, features, target=None, **kwargs):
        assert expected is not None
        assert features.equals(expected)
        result = original_fit(self, features, target, **kwargs)
        if "numeric" in self.named_transformers_:
            numeric = self.named_transformers_["numeric"]
            if "scaler" in numeric.named_steps:
                transformed = numeric.named_steps["imputer"].transform(features)
                np.testing.assert_allclose(
                    numeric.named_steps["scaler"].mean_, np.mean(transformed, axis=0)
                )
        return result

    runner._run_study = study
    runner.execute_fold = execute
    # Inner studies use execution.runner.execute_fold too.
    import benchforge.execution.runner as kernel

    kernel.execute_fold = execute
    ColumnTransformer.fit_transform = fit
    outcome = _outer_process_entry(request)
    # Mutation after evaluation cannot escape this interpreter into another task/coordinator.
    request.dataset.features.iloc[:, :] = -999
    assert outcome.result is not None, outcome.message
    return outcome


def refit_failure(request: TaskRequest) -> TaskOutcome:
    from benchforge.execution.runner import FoldExecutionError
    from benchforge.search import runner

    if request.task.candidate_index == 0:

        def failed(*args, **kwargs):
            raise FoldExecutionError("outer refit refused")

        runner.execute_fold = failed
    return _outer_process_entry(request)
