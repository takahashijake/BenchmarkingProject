import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from benchforge.core.config import PreprocessingConfig
from benchforge.preprocessing import build_preprocessor


def test_preprocessing_statistics_are_learned_from_training_fold_only() -> None:
    # The held-out value is deliberately extreme; fitting before splitting would shift the mean.
    features = pd.DataFrame({"value": [0.0, 1.0, 2.0, 3.0, 10_000.0]})
    target = pd.Series([0, 0, 1, 1, 1])
    pipeline = Pipeline(
        [
            ("preprocessing", build_preprocessor(PreprocessingConfig(), ["value"])),
            ("model", LogisticRegression()),
        ]
    )
    pipeline.fit(features.iloc[:4], target.iloc[:4])
    numeric = pipeline.named_steps["preprocessing"].named_transformers_["numeric"]
    scaler = numeric.named_steps["scaler"]
    assert isinstance(scaler, StandardScaler)
    assert scaler.mean_[0] == np.mean([0.0, 1.0, 2.0, 3.0])
    assert scaler.mean_[0] != features["value"].mean()


def test_preprocessor_builder_returns_fresh_instances() -> None:
    config = PreprocessingConfig()
    assert build_preprocessor(config, ["x"]) is not build_preprocessor(config, ["x"])
