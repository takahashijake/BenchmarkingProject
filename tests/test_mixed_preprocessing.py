import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from benchforge.core.config import PreprocessingConfig
from benchforge.preprocessing import build_preprocessor


def test_mixed_missing_values_and_unseen_categories_are_safe() -> None:
    train = pd.DataFrame(
        {
            "number": [1.0, np.nan, 3.0, 5.0],
            "category": ["known", "other", None, "known"],
        }
    )
    validation = pd.DataFrame({"number": [1000.0], "category": ["never-seen"]})
    preprocessor = build_preprocessor(
        PreprocessingConfig(), ["number"], ["category"], dense_output=True
    )
    preprocessor.fit(train)
    transformed = preprocessor.transform(validation)
    assert transformed.shape[0] == 1
    assert np.isfinite(transformed).all()


def test_numeric_and_categorical_learning_is_fold_local() -> None:
    train = pd.DataFrame({"number": [0.0, 1.0, 2.0], "category": ["train-a", "train-a", "train-b"]})
    held_out = pd.DataFrame({"number": [10_000.0], "category": ["validation-only"]})
    preprocessor = build_preprocessor(
        PreprocessingConfig(), ["number"], ["category"], dense_output=True
    )
    preprocessor.fit(train)
    preprocessor.transform(held_out)
    numeric = preprocessor.named_transformers_["numeric"]
    categorical = preprocessor.named_transformers_["categorical"]
    scaler = numeric.named_steps["scaler"]
    encoder = categorical.named_steps["one_hot"]
    assert isinstance(scaler, StandardScaler)
    assert isinstance(encoder, OneHotEncoder)
    assert scaler.mean_[0] == 1.0
    assert "validation-only" not in encoder.categories_[0]
