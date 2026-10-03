from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from benchforge.core.config import PreprocessingConfig


def build_preprocessor(
    config: PreprocessingConfig,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str] = (),
    *,
    dense_output: bool = False,
) -> ColumnTransformer:
    transformers: list[tuple[str, object, list[str]]] = []
    if numeric_features:
        numeric_steps: list[tuple[str, object]] = []
        if config.median_imputation:
            numeric_steps.append(("imputer", SimpleImputer(strategy="median")))
        if config.standardize:
            numeric_steps.append(("scaler", StandardScaler()))
        if not numeric_steps:
            numeric_steps.append(("identity", "passthrough"))
        transformers.append(("numeric", Pipeline(numeric_steps), list(numeric_features)))

    if categorical_features:
        categorical_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                ),
                (
                    "one_hot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=not dense_output,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, list(categorical_features)))

    if not transformers:
        raise ValueError("preprocessing requires at least one feature")
    return ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0.0 if dense_output else 1.0,
        verbose_feature_names_out=False,
    )
