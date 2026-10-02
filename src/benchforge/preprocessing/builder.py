from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from benchforge.core.config import PreprocessingConfig


def build_preprocessor(
    config: PreprocessingConfig, numeric_features: Sequence[str]
) -> ColumnTransformer:
    steps: list[tuple[str, object]] = []
    if config.median_imputation:
        steps.append(("imputer", SimpleImputer(strategy="median")))
    if config.standardize:
        steps.append(("scaler", StandardScaler()))
    if not steps:
        steps.append(("identity", "passthrough"))
    return ColumnTransformer(
        [("numeric", Pipeline(steps), list(numeric_features))],
        remainder="drop",
        verbose_feature_names_out=False,
    )
