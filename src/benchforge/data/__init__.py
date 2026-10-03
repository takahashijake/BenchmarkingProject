"""Dataset contracts and registry."""

from benchforge.data.registry import Dataset, DatasetRegistry, default_dataset_registry
from benchforge.data.tabular import load_tabular_dataset

__all__ = ["Dataset", "DatasetRegistry", "default_dataset_registry", "load_tabular_dataset"]
