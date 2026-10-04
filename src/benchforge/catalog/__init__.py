"""Local persistent index over immutable experiment evidence."""

from benchforge.catalog.compare import compare
from benchforge.catalog.database import Catalog, CatalogError
from benchforge.catalog.models import CatalogExperiment, Comparison

__all__ = ["Catalog", "CatalogError", "CatalogExperiment", "Comparison", "compare"]
