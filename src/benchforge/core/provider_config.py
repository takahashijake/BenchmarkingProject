"""Data-only provider contracts, safe to validate without importing extensions."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

IDENTIFIER = r"^[a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*)*$"
VERSION = r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][a-zA-Z0-9.-]+)?$"
Task = Literal["binary_classification", "multiclass_classification", "regression"]


def validate_identifier(value: str) -> str:
    if len(value) > 128 or re.fullmatch(IDENTIFIER, value) is None:
        raise ValueError(f"unsafe provider/component identifier: {value!r}")
    return value


class ProviderData(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ProviderRef(ProviderData):
    id: str = Field(pattern=IDENTIFIER, max_length=128)
    version: str = Field(pattern=VERSION, max_length=128)


class Component(ProviderData):
    id: str = Field(pattern=IDENTIFIER, max_length=128)
    tasks: tuple[Task, ...]

    @field_validator("tasks", mode="before")
    @classmethod
    def normalize_tasks(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def valid_tasks(self) -> Component:
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("supported tasks must be nonempty and unique")
        return self


class EstimatorDeclaration(Component):
    requires_dense: bool = False


class MetricDeclaration(Component):
    direction: Literal["maximize", "minimize"]
    requires_score: bool = False
    display_name: str = Field(min_length=1, max_length=200)
    semantics_version: str = Field(pattern=VERSION, max_length=128)
    validation: Literal["finite-scalar-v1"] = "finite-scalar-v1"

    @model_validator(mode="after")
    def score_contract(self) -> MetricDeclaration:
        if self.requires_score and self.tasks != ("binary_classification",):
            raise ValueError("scalar score metrics currently support binary classification only")
        return self


class SpaceDeclaration(ProviderData):
    id: str = Field(pattern=IDENTIFIER, max_length=128)
    version: int = Field(ge=1)
    # Canonical JSON avoids nested mutable objects in a resolved declaration.
    definition_json: str
    defaults_json: str = "{}"

    @model_validator(mode="after")
    def valid_json(self) -> SpaceDeclaration:
        from benchforge.providers.json import canonical_object

        if canonical_object(self.defaults_json) != self.defaults_json:
            raise ValueError("defaults_json must be a canonical JSON object")
        if canonical_object(self.definition_json) != self.definition_json:
            raise ValueError("search definition_json must be a canonical JSON object")
        return self


class ProviderManifest(ProviderRef):
    api_version: Literal[1] = 1
    datasets: tuple[Component, ...] = ()
    estimators: tuple[EstimatorDeclaration, ...] = ()
    metrics: tuple[MetricDeclaration, ...] = ()
    search_spaces: tuple[SpaceDeclaration, ...] = ()

    @field_validator("datasets", "estimators", "metrics", "search_spaces", mode="before")
    @classmethod
    def normalize_components(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("api_version", mode="before")
    @classmethod
    def exact_api_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("unsupported provider API version; expected integer 1")
        return value

    @model_validator(mode="after")
    def namespace_and_duplicates(self) -> ProviderManifest:
        groups = (self.datasets, self.estimators, self.metrics, self.search_spaces)
        if not any(groups):
            raise ValueError("provider must declare at least one component")
        for group in groups:
            identifiers = [component.id for component in group]
            if len(set(identifiers)) != len(identifiers):
                raise ValueError("duplicate component identifier")
            if any(not name.startswith(self.id + ".") for name in identifiers):
                raise ValueError("component identifiers must be namespaced by provider ID")
        if not {s.id for s in self.search_spaces} <= {m.id for m in self.estimators}:
            raise ValueError("search spaces must reference this provider's estimators")
        return self

    def canonical_dict(self) -> dict[str, object]:
        value = self.model_dump(mode="json")
        for key in ("datasets", "estimators", "metrics", "search_spaces"):
            value[key] = sorted(value[key], key=lambda component: component["id"])
            for component in value[key]:
                if "tasks" in component:
                    component["tasks"] = sorted(component["tasks"])
        return value


class ProviderDistribution(ProviderData):
    """Installed provenance; never interpreted as executable import instructions."""

    id: str = Field(pattern=IDENTIFIER, max_length=128)
    entry_point: str = Field(min_length=1, max_length=1024)
    distribution: str = Field(min_length=1, max_length=255)
    distribution_version: str = Field(min_length=1, max_length=255)
