"""Explicit activation and immutable composition of installed provider bundles."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from importlib import metadata
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from benchforge.core.config import ExperimentConfig
from benchforge.core.provider_config import ProviderManifest, ProviderRef, validate_identifier
from benchforge.data.registry import (
    Dataset,
    DatasetLoader,
    DatasetRegistry,
    default_dataset_registry,
)
from benchforge.models.registry import ModelRegistry, ModelSpec, default_model_registry
from benchforge.search.registry import SearchSpaceRegistry, default_search_space_registry
from benchforge.search.spaces import SearchSpace

MetricFunction = Callable[[NDArray[Any], NDArray[Any], NDArray[np.float64] | None], float]
GROUP = "benchforge.providers"


@dataclass(frozen=True)
class Provider:
    """Entry-point factory result. Callbacks must be stateless and fold-local."""

    manifest: ProviderManifest
    datasets: Mapping[str, DatasetLoader]
    estimators: Mapping[str, ModelSpec]
    metrics: Mapping[str, MetricFunction]
    search_spaces: Mapping[str, SearchSpace]

    def __post_init__(self) -> None:
        for field in ("datasets", "estimators", "metrics", "search_spaces"):
            object.__setattr__(self, field, MappingProxyType(dict(getattr(self, field))))


@dataclass(frozen=True)
class DiscoveredProvider:
    id: str
    entry_point: str
    distribution: str
    distribution_version: str


@dataclass(frozen=True)
class ResolvedProviders:
    manifests: tuple[ProviderManifest, ...]
    provenance: tuple[DiscoveredProvider, ...]
    datasets: DatasetRegistry
    models: ModelRegistry
    spaces: SearchSpaceRegistry
    metrics: Mapping[str, MetricFunction]


def _entries() -> tuple[metadata.EntryPoint, ...]:
    try:
        return tuple(metadata.entry_points(group=GROUP))
    except Exception as exc:
        raise ValueError(f"cannot discover installed providers: {exc}") from exc


def _describe(entry: metadata.EntryPoint) -> DiscoveredProvider:
    validate_identifier(entry.name)
    distribution = entry.dist
    return DiscoveredProvider(
        entry.name,
        entry.value,
        distribution.metadata["Name"] if distribution else "unknown",
        distribution.version if distribution else "unknown",
    )


def discover() -> tuple[DiscoveredProvider, ...]:
    """Read installed package metadata only; never import entry-point targets."""
    descriptions = tuple(_describe(entry) for entry in _entries())
    ids = [item.id for item in descriptions]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate installed provider IDs; uninstall conflicting distributions")
    return tuple(sorted(descriptions, key=lambda item: item.id))


def _load(
    ref: ProviderRef, entries: tuple[metadata.EntryPoint, ...]
) -> tuple[Provider, DiscoveredProvider]:
    selected = [entry for entry in entries if entry.name == ref.id]
    if len(selected) != 1:
        raise ValueError(
            f"provider '{ref.id}' requires exactly one installed entry point; found {len(selected)}"
        )
    entry = selected[0]
    try:
        factory = entry.load()
        provider = factory()
    except Exception as exc:
        raise ValueError(f"provider '{ref.id}' could not initialize: {exc}") from exc
    if not isinstance(provider, Provider):
        raise ValueError(
            f"provider '{ref.id}' factory must return benchforge.providers.runtime.Provider"
        )
    if not isinstance(provider.manifest, ProviderManifest):
        raise ValueError(f"provider {ref.id!r} has an invalid manifest")
    manifest = ProviderManifest.model_validate(provider.manifest.model_dump(mode="json"))
    if manifest.id != ref.id or manifest.version != ref.version:
        raise ValueError(f"provider '{ref.id}' identity/version mismatch; requested {ref.version}")
    for field in ("datasets", "estimators", "metrics", "search_spaces"):
        expected = {component.id for component in getattr(manifest, field)}
        actual = getattr(provider, field)
        if set(actual) != expected:
            raise ValueError(f"provider '{ref.id}' {field} callbacks disagree with declarations")
    if any(
        not callable(value) for value in (*provider.datasets.values(), *provider.metrics.values())
    ):
        raise ValueError(f"provider '{ref.id}' dataset/metric callbacks must be callable")
    for declaration in manifest.estimators:
        spec = provider.estimators[declaration.id]
        if not isinstance(spec, ModelSpec) or not callable(spec.factory):
            raise ValueError(f"invalid estimator factory: {declaration.id}")
        if {task.value for task in spec.supported_tasks} != set(
            declaration.tasks
        ) or spec.capabilities.requires_dense != declaration.requires_dense:
            raise ValueError(f"estimator capabilities disagree: {declaration.id}")
    for space_declaration in manifest.search_spaces:
        import json

        space = provider.search_spaces[space_declaration.id]
        if (
            not isinstance(space, SearchSpace)
            or space.name != space_declaration.id
            or space.version != space_declaration.version
            or space.definition != json.loads(space_declaration.definition_json)
            or space.defaults != json.loads(space_declaration.defaults_json)
            or not callable(space.suggest)
        ):
            raise ValueError(f"search space identity disagrees: {space_declaration.id}")
    return replace(provider, manifest=manifest), _describe(entry)


def resolve(refs: tuple[ProviderRef, ...]) -> ResolvedProviders:
    """Load exactly the selected, version-pinned providers into new registries."""
    if len({ref.id for ref in refs}) != len(refs):
        raise ValueError("duplicate provider identifier")
    entries = _entries() if refs else ()
    manifests = []
    provenance = []
    datasets: dict[str, DatasetLoader] = {}
    models: dict[str, ModelSpec] = {}
    spaces: dict[str, SearchSpace] = {}
    metrics: dict[str, MetricFunction] = {}
    for ref in sorted(refs, key=lambda item: item.id):
        provider, origin = _load(ref, entries)
        manifests.append(provider.manifest)
        provenance.append(origin)
        _merge(datasets, provider.datasets, default_dataset_registry.names, ref.id)
        _merge(models, provider.estimators, default_model_registry.names, ref.id)
        _merge(spaces, provider.search_spaces, default_search_space_registry.names, ref.id)
        _merge(metrics, provider.metrics, (), ref.id)
    return ResolvedProviders(
        tuple(manifests),
        tuple(provenance),
        default_dataset_registry.extended(_validated_loaders(datasets, tuple(manifests))),
        default_model_registry.extended(models),
        default_search_space_registry.extended(spaces),
        MappingProxyType(metrics),
    )


def _merge[V](
    destination: dict[str, V],
    incoming: Mapping[str, V],
    builtins: tuple[str, ...],
    provider_id: str,
) -> None:
    if set(incoming) & (set(destination) | set(builtins)):
        raise ValueError(f"provider '{provider_id}' has duplicate component identifiers")
    destination.update(incoming)


def _validated_loaders(
    loaders: dict[str, DatasetLoader], manifests: tuple[ProviderManifest, ...]
) -> dict[str, DatasetLoader]:
    declarations = {c.id: c for m in manifests for c in m.datasets}

    def checked(name: str, loader: DatasetLoader) -> DatasetLoader:
        def load() -> Dataset:
            from benchforge.providers.validation import validate_dataset

            try:
                dataset = loader()
                validate_dataset(dataset, declarations[name].tasks)
            except Exception as exc:
                raise ValueError(f"provider dataset '{name}' failed to load: {exc}") from exc
            import hashlib
            import json

            from benchforge.data.identity import dataset_digest

            schema = json.dumps(
                {
                    "content": dataset_digest(dataset),
                    "target_labels": dataset.target_labels,
                    "target_name": dataset.target_name,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            digest = hashlib.sha256(schema.encode()).hexdigest()
            return replace(dataset, identity=f"{dataset.identity}:sha256:{digest}")

        return load

    return {name: checked(name, loader) for name, loader in loaders.items()}


def activate[C: ExperimentConfig](config: C) -> tuple[C, ResolvedProviders | None]:
    """Resolve a configuration copy; never mutate the caller's configuration."""
    if not config.providers:
        if config.provider_manifests:
            raise ValueError("provider manifests require explicit provider selections")
        for metric in getattr(config, "metrics", ()):
            config.metric_spec(metric)
        return config, None
    runtime = config._provider_runtime or resolve(config.providers)
    if {p.id: p.version for p in runtime.manifests} != {p.id: p.version for p in config.providers}:
        raise ValueError("resolved provider identity/version differs from requested selection")
    if config.provider_manifests and (
        sorted((m.canonical_dict() for m in config.provider_manifests), key=lambda m: str(m["id"]))
        != sorted((m.canonical_dict() for m in runtime.manifests), key=lambda m: str(m["id"]))
    ):
        raise ValueError("installed provider declarations differ from experiment metadata")
    bound = config.model_copy(update={"provider_manifests": runtime.manifests})
    bound._provider_runtime = runtime
    task = bound.task
    for metric in getattr(bound, "metrics", ()):
        if not bound.metric_spec(metric).supports_task(task):
            raise ValueError(f"metric '{metric}' is incompatible with task '{task}'")
    return bound, runtime
