# BenchForge V0 architecture

Dependency flow is intentionally simple: the CLI calls the execution API; execution coordinates
domain services; domain services depend on typed core contracts. Dataset loading, model creation,
evaluation, and storage are separate from orchestration.

```text
CLI -> execution runner -> data registry
                     |---> split planner
                     |---> preprocessing builder + model registry
                     |---> evaluation
                     `---> artifact store
```

For every fold, the runner constructs a new sklearn `Pipeline` containing a new preprocessor and a
new estimator. It then fits that pipeline only on the training indices. This construction is the
central V0 leakage boundary. Out-of-fold records retain the original sample index and fold ID.

The configuration fingerprint is SHA-256 over canonical, sorted JSON. It covers experiment
semantics but excludes the artifact output directory. Creation time and runtime metadata are also
excluded. Consequently, moving output or rerunning later does not change experiment identity.

Registries currently use immutable instance-local mappings. They provide a narrow extension point
without a plugin framework or mutable process-global registration. Future dataset sources and model
families can be added behind these contracts while the execution lifecycle remains unchanged.
