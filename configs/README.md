# Experiment configurations

Create a separate, version-controlled config for each experiment ID. The
`cpu-smoke.json` example uses the fixed 10,000-image fit subset and official
10,000-image validation partition. Preserve the saved split seed across all
experiments; representation-comparison code must use identical
`cluster_train`, `cluster_validation`, and `cluster_test` index arrays.
