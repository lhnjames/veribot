# VeriBot research code

This repository collects the existing VeriBot implementation and experiment utilities in one place.

- `source/verifiable_bot/`: source-grounded evidence construction, graph models, source interventions, coalition search, and verification helpers.
- `source/hgt_import/`: HGT and RelNet model implementations and data-building utilities.
- `source/scripts/`: training, evaluation, and analysis entry points accumulated during the project.
- `source/configs/`: experiment configurations.
- `source/tests/`: focused model and source-processing checks.
- `source/legacy_v1_backup/`: earlier code kept for reference.
- `tools/`: data preparation and evidence utilities from the project root.

The export contains code and configuration only. Dataset files, model checkpoints, predictions, logs, cached features, virtual environments, manuscripts, and access credentials are excluded. Dataset paths and environment-specific settings in some research scripts must be configured for a new machine. The active research direction and evaluation claims should be determined from the corresponding experiments rather than inferred from file names.

The main Python requirements are listed in `source/requirements-verifiable.txt`.
