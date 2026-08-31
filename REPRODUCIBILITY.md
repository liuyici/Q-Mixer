# Reproducibility Notes

The four dataset entry points use the same evaluation protocol.

- Pre-training runs for the fixed `--n_epochs` value. The target labels are not read in the epoch loop.
- Fine-tuning runs for the fixed `--max_iter2` value. No target checkpoint or iteration is selected by accuracy, F1, or AUC.
- Target labels are consumed once, after optimization, for the held-out report only.
- The default seed schedule is five independent seeds: `1, 2, 3, 4, 5` (`--seed` is the first seed and `--num_seeds` controls the count).
- The objective is `classification + beta * MMD + gamma * adversarial`, with `beta` and `gamma` exposed as CLI arguments.
- F1 is weighted. AUC uses softmax probabilities: the positive-class probability for binary tasks and one-vs-rest probabilities for multiclass tasks.
- The quaternion tuple is ordered `(temporal, shared, interaction, spatial)` and uses `D=512`.

The exact defaults are recorded in `PAPER_CONFIG.json`. The final report is written to `snapshot.txt` in the selected dataset directory.
