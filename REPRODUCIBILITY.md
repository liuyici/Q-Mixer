# Reproducibility Notes

The four dataset entry points use the same evaluation protocol.

- Pre-training runs for the fixed `--n_epochs` value. The target labels are not read in the epoch loop.
- Fine-tuning runs for the fixed `--max_iter2` value. No target checkpoint or iteration is selected by accuracy, F1, or AUC.
- Target labels are consumed once, after optimization, for the held-out report only.
- The default seed schedule is five independent seeds: `1, 2, 3, 4, 5` (`--seed` is the first seed and `--num_seeds` controls the count).
- The objective is `L = L_cls + beta * L_adv + gamma * L_mmd`. The four entry points default to `beta=0.01` (adversarial) and `gamma=0.5` (MMD); both remain overrideable through the CLI.
- Class-conditional MMD uses hard target pseudo-labels: `argmax(softmax(logits))`, encoded as one-hot vectors. Target ground-truth labels are never passed to this loss.
- The optimization target loader returns only `Tx`; `Ty` exists only in the separate held-out test loader used after training.
- F1 is weighted. AUC uses softmax probabilities: the positive-class probability for binary tasks and one-vs-rest probabilities for multiclass tasks.
- The quaternion tuple is ordered `(temporal, shared, interaction, spatial)` and uses `D=512`.
- The interaction component is the Hamilton product of the normalized channel and temporal vectors, implemented in each `main.py` as `hamilton_product(c, t)`.
- Parameter counts are generated from the instantiated models, not inferred from the paper: `python tools/count_parameters.py`. The report lists every layer and includes biases and normalization parameters for both `QuantGate` and the domain discriminator. The public D=512 entry points are separate from the private JSED model. The manuscript's 22.6K number is recorded as an author-reported JSED value in `JSED_PARAMETER_MANIFEST.json`; it is not independently executable until the private architecture or a complete layer manifest/checkpoint hash is supplied.

Dataset acquisition and preprocessing disclosures, including the information supported by the supplied clinical protocol, are collected in `DATASET_AND_PREPROCESSING.md`. That document deliberately marks facts that are not present in the protocol or public code instead of fabricating demographic counts or rater statistics.

The exact defaults are recorded in `PAPER_CONFIG.json`. The final report is written to `snapshot.txt` in the selected dataset directory.
