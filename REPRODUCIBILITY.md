# Implementation notes

## Training and evaluation

The four public entry points use transductive leave-one-subject-out adaptation:

1. Pre-train on all source subjects with classification, source-domain
   adversarial loss, and source-target MMD.
2. Fine-tune with source classification loss. Unlabeled target forward passes
   update batch-normalization statistics.
3. Evaluate the fixed final model of each stage. Target metrics do not select
   checkpoints or stopping iterations.

Pre-training uses label-smoothed cross-entropy (`alpha=0.1`), Adam with
`lr=0.002`, and betas `(0.5, 0.999)`. Fine-tuning uses ordinary cross-entropy and
Adam with `lr=0.0002`. The adversarial and MMD weights are `beta=0.01` and
`gamma=0.5`. Class-conditional MMD uses hard target pseudo-labels.

| Entry point | Pre-training epochs | Fine-tuning steps | Source batch per subject | Fine-tuning batch |
| --- | ---: | ---: | ---: | ---: |
| `BCI_14_7513_sota` | 100 | 33 | 50 | 32 |
| `BCI_2a9_7922_sota` | 100 | 20 | 50 | 144 |
| `BCI_IVNO7_8293_sota` | 15 | 55 | 50 | 32 |
| `BCI_IVA5_8479_sota` | 100 | 102 | 50 | 32 |

The default seeds are 1-5. The final target metrics are accuracy, weighted F1,
and probability-based AUC. Their averages pool subjects and seeds.
The older `snapshot.txt` logs remain in Git history; new logs are ignored by Git.

## Differences from the manuscript

| Manuscript | Public implementation |
| --- | --- |
| Section III-B, Eq. (9): adjacent groups of four, quaternion branch layers, HModReLU | Real-valued temporal and feature-axis linear layers; no adjacent-group construction or HModReLU |
| Section III-C, Eq. (11): direct components `[Ht, (Hc+Ht)/2, Hc⊗Ht, Hc]` | Layer normalization and four independent quaternion projections reduce each component from `D` to `D/4` before assembly |
| Table I / Eq. (17): standard linear classifier | Residual mixer blocks, a multi-layer bottleneck, and a normalized-weight `SLR_layer` classifier |
| Eq. (18): class-conditional alignment | A mixture of marginal and class-conditional RBF-MMD, plus label smoothing during pre-training |
| Section IV-C: 50 MI epochs, Adam at `0.001`; BNCI2014001 batch 144 | The two-stage schedules and batch sizes in the table above |
| Section IV-C: mean and standard deviation across five seeds | The current report gives pooled means; it does not compute across-seed standard deviations |

The MMD mixture is `0.5 * marginal + 0.5 * conditional` for BNCI2014002, and
`marginal + conditional` for the other three entry points. Blankertz2007 and
MI5 use a single temporal window, so those runs do not mix multiple temporal
windows.

## Corrections in this revision

- Use all 13 source subjects in the BNCI2014002 pre-training fold.
- Unpack the single-tensor target batch during fine-tuning.
- Normalize smoothed domain labels by the discriminator's class count.
- Preserve the scalar component and normalize rotation weights to unit length;
  zero weights use the identity rotation.
- Log the actual held-out subject ID and honor the selected device.

These corrections change execution or numerical results. Previously reported
accuracies have not been revalidated with the corrected code.

## Parameter counts

Run `python tools/count_parameters.py` for per-layer and total trainable counts.
The model and domain discriminator are counted separately. Expected totals are
stored in `PAPER_CONFIG.json`.

The manuscript's 22.6K value belongs to the private JSED configuration.
`JSED_PARAMETER_MANIFEST.json` records that reported value; a JSED architecture
or checkpoint is not included here.
