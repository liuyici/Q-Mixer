# Dataset And Preprocessing Disclosure

This document separates facts that are visible in this repository, facts
provided by the supplied clinical protocol, and fields that are still required
for an independently reproducible manuscript. It is intentionally explicit
about unknown values; no demographic or clinical statistic is inferred from a
protocol document.

## Public EEG Benchmarks

### Motor imagery (MI)

The paper describes the public MI inputs as standard MOABB preprocessing with
notch and band-pass filtering, followed by tangent-space feature extraction.
The checked-in MATLAB examples make the following operations explicit:

- `BCIIV2a.m`: event type 768; 1,000 samples starting at event position +500;
  22 channels at 250 Hz; sixth-order Chebyshev type-II 4--40 Hz band-pass with
  60 dB stop-band attenuation; NaNs replaced by zero.
- `BCIIV2b.m`: event type 768; 1,000 samples starting at event position +750;
  3 channels at 250 Hz; the same sixth-order Chebyshev type-II 4--40 Hz
  band-pass and NaN handling.
- The checked-in examples do not apply rereferencing or ICA. Standardization is
  commented out in those MATLAB files. The training entry points consume
  precomputed `.mat` tangent/log-map features, so the exact feature-generation
  command and its metadata must accompany any released MI result.
- `centroid_align.py` uses a mean covariance reference (Riemannian,
  log-Euclidean, or Euclidean according to the selected option) and does not
  use target labels. `logmap.py` maps the covariance representation to tangent
  features.

The phrase "adjacent channels" is not a preprocessing operation in the public
MI scripts. No electrode reordering or topology file is included; therefore a
manuscript must state the channel order supplied by each source dataset before
describing neighboring-channel quaternion construction.

### SEED features

The public SEED loader reads official differential-entropy/LDS feature files,
swaps the stored axes, retains the last 185 samples per trial, and creates
12-sample windows with stride one. Target features are z-scored using the
unlabeled target feature array. This is a transductive feature-normalization
step and must be stated as such; target ground-truth labels are not used for
normalization, checkpoint selection, or loss computation.

The repository does not regenerate the official DE files. The release should
therefore record the exact official feature version, session selection, and
normalization provenance used to create those files.

## JSED Clinical Dataset

### What the supplied protocol supports

The supplied investigator-initiated protocol describes a pediatric dental
fear/anxiety study with children aged 5--8 years, ASA I/II health status, and
guardian consent. Its stated inclusion criteria are:

- ASA I/II;
- guardian SCARED-P score below 25;
- retained primary-tooth extraction, asymptomatic caries filling, or qualifying
  chronic-pulpitis root-canal treatment; and
- child/guardian informed consent.

Its stated exclusions are traumatic medical/oral-surgery history or severe
non-cooperation, epilepsy/autism/neurologic disease affecting VR or requiring
general anesthesia, severe motion sickness/VR intolerance, and SCARED-P at or
above 25.

The protocol plans three intervention conditions (active VR, passive VR, and
conventional Tell--Show--Do) crossed with three treatment types, with a planned
450 participants (50 per cell). This planned sample is not the JSED analysis
sample. The manuscript reports that only the conventional TSD arm was used for
the EEG benchmark and that EEG quality control left 25 valid subjects, one EEG
channel, 500 Hz sampling, three anxiety classes, and 3,750 EEG trials.

The protocol defines the physiological time points as:

- `T0`: pre-treatment baseline; MCDASf and five minutes of resting ECG/EEG;
- `T1`: an intra-operative high-anxiety step (for example injection or
  high-speed handpiece), with continuous physiology and behavior assessment;
- `T2`: immediately post-treatment; MCDASf and Wong--Baker FPS-R, followed by
  device removal.

The protocol separately labels saliva collection as baseline, immediate
post-treatment, and 20-minute recovery. The manuscript should not silently
reuse `T1/T2/T3` for the EEG/ECG time points.

### JSED signal processing stated by the protocol/paper

- EEG was acquired from Fp1/Fp2 in the clinical protocol; the manuscript's
  benchmark is described as one channel at 500 Hz.
- The stated frequency bands are delta 1--4 Hz, theta 4--8 Hz, alpha 8--13 Hz,
  beta 13--30 Hz, and gamma 30--45 Hz.
- The paper's feature description uses four-second EEG segments, linear
  detrending per segment, a one-second Welch window with 0.5-second stride,
  and ten narrow bands: delta 1--4, theta-low 4--6, theta-high 6--8,
  alpha-low 8--10, alpha-high 10--13, beta-low 13--18, beta-mid 18--24,
  beta-high 24--30, gamma-low 30--38, and gamma-high 38--45 Hz.
- The reported feature types are log power, relative power, and PSD-based
  differential entropy. No channel-topology operation is possible for a
  single-channel benchmark.
- The paper describes hard target pseudo-labels (`argmax` of the classifier)
  for class-conditional MMD. The four public entry points now implement this
  rule using one-hot pseudo-labels.

The public repository does not contain the JSED recording, quality-control
code, or the private JSED model implementation. Consequently, the exact
artifact-rejection rule, rereference, filter design/order, rejected-segment
counts, channel choice when both Fp1/Fp2 are available, and DE normalization
fit set cannot be verified from this repository. These items must be added to
the manuscript or supplied to reviewers with the private archive.

### Clinical reporting required before claiming reliability

The current attachments do not provide the following realized-study values and
they must not be fabricated:

- enrolled and excluded counts by arm and treatment type, with the timing and
  reason for every exclusion (including the 13 subjects shown as excluded in
  Figure 3);
- age summary, sex counts, and per-class/per-arm sample counts for the 25
  analyzed subjects;
- the operational thresholds mapping MCDASf, FLACC-R, Frankl, FPS-R, or
  SCARED-P to severe/moderate/mild anxiety;
- assessor training and inter-rater agreement (for example Cohen kappa or an
  ICC with confidence interval);
- whether EEG quality control and exclusion were specified before inspecting
  class labels/outcomes, and whether assessors were blinded; and
- the exact subject IDs and trial counts contributing to each class.

Because a four-second segment with one-second windows and 0.5-second stride
creates overlapping, correlated examples, JSED results must use the subject as
the independent unit. Report leave-one-subject-out predictions, per-subject
metrics, a subject-level mean and confidence interval, and a clustered or
subject-bootstrap uncertainty estimate. Trial-level standard deviations alone
do not establish clinical reliability. The manuscript should describe the
reported approximately 60% accuracy and its large between-subject variation as
an exploratory benchmark unless the above subject-level analysis supports a
stronger claim.

## Reproduction checklist

For every released result, record the following in the same commit as the
configuration:

1. raw-data source/version and subject split;
2. filter, rereference, artifact, detrending, segmentation, and feature
   extraction parameters;
3. normalization fit set (source-only, target-unlabeled, or other);
4. channel order/topology, if a multi-channel branch is used;
5. hard/soft pseudo-label rule and confidence filtering, if any;
6. subject-level aggregation and confidence-interval method; and
7. ethics approval, registry identifier, consent, anonymization, and the
   private-data access procedure.

The protocol reports ethics approval `PJ2025-022-01` and Chinese Clinical Trial
Registry identifier `ChiCTR2500106971`; these identify the study but do not
replace reporting the realized JSED cohort and preprocessing audit trail.
