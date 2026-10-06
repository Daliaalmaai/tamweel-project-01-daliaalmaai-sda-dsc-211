# Tamweel Lite — Financing Default Risk

An educational machine learning project for estimating the probability
of default within 90 days after a financing application, using information
available at application time.

**Course:** SDA-DSC-211 — Advanced Machine Learning Methods  
**Project type:** Individual learner project  
**Current stage:** Day 1 — Baseline and boosting comparison completed  
**Initial candidate:** XGBoost, subject to further validation

> This project uses synthetic course data. It must not be used to make
> real financing decisions. This is a learner repository, not an official
> SDAIA repository.

[Executed Day 1 notebook](notebooks/01_baseline_boosting.ipynb) ·
[Model comparison](artifacts/day1_model_comparison.csv) ·
[Written decision](artifacts/day1_reflection.json)

## Project Overview

The project investigates whether machine learning models can estimate
90-day default risk from application-time features.

The five-day workflow progressively adds validation, decision analysis,
interpretation, calibration, and final documentation.

Day 1 establishes a reproducible baseline and compares Logistic Regression,
XGBoost, and LightGBM. The objective is to select an initial candidate
using evidence rather than model complexity or reputation.

## Project Progress

| Stage | Focus | Status |
|---|---|---|
| Readiness | Environment and data checks | Completed |
| Day 1 | Baseline and boosting comparison | Completed |
| Day 2 | Customer-aware and time-aware validation; tuning | Planned |
| Day 3 | Class imbalance and decision costs | Planned |
| Day 4 | Interpretation and calibration | Planned |
| Day 5 | Ensembles, Model Card, and final delivery | Planned |

## Dataset and Prediction Task

All records are synthetic.

| Item | Value |
|---|---|
| Training applications | 10,000 |
| Model predictors | 22 |
| Target | `default_within_90d` |
| Overall default prevalence | 7.89% |
| Missing feature cells | 766 |
| Prediction time | At application submission |

The target records whether a default event occurs within 90 days after
the application. It does not mean a delay of exactly 90 days.

Identifiers, dates, split-control columns, and the target are excluded
from model inputs. Challenge data is not used in the Day 1 comparison.

## Day 1 Technical Pipeline

1. Prepare the free CPU environment and verify pinned course files.
2. Load and inspect the synthetic training data.
3. Separate development, internal selection, and comparison roles.
4. Fit a Logistic Regression baseline using a preprocessing pipeline.
5. Select boosting tree counts using inner-stop log loss.
6. Refit the boosting models on all development rows.
7. Evaluate all three models on the same comparison rows.
8. Export predictions, metrics, figures, split membership, and reflection.

### Data Roles

| Role | Applications | Purpose |
|---|---:|---|
| Inner fit | 6,000 | Fit preprocessing and trees during tree-count selection |
| Inner stop | 2,000 | Select the tree count using log loss |
| Development | 8,000 | Fit or refit models after internal decisions are fixed |
| Comparison | 2,000 | Evaluate the fitted models |

Inner fit and inner stop are subsets of development, not additional data.
The comparison set is not used for early stopping or tree-count selection.

### Baseline and Boosting

The Logistic Regression baseline uses median imputation, feature scaling,
and classification in a single pipeline. Learned preprocessing values
come from development rows only.

For boosting, tree-count selection uses the internal split, followed by
refitting on all development rows.

| Model | Selected trees | Evaluated rounds |
|---|---:|---:|
| XGBoost | 61 | 91 |
| LightGBM | 41 | 71 |

Neither model reached the 300-tree search ceiling.

## Key Results — Day 1

Evaluation used 2,000 comparison applications, including 158 defaults
(7.9% prevalence).

| Model | ROC-AUC | Average Precision | Training time (s) | Selected trees |
|---|---:|---:|---:|---:|
| Logistic Regression | **0.8213** | 0.3258 | **0.0386** | Not applicable |
| XGBoost | 0.8124 | **0.3338** | 0.3610 | 61 |
| LightGBM | 0.8138 | 0.3248 | 0.2556 | 41 |

Training time includes tree-count selection and refitting for boosting,
and fitting for Logistic Regression. Download and plotting time are
excluded. These timings describe this run, not a hardware guarantee.

In this course, PR-AUC refers to scikit-learn Average Precision,
not trapezoidal integration of the precision–recall curve.

[Full comparison table](artifacts/day1_model_comparison.csv) ·
[Comparison predictions](artifacts/day1_comparison_predictions.csv)

## Initial Model-Selection Decision

XGBoost was selected as an initial candidate because it achieved the
highest Average Precision: 0.3338 versus 0.3258 for Logistic Regression.

The improvement was modest at 0.0080. Logistic Regression was faster
and achieved a higher ROC-AUC, making it a strong competing baseline.

This selection is provisional. If XGBoost's AP advantage disappears
under customer-aware and time-aware validation, Logistic Regression
may be preferred for its simplicity and faster training.

## Visual Evidence

### Learning Curves

Training loss continued to decline after inner-stop loss stopped
improving. Tree counts were selected using the internal rows only.

![Day 1 learning curves](artifacts/day1_learning_curves.png)

### ROC and Precision–Recall

The curves are close and intersect. No model dominates across all
operating points. The precision reference line represents the
comparison prevalence of 7.9%.

![Day 1 ROC and precision–recall curves](artifacts/day1_roc_pr.png)

## Interpretation and Limitations

- Accuracy alone is unsuitable as the headline metric: predicting
  non-default for every application would achieve approximately 92%
  accuracy while detecting no defaults.
- AP is considered alongside ROC-AUC because defaults are uncommon.
- There are 1,226 customers shared between development and comparison.
- The random split does not fully address customer overlap, temporal
  leakage, or target-maturity boundaries.
- This single split provides no confidence interval and does not
  establish general model superiority.
- These results do not establish probability calibration or operational
  financing value.
- Day 1 does not select a final approval, review, or rejection threshold.

## Reproducibility

| Setting | Value |
|---|---|
| Runtime | Google Colab, free CPU |
| Python | 3.13.16 |
| Seed | 211 |
| CPU threads | 2 |
| Mode | `FAST_MODE=True` |
| Maximum boosting trees | 300 |
| Early-stopping patience | 30 |
| Learning rate | 0.05 |
| XGBoost depth | 3 |
| LightGBM leaves | 15 |
| LightGBM minimum child samples | 50 |

Package versions are recorded in
[environment.json](artifacts/environment.json).

Run configuration, file hashes, selection curves, and split summaries
are recorded in [day1_run.json](artifacts/day1_run.json).

### Run the Day 1 Notebook

1. Open [the executed notebook](notebooks/01_baseline_boosting.ipynb).
2. Open it in Google Colab and save a personal copy before editing.
3. Select the free CPU runtime.
4. Execute cells sequentially, waiting for each cell to finish.
5. Complete the learner reflection using the new run's evidence.
6. Run the export cell and download the evidence files.

No paid subscription, GPU, API key, or Google Drive mount is required.

If the runtime restarts, rerun the prerequisite cells in order.
Training times may change between runs; update the reflection to match
the exported comparison table.

## Repository Structure

```text
.
├── README.md
├── notebooks/       # Readiness and daily notebooks
├── artifacts/       # Metrics, predictions, figures, and run evidence
├── data/            # Synthetic course data and data contract
├── reports/         # Project report templates and completed reports
├── presentation/    # Presentation materials
├── submission/      # Final prediction and manifest files
├── tamweel/         # Project inference code
└── scripts/         # Setup, validation, and supporting utilities
```

## Day 1 Evidence Map

| Evidence | File |
|---|---|
| Executed notebook | [01_baseline_boosting.ipynb](notebooks/01_baseline_boosting.ipynb) |
| Model comparison | [day1_model_comparison.csv](artifacts/day1_model_comparison.csv) |
| Prediction evidence | [day1_comparison_predictions.csv](artifacts/day1_comparison_predictions.csv) |
| Split membership | [day1_split_membership.csv](artifacts/day1_split_membership.csv) |
| Learning curves | [day1_learning_curves.png](artifacts/day1_learning_curves.png) |
| ROC and precision–recall | [day1_roc_pr.png](artifacts/day1_roc_pr.png) |
| Written reflection | [day1_reflection.json](artifacts/day1_reflection.json) |
| Run metadata | [day1_run.json](artifacts/day1_run.json) |
| Environment | [environment.json](artifacts/environment.json) |

## References and Attribution

- Course: **SDA-DSC-211 — Advanced Machine Learning Methods**
- Course instructor: **Meaad Al-Marri**
- [Course template](https://github.com/almiyead-rgb/sda-dsc-211-student-template)
- [Day 1 guide](https://github.com/almiyead-rgb/sda-dsc-211-student-template/blob/main/DAY1_GUIDE.md)
- [SDAIA Academy on GitHub](https://github.com/SDAIAAcademy)

The notebook and supporting code originate from the course template.
Results were produced through the learner's executed Colab run.

ChatGPT/Codex assistance was used to explain code and metrics, check
artifact consistency, and draft the reflection and README wording.
