# Auditing Content Moderation AI for Bias, Adversarial Robustness & Safety

FAST-NUCES Responsible & Explainable AI — Assignment 2.

This repository audits a DistilBERT-based toxicity classifier trained on the Jigsaw Unintended Bias in Toxicity Classification dataset. It measures bias between identity cohorts, exercises two adversarial attacks, applies three mitigation techniques, and wraps the best mitigated model in a three-layer production guardrail pipeline.

## Repository layout

| File | Purpose |
| --- | --- |
| `part1.ipynb` | Fine-tune DistilBERT on the Jigsaw dataset and compute baseline metrics. |
| `part2.ipynb` | Bias audit between high-black and reference (white) cohorts. |
| `part3.ipynb` | Character-level evasion attack and label-flipping poisoning attack. |
| `part4.ipynb` | Reweighing, ThresholdOptimizer, and oversampling mitigations. |
| `part5.ipynb` | End-to-end demonstration of the guardrail pipeline on 1,000 comments. |
| `pipeline.py` | `ModerationPipeline` class with the three-layer `.predict()` method. |
| `requirements.txt` | Pinned dependencies. |

## Environment

- Python 3.10
- GPU: NVIDIA T4 (Google Colab free tier) — CUDA 12.1
- DistilBERT training on 100k rows for 3 epochs takes roughly 30 minutes on a T4.

## Reproducing the work

1. Create a free Kaggle account and accept the competition rules at
   `https://kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification`.
2. Download `jigsaw-unintended-bias-train.csv` and `validation.csv` into a local `data/` directory
   (gitignored — do not commit).
3. Install the pinned dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the notebooks in order. Each notebook saves artefacts that the next one consumes:
   - `part1.ipynb` writes the baseline model checkpoint to `distilbert_baseline/`.
   - `part3.ipynb` writes the poisoned model to `distilbert_poisoned/`.
   - `part4.ipynb` writes the best mitigated model to `distilbert_mitigated/`.
   - `part5.ipynb` loads `distilbert_mitigated/` through `pipeline.py`.
5. Colab users: set Runtime → Change runtime type → GPU before running. CPU runs are
   impractical for the fine-tuning steps.

## Notes

- Dataset files and model checkpoints are excluded from version control (see `.gitignore`). The
  Jigsaw training file alone is roughly 700 MB and model checkpoints are several hundred MB each.
- The fairness audit follows the methodology used in the Stanford NLP 2019 work on the Jigsaw
  dataset — soft identity scores binarised at `≥ 0.5` for the high-black cohort and `white ≥ 0.5
  AND black < 0.1` for the reference cohort.
