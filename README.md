# responsible-ai-assignment-2

Auditing a DistilBERT-based toxicity classifier trained on the Jigsaw Unintended Bias dataset for
bias, adversarial robustness, and safety. FAST-NUCES Responsible & Explainable AI — Assignment 2.

Short version: I trained a baseline classifier, measured a ~2× FPR gap between comments associated
with Black identity and a reference (white) cohort, broke the classifier with two different
adversarial attacks, then applied three mitigations and wrapped the best mitigated model in a
three-layer production pipeline.

## Layout

| File | What it does |
| --- | --- |
| `part1.ipynb` | Fine-tune DistilBERT on 100k Jigsaw rows; pick an operating threshold. |
| `part2.ipynb` | Cohort-based bias audit (TPR / FPR / FNR / Precision / DI / SPD / EOD). |
| `part3.ipynb` | Character-level evasion attack + 5% label-flipping poisoning attack. |
| `part4.ipynb` | Reweighing + ThresholdOptimizer + oversampling mitigations. |
| `part5.ipynb` | Run the three-layer pipeline on 1,000 comments, with a band sweep. |
| `pipeline.py` | `ModerationPipeline` class used by Part 5. |
| `requirements.txt` | Pinned dependencies. |
| `data/README.md` | Instructions for placing the Jigsaw CSV. |

## Environment

- Python 3.10
- GPU: NVIDIA T4 (Google Colab free tier), CUDA 12.1
- Training DistilBERT on 100k rows for 3 epochs: ~30 minutes per run on a T4

On CPU the fine-tuning steps take hours and the learning-rate schedule behaves differently — use a
GPU runtime.

> **Blackwell (RTX 50-series) note.** `requirements.txt` pins `torch==2.1.2`, which is a CUDA 12.1
> build with no kernels for `sm_120`. If you are reproducing this on a Blackwell GPU (RTX 5070 /
> 5080 / 5090), after `pip install -r requirements.txt` run:
> ```bash
> pip install --force-reinstall --no-deps torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
> ```
> The rest of the pinned stack (`transformers==4.38.2`, `accelerate==0.27.2`, etc.) is unaffected.
> On the original Colab T4 (sm_75), the pinned `torch==2.1.2` works as-is — no override needed.

## Reproducing

1. **Get the data.** Accept the Kaggle rules and download
   `jigsaw-unintended-bias-train.csv` into `data/`. Full instructions in `data/README.md`.
2. **Install.**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the notebooks in order.** Each notebook writes artefacts under `artifacts/` that the
   next one consumes:
   - `part1.ipynb` → `artifacts/{train,eval}.parquet`,
     `artifacts/baseline_eval_{probs,labels}.npy`, checkpoint in `distilbert_baseline/`
   - `part2.ipynb` → `artifacts/part2_summary.csv`
   - `part3.ipynb` → `distilbert_poisoned/` checkpoint
   - `part4.ipynb` → `distilbert_reweighed/`, `distilbert_oversampled/`,
     `artifacts/best_mitigated.json`
   - `part5.ipynb` → loads `pipeline.ModerationPipeline` against the best mitigated model

4. **Quick sanity check** for the regex layer (no GPU, no checkpoint required):
   ```bash
   python pipeline.py
   ```

## Notes on what is and isn't committed

- `*.csv`, `*.parquet`, `*.pt`, `*.bin`, and the `data/`, `artifacts/`, and `distilbert_*/`
  directories are all git-ignored — they're each hundreds of MB and not part of the submission.
- Only the `data/README.md` placeholder inside `data/` is tracked.

## Methodology pointers

- Cohort filters follow the Stanford NLP 2019 methodology on this dataset:
  - **high_black** = `black ≥ 0.5`
  - **reference** = `black < 0.1 AND white ≥ 0.5`
- Evaluation threshold is the macro-F1-maximising `t = 0.4` chosen in Part 1.
- Part 4 treats `group = 1` as the privileged (reference) cohort and `group = 0` as the
  unprivileged (high-black) cohort — the AIF360 convention.
