# Dataset placeholder

The Jigsaw CSVs are **not** committed to this repo — they are 700MB+ and against
the submission rules in Section 5 of the assignment.

Place the two files below in this directory before running the notebooks:

- `jigsaw-unintended-bias-train.csv`
- `validation.csv` *(optional, only for independent sanity checks)*

## How to get them

1. Sign in at [kaggle.com](https://kaggle.com).
2. Accept the rules at
   <https://kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification>.
3. From the competition **Data** tab, download only `jigsaw-unintended-bias-train.csv`
   (and `validation.csv` if you want the sanity-check path).

Or, with the Kaggle CLI:

```bash
pip install kaggle
kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification \
  -f jigsaw-unintended-bias-train.csv -p data/
unzip data/jigsaw-unintended-bias-train.csv.zip -d data/
```

The `.gitignore` at the repo root already excludes `*.csv`, so nothing in here
will be committed by accident.
