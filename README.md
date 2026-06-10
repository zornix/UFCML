# UFCML

UFC fight analysis and winner prediction (1994-2026).

## Contents

- `UFC/ufc-dataset-eda-1994-2026.ipynb` - exploratory data analysis of the fights/fighters datasets
- `UFC/update_data.py` - refresh completed UFCStats events into the gold CSV schema
- `UFC/features.py` - leakage-free feature engineering: every stat a fighter carries into a fight is computed only from prior fights
- `UFC/train.py` - XGBoost winner model with time-series CV, chronological holdout evaluation, selective-prediction metrics, and saved artifacts
- `UFC/train_torch.py` - PyTorch MLP and XGB/MLP blend evaluation
- `UFC/app.py` - Streamlit UI: pick two fighters, get blended probabilities, model-implied odds, confidence, and tale of the tape

## Setup

```bash
pip install xgboost scikit-learn streamlit pandas numpy scipy torch
```

## Train

```bash
cd UFC
python train.py          # full 60-iteration search
python train.py --quick  # fast smoke test
python train_torch.py    # train/evaluate sym-MLP and XGB+MLP blend
```

Current XGB holdout results (970 fights from 2024-03-09 onward, model trained only on earlier fights):
**accuracy 64.6%** (baseline "pick the better win rate": 60.3%), ROC-AUC 0.703, log loss 0.634.

Current best holdout result is the 50/50 sym-MLP + XGB blend:
**accuracy 65.3%**, ROC-AUC 0.710, log loss 0.631.

False positives are treated as 5x more costly than correct confident picks, so the app supports abstention instead of forcing a winner. XGB selective holdout precision:

| confidence | precision | coverage |
|---:|---:|---:|
| 0.55 | 68.8% | 74.2% |
| 0.60 | 72.3% | 48.5% |
| 0.65 | 77.8% | 26.5% |
| 0.70 | 82.3% | 9.9% |
| 0.75 | 90.9% | 2.3% |

## Predict

```bash
cd UFC
streamlit run app.py
```

Pick two fighters; probabilities are symmetrized over corner orderings, so A-vs-B and B-vs-A always agree.

## Notes

- Per-fight stat columns (`F1_KD`, `F1_Sig_Landed`, ...) are outcomes of the fight itself and are never used directly as features - only as history for later fights.
- Fighter_1 wins 64% of fights in the raw data (ufcstats ordering bias); training rows are randomly re-oriented so the target is balanced.
- Draw/NC fights are excluded; training requires both fighters to have at least one prior UFC fight.
