"""PyTorch MLP for UFC fight winner prediction.

Reuses the exact same dataset construction and train/test split as train.py
(make_dataset + split_train_test logic).  A temporal validation slice is carved
from the training period for early stopping.

Usage:
    python train_torch.py          # full run
    python train_torch.py --quick  # smoke test (fewer epochs / configs)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

# ── local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F

MODEL_DIR = Path(__file__).resolve().parent / 'model'
SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ══════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int = SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════════════
# Dataset helpers  (mirrors train.py, no import of that module)
# ══════════════════════════════════════════════════════════════════════════════

def make_dataset():
    fights, fighters = F.load_data()
    X = F.build_fight_matrix(fights, fighters, seed=SEED)
    min_n = (X['total_fights'] - X['d_n_fights'].abs()) / 2
    X = X[min_n >= 1].sort_values('date').reset_index(drop=True)
    return X, fights, fighters


def split_train_test(X: pd.DataFrame, test_frac: float = 0.15):
    cutoff = X['date'].quantile(1 - test_frac)
    train = X[X['date'] < cutoff].copy()
    test  = X[X['date'] >= cutoff].copy()
    return train, test, cutoff


def split_train_val(train: pd.DataFrame, val_frac: float = 0.15):
    """Temporal validation slice from the *end* of the training period."""
    cutoff = train['date'].quantile(1 - val_frac)
    tr  = train[train['date'] < cutoff].copy()
    val = train[train['date'] >= cutoff].copy()
    return tr, val, cutoff


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing  (fit on train, apply to val/test)
# ══════════════════════════════════════════════════════════════════════════════

class TabularPreprocessor:
    """Median imputation + missing indicator + standardisation.

    For comparability with the XGBoost baseline the same MODEL_FEATURES list
    is used; NaN-heavy columns get a binary indicator appended.
    """

    MISSING_THRESH = 0.01   # add indicator for columns with >1% NaN in train

    def fit(self, X: pd.DataFrame, feature_cols: Sequence[str]):
        self.features = list(feature_cols)
        arr = X[self.features].values.astype(np.float32)

        self.medians = np.nanmedian(arr, axis=0)

        miss_rate = np.isnan(arr).mean(axis=0)
        self.indicator_mask = miss_rate > self.MISSING_THRESH
        self.indicator_names = [f'{c}_missing' for c, m in
                                zip(self.features, self.indicator_mask) if m]

        filled = arr.copy()
        for j in range(filled.shape[1]):
            nans = np.isnan(filled[:, j])
            if nans.any():
                filled[nans, j] = self.medians[j]

        self.mean = filled.mean(axis=0)
        self.std  = filled.std(axis=0)
        self.std[self.std < 1e-8] = 1.0    # avoid div-by-zero for constant cols
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        arr = X[self.features].values.astype(np.float32)

        # Binary missing indicators (before imputation)
        indicators = arr[:, self.indicator_mask]
        ind_cols = (~np.isnan(indicators)).astype(np.float32)   # 1 = present, 0 = missing
        # Actually encode missingness: 1 if missing, 0 if present
        ind_cols = np.isnan(indicators).astype(np.float32)

        # Impute
        for j in range(arr.shape[1]):
            nans = np.isnan(arr[:, j])
            if nans.any():
                arr[nans, j] = self.medians[j]

        # Standardise
        arr = (arr - self.mean) / self.std

        return np.concatenate([arr, ind_cols], axis=1).astype(np.float32)

    @property
    def n_features(self) -> int:
        return len(self.features) + int(self.indicator_mask.sum())


# ══════════════════════════════════════════════════════════════════════════════
# Model definition
# ══════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    def __init__(self, in_features: int, hidden: Sequence[int],
                 dropout: float = 0.4, norm: str = 'batch'):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if norm == 'batch':
                layers.append(nn.BatchNorm1d(h))
            else:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
        X_tr: np.ndarray, y_tr: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        hidden: Sequence[int],
        dropout: float,
        lr: float,
        weight_decay: float,
        batch_size: int,
        max_epochs: int,
        patience: int,
        device: torch.device,
        norm: str = 'batch',
) -> tuple[MLP, int, list[float]]:
    seed_everything(SEED)

    model = MLP(X_tr.shape[1], hidden, dropout=dropout, norm=norm).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    crit = nn.BCEWithLogitsLoss()

    tr_ds  = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.float32))
    tr_loader  = DataLoader(tr_ds,  batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=512,        shuffle=False)

    best_val_loss = float('inf')
    best_state    = None
    best_epoch    = 0
    no_improve    = 0
    val_losses    = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vl_sum = 0.0
            vl_cnt = 0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vl_sum += crit(model(xb), yb).item() * len(yb)
                vl_cnt += len(yb)
        vl = vl_sum / vl_cnt
        val_losses.append(vl)

        if vl < best_val_loss - 1e-5:
            best_val_loss = vl
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch    = epoch
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_epoch, val_losses


# ══════════════════════════════════════════════════════════════════════════════
# Inference helpers
# ══════════════════════════════════════════════════════════════════════════════

def predict_proba(model: MLP, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    ds  = DataLoader(TensorDataset(torch.tensor(X)), batch_size=512, shuffle=False)
    probs = []
    with torch.no_grad():
        for (xb,) in ds:
            logits = model(xb.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def predict_proba_sym(
        model: MLP,
        X_pos: np.ndarray,
        X_neg: np.ndarray,
        device: torch.device,
) -> np.ndarray:
    """Antisymmetric inference: p_sym = (p(x) + 1 - p(-x)) / 2.

    d_* features negate when fighters are swapped, so p(-x) corresponds to
    the negated feature vector.  The average enforces the consistency
    constraint p(A vs B) + p(B vs A) = 1.
    """
    p_pos = predict_proba(model, X_pos, device)
    p_neg = predict_proba(model, X_neg, device)
    return (p_pos + 1.0 - p_neg) / 2.0


def metrics_dict(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        'accuracy': float(accuracy_score(y, p >= 0.5)),
        'roc_auc':  float(roc_auc_score(y, p)),
        'log_loss': float(log_loss(y, p)),
        'brier':    float(brier_score_loss(y, p)),
        'n_test':   int(len(y)),
    }


def selective_metrics(
        y: np.ndarray,
        p: np.ndarray,
        thresholds: Sequence[float] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75),
) -> list[dict]:
    """Precision/coverage when the model may abstain on low-confidence fights."""
    y_bool = y.astype(bool)
    pred_bool = p >= 0.5
    confidence = np.maximum(p, 1.0 - p)
    rows = []

    for threshold in thresholds:
        called = confidence >= threshold
        n_called = int(called.sum())
        n_total = int(len(y))
        correct = int((pred_bool[called] == y_bool[called]).sum()) if n_called else 0
        false_positives = n_called - correct
        rows.append({
            'threshold': float(threshold),
            'called': n_called,
            'coverage': float(n_called / n_total) if n_total else 0.0,
            'abstentions': int(n_total - n_called),
            'precision': float(correct / n_called) if n_called else None,
            'false_positives': false_positives,
        })

    return rows


def disagreement_veto_metrics(
        y: np.ndarray,
        p_pick: np.ndarray,
        p_xgb: np.ndarray,
        p_mlp: np.ndarray,
        thresholds: Sequence[float] = (0.55, 0.60, 0.65, 0.70),
        max_disagreements: Sequence[float] = (0.05, 0.10, 0.15, 0.20),
) -> list[dict]:
    """Candidate abstention rules using confidence plus XGB/MLP agreement."""
    y_bool = y.astype(bool)
    pred_bool = p_pick >= 0.5
    confidence = np.maximum(p_pick, 1.0 - p_pick)
    disagreement = np.abs(p_xgb - p_mlp)
    rows = []

    for threshold in thresholds:
        for max_disagreement in max_disagreements:
            called = (confidence >= threshold) & (disagreement <= max_disagreement)
            n_called = int(called.sum())
            n_total = int(len(y))
            correct = int((pred_bool[called] == y_bool[called]).sum()) if n_called else 0
            false_positives = n_called - correct
            rows.append({
                'threshold': float(threshold),
                'max_disagreement': float(max_disagreement),
                'called': n_called,
                'coverage': float(n_called / n_total) if n_total else 0.0,
                'abstentions': int(n_total - n_called),
                'precision': float(correct / n_called) if n_called else None,
                'false_positives': false_positives,
                'avg_disagreement': float(disagreement[called].mean()) if n_called else None,
            })

    return rows


def print_selective_table(title: str, rows: list[dict]) -> None:
    print(f'\n── {title} ─────────────────────────────────────────────')
    header = (f"{'conf>=':>7} {'called':>7} {'coverage':>9} {'precision':>10} "
              f"{'false+':>7} {'abstain':>8}")
    print(header)
    print('─' * len(header))
    for r in rows:
        precision = 'n/a' if r['precision'] is None else f"{r['precision']:.4f}"
        print(f"{r['threshold']:>7.2f} {r['called']:>7} {r['coverage']:>9.2%} "
              f"{precision:>10} {r['false_positives']:>7} {r['abstentions']:>8}")


def print_disagreement_veto_table(title: str, rows: list[dict]) -> None:
    print(f'\n── {title} ─────────────────────────────────────────────')
    header = (f"{'conf>=':>7} {'dis<=':>7} {'called':>7} {'coverage':>9} "
              f"{'precision':>10} {'false+':>7} {'abstain':>8}")
    print(header)
    print('─' * len(header))
    for r in rows:
        precision = 'n/a' if r['precision'] is None else f"{r['precision']:.4f}"
        print(f"{r['threshold']:>7.2f} {r['max_disagreement']:>7.2f} "
              f"{r['called']:>7} {r['coverage']:>9.2%} {precision:>10} "
              f"{r['false_positives']:>7} {r['abstentions']:>8}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='fewer epochs + one config for smoke tests')
    args = ap.parse_args()

    seed_everything(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        print(f'  CUDA: {torch.version.cuda}')

    # ── Dataset ────────────────────────────────────────────────────────────
    print('\nBuilding dataset ...')
    X, fights, fighters = make_dataset()
    train_df, test_df, test_cutoff = split_train_test(X, test_frac=0.15)
    tr_df, val_df, val_cutoff = split_train_val(train_df, val_frac=0.15)

    print(f'  total {len(X)} | train {len(tr_df)} | val {len(val_df)} | '
          f'test {len(test_df)} (from {test_cutoff.date()})')
    print(f'  val cutoff: {val_cutoff.date()}')

    # ── Preprocessing ──────────────────────────────────────────────────────
    prep = TabularPreprocessor().fit(tr_df, F.MODEL_FEATURES)
    print(f'  features after preprocessing: {prep.n_features} '
          f'({len(F.MODEL_FEATURES)} base + '
          f'{prep.n_features - len(F.MODEL_FEATURES)} missing indicators)')

    X_tr   = prep.transform(tr_df)
    X_val  = prep.transform(val_df)
    X_te   = prep.transform(test_df)
    # Negated test features for antisymmetric inference
    # Only the d_* columns should be negated; symmetric cols (mean_*, total_*,
    # open_stance) stay the same.  Missing indicators are also symmetric.
    d_cols_mask = np.array(
        [c.startswith('d_') for c in F.MODEL_FEATURES] +
        [False] * (prep.n_features - len(F.MODEL_FEATURES)),
        dtype=bool,
    )
    X_te_neg = X_te.copy()
    X_te_neg[:, d_cols_mask] *= -1.0

    y_tr  = tr_df['y'].values.astype(np.float32)
    y_val = val_df['y'].values.astype(np.float32)
    y_te  = test_df['y'].values.astype(np.float32)

    # ── Config sweep ───────────────────────────────────────────────────────
    # Small manual grid; winner picked by validation log-loss.
    if args.quick:
        configs = [
            dict(hidden=(128, 64), dropout=0.4, lr=3e-4,
                 weight_decay=1e-3, batch_size=128, max_epochs=50,  patience=10),
        ]
    else:
        configs = [
            dict(hidden=(256, 128),     dropout=0.3, lr=3e-4,
                 weight_decay=1e-3, batch_size=128, max_epochs=300, patience=30),
            dict(hidden=(256, 128, 64), dropout=0.4, lr=3e-4,
                 weight_decay=1e-3, batch_size=128, max_epochs=300, patience=30),
            dict(hidden=(128, 64),      dropout=0.4, lr=1e-3,
                 weight_decay=5e-4, batch_size=256, max_epochs=300, patience=30),
            dict(hidden=(256, 128),     dropout=0.5, lr=1e-3,
                 weight_decay=5e-3, batch_size=128, max_epochs=300, patience=30),
        ]

    best_val_ll  = float('inf')
    best_model   = None
    best_cfg     = None
    best_ep      = None

    print('\n── Config sweep (picked on validation log-loss) ────────────────────')
    for i, cfg in enumerate(configs):
        seed_everything(SEED)
        model, ep, val_losses = train_model(
            X_tr, y_tr, X_val, y_val,
            device=device,
            norm='batch',
            **cfg,
        )
        p_val = predict_proba(model, X_val, device)
        val_ll = log_loss(y_val, np.clip(p_val, 1e-7, 1 - 1e-7))
        val_acc = accuracy_score(y_val, p_val >= 0.5)
        print(f'  cfg {i+1}/{len(configs)} hidden={cfg["hidden"]} '
              f'drop={cfg["dropout"]} lr={cfg["lr"]} '
              f'→ val_ll={val_ll:.4f}  val_acc={val_acc:.4f}  '
              f'stopped_ep={ep}')
        if val_ll < best_val_ll:
            best_val_ll = val_ll
            best_model  = model
            best_cfg    = cfg
            best_ep     = ep

    print(f'\nBest config: hidden={best_cfg["hidden"]} dropout={best_cfg["dropout"]} '
          f'lr={best_cfg["lr"]} wd={best_cfg["weight_decay"]} '
          f'bs={best_cfg["batch_size"]}')
    print(f'Early-stopping epoch: {best_ep}')

    # ── MLP holdout evaluation ─────────────────────────────────────────────
    p_mlp     = predict_proba(best_model, X_te, device)
    p_mlp_sym = predict_proba_sym(best_model, X_te, X_te_neg, device)

    m_mlp     = metrics_dict(y_te, p_mlp)
    m_mlp_sym = metrics_dict(y_te, p_mlp_sym)

    # ── XGBoost honest baseline (best_params from metadata, train-only fit) ─
    print('\nTraining holdout-honest XGBoost (best_params from metadata.json) ...')
    meta_path = MODEL_DIR / 'metadata.json'
    with open(meta_path) as f:
        saved_meta = json.load(f)

    bp = saved_meta['best_params']
    xgb = XGBClassifier(
        objective='binary:logistic',
        tree_method='hist',
        eval_metric='logloss',
        random_state=SEED,
        n_jobs=-1,
        n_estimators=int(bp['n_estimators']),
        learning_rate=float(bp['learning_rate']),
        max_depth=int(bp['max_depth']),
        min_child_weight=int(bp['min_child_weight']),
        subsample=float(bp['subsample']),
        colsample_bytree=float(bp['colsample_bytree']),
        gamma=float(bp['gamma']),
        reg_alpha=float(bp['reg_alpha']),
        reg_lambda=float(bp['reg_lambda']),
    )
    xgb.fit(train_df[F.MODEL_FEATURES], train_df['y'])
    p_xgb = xgb.predict_proba(test_df[F.MODEL_FEATURES])[:, 1]
    m_xgb = metrics_dict(y_te, p_xgb)

    # ── Blend ──────────────────────────────────────────────────────────────
    p_blend = 0.5 * p_mlp + 0.5 * p_xgb
    m_blend = metrics_dict(y_te, p_blend)

    p_blend_sym = 0.5 * p_mlp_sym + 0.5 * p_xgb
    m_blend_sym = metrics_dict(y_te, p_blend_sym)

    # ── Results table ──────────────────────────────────────────────────────
    xgb_saved = saved_meta['holdout_metrics']
    header = f"{'Model':<25} {'accuracy':>9} {'roc_auc':>9} {'log_loss':>10} {'brier':>8}"
    div    = '─' * len(header)
    print(f'\n{div}')
    print(header)
    print(div)

    def row(name, m):
        return (f"{name:<25} {m['accuracy']:>9.4f} {m['roc_auc']:>9.4f} "
                f"{m['log_loss']:>10.4f} {m['brier']:>8.4f}")

    print(row('XGB (saved baseline)',   xgb_saved))
    print(row('XGB (train-only refit)', m_xgb))
    print(row('MLP',                    m_mlp))
    print(row('MLP (symmetrized)',       m_mlp_sym))
    print(row('Blend MLP+XGB',          m_blend))
    print(row('Blend sym-MLP+XGB',      m_blend_sym))
    print(div)
    print(f'  test set: {int(m_mlp["n_test"])} fights  (from {test_cutoff.date()})')

    # ── Selective-prediction / false-positive reporting ───────────────────
    selective_report = {
        'mlp_sym': selective_metrics(y_te, p_mlp_sym),
        'blend_sym_mlp_xgb': selective_metrics(y_te, p_blend_sym),
        'blend_sym_mlp_xgb_disagreement_veto': disagreement_veto_metrics(
            y_te,
            p_blend_sym,
            p_xgb,
            p_mlp_sym,
        ),
    }
    print_selective_table(
        'Selective prediction: MLP (symmetrized)',
        selective_report['mlp_sym'],
    )
    print_selective_table(
        'Selective prediction: Blend sym-MLP+XGB',
        selective_report['blend_sym_mlp_xgb'],
    )
    print_disagreement_veto_table(
        'Blend sym-MLP+XGB with XGB/MLP disagreement veto candidates',
        selective_report['blend_sym_mlp_xgb_disagreement_veto'],
    )

    # ── Save model artifact ────────────────────────────────────────────────
    MODEL_DIR.mkdir(exist_ok=True)
    torch.save({
        'model_state_dict':  best_model.state_dict(),
        'architecture': {
            'in_features':  prep.n_features,
            'hidden':       list(best_cfg['hidden']),
            'dropout':      best_cfg['dropout'],
            'norm':         'batch',
        },
        'preprocessing': {
            'features':        prep.features,
            'medians':         prep.medians,
            'mean':            prep.mean,
            'std':             prep.std,
            'indicator_mask':  prep.indicator_mask,
        },
        'model_features': F.MODEL_FEATURES,
    }, MODEL_DIR / 'ufc_mlp.pt')

    meta_torch = {
        'model_features':    F.MODEL_FEATURES,
        'n_input_features':  prep.n_features,
        'missing_indicators': prep.indicator_names,
        'architecture': {
            'hidden':    list(best_cfg['hidden']),
            'dropout':   best_cfg['dropout'],
            'norm':      'batch',
        },
        'training_config': {
            'lr':           best_cfg['lr'],
            'weight_decay': best_cfg['weight_decay'],
            'batch_size':   best_cfg['batch_size'],
            'max_epochs':   best_cfg['max_epochs'],
            'patience':     best_cfg['patience'],
            'best_epoch':   best_ep,
        },
        'holdout_cutoff': str(test_cutoff.date()),
        'holdout_metrics': {
            'xgb_saved_baseline': xgb_saved,
            'xgb_train_only':     m_xgb,
            'mlp':                m_mlp,
            'mlp_sym':            m_mlp_sym,
            'blend_mlp_xgb':      m_blend,
            'blend_sym_mlp_xgb':  m_blend_sym,
        },
        'selective_prediction': selective_report,
        'n_train':  int(len(tr_df)),
        'n_val':    int(len(val_df)),
        'n_test':   int(len(test_df)),
    }
    (MODEL_DIR / 'metadata_torch.json').write_text(json.dumps(meta_torch, indent=2))
    print(f'\nArtifacts saved to {MODEL_DIR}/')
    print('  ufc_mlp.pt')
    print('  metadata_torch.json')

    # ── Honest verdict ─────────────────────────────────────────────────────
    print('\n── Verdict ─────────────────────────────────────────────────────────')
    mlp_better_auc  = m_mlp['roc_auc']  > xgb_saved['roc_auc']
    mlp_better_ll   = m_mlp['log_loss'] < xgb_saved['log_loss']
    blend_better_ll = m_blend['log_loss'] < xgb_saved['log_loss']
    sym_helps       = m_mlp_sym['log_loss'] < m_mlp['log_loss']

    print(f"MLP vs XGB (saved): AUC {'MLP wins' if mlp_better_auc else 'XGB wins'}, "
          f"log-loss {'MLP wins' if mlp_better_ll else 'XGB wins'}")
    print(f"Symmetrized MLP: log-loss {'improves' if sym_helps else 'no improvement'} "
          f"({m_mlp_sym['log_loss']:.4f} vs {m_mlp['log_loss']:.4f})")
    print(f"Blend: log-loss {'improves on XGB' if blend_better_ll else 'does not beat XGB'} "
          f"({m_blend['log_loss']:.4f} vs {xgb_saved['log_loss']:.4f})")
    print()
    print('Deep learning on small tabular data (~6 k training rows, 35 features):')
    print('  - GBDTs (XGBoost) typically win outright on accuracy/AUC.')
    print('  - An MLP may help calibration and can contribute to a blend.')
    print('  - The blend is worth keeping if it lowers log-loss ≥ ~0.005 vs XGB alone.')
    print('  - The MLP alone is not recommended as the primary model for this dataset.')


if __name__ == '__main__':
    main()
