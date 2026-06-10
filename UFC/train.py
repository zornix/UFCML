"""Train, tune (randomized search + time-series CV) and evaluate an XGBoost
UFC winner model, then save artifacts for the Streamlit app.

Usage: python train.py [--quick]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

import features as F

MODEL_DIR = Path(__file__).resolve().parent / 'model'
SEED = 42
SELECTIVE_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75]
RELIABILITY_BINS = 10


def make_dataset():
    fights, fighters = F.load_data()
    X = F.build_fight_matrix(fights, fighters, seed=SEED)

    # Keep fights where both fighters have at least one prior UFC fight —
    # debut fighters carry no history and the UI only serves rostered fighters.
    min_n = (X['total_fights'] - X['d_n_fights'].abs()) / 2
    X = X[min_n >= 1].sort_values('date').reset_index(drop=True)
    return X, fights, fighters


def split_train_test(X: pd.DataFrame, test_frac: float = 0.15):
    cutoff = X['date'].quantile(1 - test_frac)
    train, test = X[X['date'] < cutoff], X[X['date'] >= cutoff]
    return train, test, cutoff


def tune(train: pd.DataFrame, n_iter: int) -> RandomizedSearchCV:
    param_dist = {
        'n_estimators': randint(150, 800),
        'learning_rate': loguniform(0.005, 0.15),
        'max_depth': randint(2, 7),
        'min_child_weight': randint(1, 25),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.5, 0.5),
        'gamma': loguniform(1e-3, 2.0),
        'reg_alpha': loguniform(1e-3, 5.0),
        'reg_lambda': loguniform(0.1, 20.0),
    }
    xgb = XGBClassifier(
        objective='binary:logistic',
        tree_method='hist',
        eval_metric='logloss',
        random_state=SEED,
        n_jobs=1,
    )
    search = RandomizedSearchCV(
        xgb, param_dist,
        n_iter=n_iter,
        cv=TimeSeriesSplit(n_splits=5),
        scoring='neg_log_loss',
        random_state=SEED,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(train[F.MODEL_FEATURES], train['y'])
    return search


def _prediction_frame(test: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    pred_y = (p >= 0.5).astype(int)
    actual_y = test['y'].to_numpy()
    confidence = np.maximum(p, 1 - p)

    out = test[['date', 'fighter_a', 'fighter_b', 'y']].copy()
    out['p_fighter_a'] = p
    out['pred_y'] = pred_y
    out['confidence'] = confidence
    out['predicted_fighter'] = np.where(pred_y == 1, out['fighter_a'], out['fighter_b'])
    out['actual_winner'] = np.where(actual_y == 1, out['fighter_a'], out['fighter_b'])
    out['correct'] = pred_y == actual_y
    return out


def selective_metrics(preds: pd.DataFrame,
                      thresholds: list[float] = SELECTIVE_THRESHOLDS) -> list[dict]:
    rows = []
    n = len(preds)
    for threshold in thresholds:
        called = preds['confidence'] >= threshold
        n_called = int(called.sum())
        correct_called = int(preds.loc[called, 'correct'].sum())
        false_positives = int(n_called - correct_called)
        rows.append({
            'threshold': float(threshold),
            'n_called': n_called,
            'coverage': float(n_called / n) if n else None,
            'precision': float(correct_called / n_called) if n_called else None,
            'correct_picks': correct_called,
            'false_positives': false_positives,
            'abstentions': int(n - n_called),
        })
    return rows


def reliability_metrics(preds: pd.DataFrame, n_bins: int = RELIABILITY_BINS) -> dict:
    """Confidence reliability for the picked fighter, binned from 50%-100%."""
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    confidence = preds['confidence'].to_numpy()
    correct = preds['correct'].astype(float).to_numpy()

    rows = []
    ece = 0.0
    n = len(preds)
    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        count = int(mask.sum())
        if count:
            avg_conf = float(confidence[mask].mean())
            accuracy = float(correct[mask].mean())
            ece += (count / n) * abs(avg_conf - accuracy)
        else:
            avg_conf = None
            accuracy = None
        rows.append({
            'bin_left': float(left),
            'bin_right': float(right),
            'count': count,
            'avg_confidence': avg_conf,
            'accuracy': accuracy,
        })

    return {'ece': float(ece), 'bins': rows}


def write_wrong_pick_audit(preds: pd.DataFrame, path: Path,
                           min_confidence: float = min(SELECTIVE_THRESHOLDS)) -> int:
    audit = preds[(~preds['correct']) & (preds['confidence'] >= min_confidence)].copy()
    audit = audit.sort_values(['confidence', 'date'], ascending=[False, False])
    audit['date'] = pd.to_datetime(audit['date']).dt.date.astype(str)
    audit['picked_probability'] = np.where(
        audit['pred_y'] == 1, audit['p_fighter_a'], 1 - audit['p_fighter_a'])
    audit = audit[[
        'date', 'fighter_a', 'fighter_b', 'predicted_fighter', 'actual_winner',
        'p_fighter_a', 'picked_probability', 'confidence',
    ]]
    audit.to_csv(path, index=False)
    return int(len(audit))


def evaluate(model, test: pd.DataFrame) -> dict:
    p = model.predict_proba(test[F.MODEL_FEATURES])[:, 1]
    y = test['y'].values
    preds = _prediction_frame(test, p)
    return {
        'accuracy': float(accuracy_score(y, p >= 0.5)),
        'roc_auc': float(roc_auc_score(y, p)),
        'log_loss': float(log_loss(y, p)),
        'brier': float(brier_score_loss(y, p)),
        'n_test': int(len(y)),
        'selective': selective_metrics(preds),
        'reliability': reliability_metrics(preds),
    }


def baseline(test: pd.DataFrame) -> dict:
    """Pick the fighter with the better career win rate (coin flip on ties)."""
    d = test['d_win_rate'].fillna(0).values
    rng = np.random.default_rng(SEED)
    pred = np.where(d == 0, rng.random(len(d)) < 0.5, d > 0).astype(int)
    return {'accuracy': float(accuracy_score(test['y'], pred))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='small search for smoke tests')
    args = ap.parse_args()
    n_iter = 8 if args.quick else 60

    X, fights, fighters = make_dataset()
    train, test, cutoff = split_train_test(X)
    print(f'dataset: {len(X)} fights | train {len(train)} | '
          f'test {len(test)} (from {cutoff.date()})')

    search = tune(train, n_iter)
    print(f'\nbest CV log-loss: {-search.best_score_:.4f}')
    print('best params:', json.dumps(search.best_params_, default=float, indent=2))

    metrics = evaluate(search.best_estimator_, test)
    base = baseline(test)
    print(f"\nholdout ({metrics['n_test']} fights from {cutoff.date()}):")
    print(f"  accuracy : {metrics['accuracy']:.4f}  (baseline win-rate pick: {base['accuracy']:.4f})")
    print(f"  roc_auc  : {metrics['roc_auc']:.4f}")
    print(f"  log_loss : {metrics['log_loss']:.4f}")
    print(f"  brier    : {metrics['brier']:.4f}")
    print(f"  ece      : {metrics['reliability']['ece']:.4f}")

    print('\nselective holdout picks:')
    selective_df = pd.DataFrame(metrics['selective'])
    print(selective_df.to_string(index=False, formatters={
        'threshold': '{:.2f}'.format,
        'coverage': lambda x: '' if pd.isna(x) else f'{x:.4f}',
        'precision': lambda x: '' if pd.isna(x) else f'{x:.4f}',
    }))

    print('\nconfidence reliability:')
    reliability_df = pd.DataFrame(metrics['reliability']['bins'])
    reliability_df = reliability_df[reliability_df['count'] > 0]
    print(reliability_df.to_string(index=False, formatters={
        'bin_left': '{:.2f}'.format,
        'bin_right': '{:.2f}'.format,
        'avg_confidence': lambda x: '' if pd.isna(x) else f'{x:.4f}',
        'accuracy': lambda x: '' if pd.isna(x) else f'{x:.4f}',
    }))

    imp = pd.Series(
        search.best_estimator_.feature_importances_, index=F.MODEL_FEATURES,
    ).sort_values(ascending=False)
    print('\ntop features:')
    print(imp.head(15).round(4).to_string())

    # Final model for the app: best params refit on ALL data (the app
    # predicts future fights, so every past fight is fair training game).
    final = XGBClassifier(
        objective='binary:logistic', tree_method='hist',
        eval_metric='logloss', random_state=SEED, n_jobs=-1,
        **search.best_params_,
    )
    final.fit(X[F.MODEL_FEATURES], X['y'])

    MODEL_DIR.mkdir(exist_ok=True)
    holdout_p = search.best_estimator_.predict_proba(test[F.MODEL_FEATURES])[:, 1]
    holdout_preds = _prediction_frame(test, holdout_p)
    audit_path = MODEL_DIR / 'holdout_high_conf_wrong_picks.csv'
    n_audit = write_wrong_pick_audit(holdout_preds, audit_path)
    print(f'\nsaved high-confidence wrong-pick audit ({n_audit} rows) to {audit_path}')

    final.save_model(MODEL_DIR / 'ufc_xgb.ubj')

    snap = F.build_current_snapshot(fights, fighters)
    snap.to_csv(MODEL_DIR / 'fighter_snapshot.csv')

    meta = {
        'model_features': F.MODEL_FEATURES,
        'fighter_features': F.FIGHTER_FEATURES,
        'best_params': {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                        for k, v in search.best_params_.items()},
        'cv_log_loss': float(-search.best_score_),
        'holdout_metrics': metrics,
        'holdout_baseline': base,
        'holdout_cutoff': str(cutoff.date()),
        'holdout_wrong_pick_audit': str(audit_path.name),
        'n_train_fights': int(len(X)),
        'data_max_date': str(X['date'].max().date()),
    }
    (MODEL_DIR / 'metadata.json').write_text(json.dumps(meta, indent=2))
    print(f'\nsaved model, snapshot ({len(snap)} fighters) and metadata to {MODEL_DIR}/')


if __name__ == '__main__':
    main()
