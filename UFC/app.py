"""Streamlit UI for UFC fight winner prediction.

Run with:  streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from xgboost import XGBClassifier

import features as F

MODEL_DIR = Path(__file__).resolve().parent / 'model'
FALSE_POSITIVE_COST = 5
BREAK_EVEN_PRECISION = FALSE_POSITIVE_COST / (FALSE_POSITIVE_COST + 1)
CONFIDENT_PICK_THRESHOLD = 0.75
LEAN_THRESHOLD = 0.60
BLEND_WEIGHT_XGB = 0.50
BLEND_WEIGHT_MLP = 0.50

st.set_page_config(page_title='UFC Fight Predictor', page_icon='🥊', layout='wide')


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden: list[int], dropout: float, norm: str):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h) if norm == 'batch' else nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


@st.cache_resource
def load_artifacts():
    xgb = XGBClassifier()
    xgb.load_model(MODEL_DIR / 'ufc_xgb.ubj')

    mlp_artifact = torch.load(MODEL_DIR / 'ufc_mlp.pt', map_location='cpu', weights_only=False)
    arch = mlp_artifact['architecture']
    mlp = MLP(
        arch['in_features'],
        list(arch['hidden']),
        float(arch['dropout']),
        arch.get('norm', 'batch'),
    )
    mlp.load_state_dict(mlp_artifact['model_state_dict'])
    mlp.eval()

    snap = pd.read_csv(MODEL_DIR / 'fighter_snapshot.csv', index_col='fighter',
                       parse_dates=['last_fight', 'DOB'])
    meta = json.loads((MODEL_DIR / 'metadata.json').read_text())
    meta_torch = json.loads((MODEL_DIR / 'metadata_torch.json').read_text())
    return xgb, mlp, mlp_artifact, snap, meta, meta_torch


def _transform_mlp_features(matchup: pd.DataFrame, artifact: dict) -> np.ndarray:
    prep = artifact['preprocessing']
    features = list(prep['features'])
    arr = matchup[features].values.astype(np.float32)
    medians = np.asarray(prep['medians'], dtype=np.float32)
    mean = np.asarray(prep['mean'], dtype=np.float32)
    std = np.asarray(prep['std'], dtype=np.float32)
    indicator_mask = np.asarray(prep['indicator_mask'], dtype=bool)

    indicators = np.isnan(arr[:, indicator_mask]).astype(np.float32)
    for j in range(arr.shape[1]):
        nans = np.isnan(arr[:, j])
        if nans.any():
            arr[nans, j] = medians[j]
    arr = (arr - mean) / std
    return np.concatenate([arr, indicators], axis=1).astype(np.float32)


def _mlp_probability(mlp: MLP, features_arr: np.ndarray) -> float:
    with torch.no_grad():
        logits = mlp(torch.tensor(features_arr, dtype=torch.float32))
        return float(torch.sigmoid(logits)[0].item())


def predict(xgb, mlp, mlp_artifact, snap, name_a: str, name_b: str) -> dict:
    """P(fighter A beats fighter B), symmetrized over both orientations."""
    a = snap.loc[[name_a]].reset_index(drop=True)
    b = snap.loc[[name_b]].reset_index(drop=True)
    ab = F.build_matchup_features(a, b)
    ba = F.build_matchup_features(b, a)

    p_xgb_ab = xgb.predict_proba(ab[F.MODEL_FEATURES])[0, 1]
    p_xgb_ba = xgb.predict_proba(ba[F.MODEL_FEATURES])[0, 1]
    p_xgb = float((p_xgb_ab + 1 - p_xgb_ba) / 2)

    p_mlp_ab = _mlp_probability(mlp, _transform_mlp_features(ab, mlp_artifact))
    p_mlp_ba = _mlp_probability(mlp, _transform_mlp_features(ba, mlp_artifact))
    p_mlp = float((p_mlp_ab + 1 - p_mlp_ba) / 2)

    p_blend = BLEND_WEIGHT_XGB * p_xgb + BLEND_WEIGHT_MLP * p_mlp
    return {
        'blend': float(p_blend),
        'xgb': p_xgb,
        'mlp': p_mlp,
        'disagreement': float(abs(p_xgb - p_mlp)),
    }


def confidence_bucket(p_a: float, name_a: str, name_b: str):
    """Return product-facing confidence bucket and leading fighter details."""
    leader, p_leader = (name_a, p_a) if p_a >= 0.5 else (name_b, 1 - p_a)
    if p_leader >= CONFIDENT_PICK_THRESHOLD:
        return 'Confident pick', leader, p_leader
    if p_leader >= LEAN_THRESHOLD:
        return 'Lean', leader, p_leader
    return 'No pick', leader, p_leader


def american_odds(p: float) -> str:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    if p >= 0.5:
        return f'{round(-100 * p / (1 - p)):+.0f}'
    return f'{round(100 * (1 - p) / p):+.0f}'


def decimal_odds(p: float) -> str:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return f'{1 / p:.2f}'


def tape_row(label, va, vb, fmt='{:.2f}', higher_better=None):
    def show(v):
        return '—' if pd.isna(v) else fmt.format(v)
    return {'Fighter A': show(va), 'Stat': label, 'Fighter B': show(vb)}


xgb, mlp, mlp_artifact, snap, meta, meta_torch = load_artifacts()

st.title('🥊 UFC Fight Predictor')
st.caption(
    f"50/50 XGBoost + sym-MLP blend on {meta['n_train_fights']:,} UFC fights "
    f"through {meta['data_max_date']} · XGB holdout accuracy "
    f"{meta['holdout_metrics']['accuracy']:.1%} · AUC "
    f"{meta['holdout_metrics']['roc_auc']:.3f}")

# Eligible fighters: at least one UFC fight in the dataset.
names = snap.index.sort_values().tolist()

col_a, col_vs, col_b = st.columns([5, 1, 5])
with col_a:
    fighter_a = st.selectbox('Fighter A (red corner)', names,
                             index=names.index('Islam Makhachev') if 'Islam Makhachev' in names else 0)
with col_vs:
    st.markdown("<h2 style='text-align:center; padding-top:1.2rem;'>vs</h2>",
                unsafe_allow_html=True)
with col_b:
    fighter_b = st.selectbox('Fighter B (blue corner)', names,
                             index=names.index('Charles Oliveira') if 'Charles Oliveira' in names else 1)

if fighter_a == fighter_b:
    st.warning('Pick two different fighters.')
    st.stop()

p = predict(xgb, mlp, mlp_artifact, snap, fighter_a, fighter_b)
p_a = p['blend']
bucket, leader, p_leader = confidence_bucket(p_a, fighter_a, fighter_b)
leader_odds = american_odds(p_leader)
leader_decimal = decimal_odds(p_leader)

st.divider()
if bucket == 'No pick':
    st.markdown(
        f"<h3 style='text-align:center'>No pick: model confidence is too close "
        f"to call</h3>",
        unsafe_allow_html=True)
    st.caption(
        f"Current edge: {leader} at {p_leader:.1%}. Picks are withheld below "
        f"{LEAN_THRESHOLD:.0%} confidence.")
else:
    color = '#E8003D' if bucket == 'Confident pick' else '#D98200'
    st.markdown(
        f"<h3 style='text-align:center'>{bucket}: "
        f"<span style='color:{color}'>{leader}</span> · {p_leader:.1%}</h3>",
        unsafe_allow_html=True)
    st.caption(
        f"Confident picks require at least {CONFIDENT_PICK_THRESHOLD:.0%}; "
        f"leans require at least {LEAN_THRESHOLD:.0%}. With false positives priced "
        f"{FALSE_POSITIVE_COST}x higher, break-even precision is "
        f"{BREAK_EVEN_PRECISION:.1%}.")

bar_a, bar_b = st.columns(2)
bar_a.metric(
    f'{fighter_a} blend probability',
    f'{p_a:.1%}',
    f'{american_odds(p_a)} / {decimal_odds(p_a)}',
    border=True,
)
bar_b.metric(
    f'{fighter_b} blend probability',
    f'{1 - p_a:.1%}',
    f'{american_odds(1 - p_a)} / {decimal_odds(1 - p_a)}',
    border=True,
)

scale_value = int(round(p_a * 100))
st.slider(
    'Blend probability scale',
    min_value=0,
    max_value=100,
    value=scale_value,
    disabled=True,
    help='Model-implied probability for Fighter A on a 0-100 scale.',
)
st.progress(p_a, text=f'{fighter_a} {p_a:.1%} ←→ {fighter_b} {1 - p_a:.1%}')

odds_a, odds_leader, agreement = st.columns(3)
odds_a.metric('Model-implied odds', f'{leader_odds}', f'decimal {leader_decimal}', border=True)
odds_leader.metric('Pick confidence', f'{p_leader:.1%}', bucket, border=True)
agreement.metric('Model disagreement', f"{p['disagreement']:.1%}", 'XGB vs MLP', border=True)

with st.expander('Model probabilities'):
    probs = pd.DataFrame([
        ('XGBoost', p['xgb'], 1 - p['xgb']),
        ('Sym-MLP', p['mlp'], 1 - p['mlp']),
        ('50/50 blend', p['blend'], 1 - p['blend']),
    ], columns=['Model', fighter_a, fighter_b]).set_index('Model')
    st.dataframe(probs.style.format('{:.1%}'), width='stretch')

# ── Tale of the tape ─────────────────────────────────────────────────────────
st.divider()
st.subheader('Tale of the tape')

a, b = snap.loc[fighter_a], snap.loc[fighter_b]
rows = [
    ('Record (career W-L-D)', f"{a['Wins']:.0f}-{a['Losses']:.0f}-{a['Draws']:.0f}",
     f"{b['Wins']:.0f}-{b['Losses']:.0f}-{b['Draws']:.0f}"),
    ('UFC fights', f"{a['n_fights']:.0f}", f"{b['n_fights']:.0f}"),
    ('UFC win rate', f"{a['win_rate']:.0%}" if pd.notna(a['win_rate']) else '—',
     f"{b['win_rate']:.0%}" if pd.notna(b['win_rate']) else '—'),
    ('Streak', f"{a['streak']:+.0f}", f"{b['streak']:+.0f}"),
    ('Elo rating', f"{a['elo']:.0f}", f"{b['elo']:.0f}"),
    ('Elo momentum (last 3)', f"{a['elo_change3']:+.0f}", f"{b['elo_change3']:+.0f}"),
    ('Age', f"{a['age']:.0f}" if pd.notna(a['age']) else '—',
     f"{b['age']:.0f}" if pd.notna(b['age']) else '—'),
    ('Height', f"{a['height_cm']:.0f} cm" if pd.notna(a['height_cm']) else '—',
     f"{b['height_cm']:.0f} cm" if pd.notna(b['height_cm']) else '—'),
    ('Reach', f"{a['reach_cm']:.0f} cm" if pd.notna(a['reach_cm']) else '—',
     f"{b['reach_cm']:.0f} cm" if pd.notna(b['reach_cm']) else '—'),
    ('Stance', a['Stance'] if pd.notna(a['Stance']) else '—',
     b['Stance'] if pd.notna(b['Stance']) else '—'),
    ('Sig. strikes landed/min', f"{a['slpm']:.2f}", f"{b['slpm']:.2f}"),
    ('Sig. strikes absorbed/min', f"{a['sapm']:.2f}", f"{b['sapm']:.2f}"),
    ('Striking accuracy', f"{a['str_acc']:.0%}" if pd.notna(a['str_acc']) else '—',
     f"{b['str_acc']:.0%}" if pd.notna(b['str_acc']) else '—'),
    ('Striking defense', f"{a['str_def']:.0%}" if pd.notna(a['str_def']) else '—',
     f"{b['str_def']:.0%}" if pd.notna(b['str_def']) else '—'),
    ('Takedowns /15 min', f"{a['td_per15']:.2f}", f"{b['td_per15']:.2f}"),
    ('Takedown defense', f"{a['td_def']:.0%}" if pd.notna(a['td_def']) else '—',
     f"{b['td_def']:.0%}" if pd.notna(b['td_def']) else '—'),
    ('Sub attempts /15 min', f"{a['sub_per15']:.2f}", f"{b['sub_per15']:.2f}"),
    ('Control time %', f"{a['ctrl_pct']:.0%}" if pd.notna(a['ctrl_pct']) else '—',
     f"{b['ctrl_pct']:.0%}" if pd.notna(b['ctrl_pct']) else '—'),
    ('KO wins (share of UFC fights)', f"{a['ko_win_pct']:.0%}", f"{b['ko_win_pct']:.0%}"),
    ('Sub wins (share of UFC fights)', f"{a['sub_win_pct']:.0%}", f"{b['sub_win_pct']:.0%}"),
    ('Last fight', a['last_fight'].date().isoformat(), b['last_fight'].date().isoformat()),
]
tape = pd.DataFrame(rows, columns=['Stat', fighter_a, fighter_b]).set_index('Stat')
st.table(tape)

st.caption(
    'Stats are computed from UFC fight history only (point-in-time features, '
    'same as the model was trained on). Probabilities are symmetrized over both '
    'corner orderings. Odds are model-implied, not market prices. For '
    'entertainment purposes — not betting advice.')
