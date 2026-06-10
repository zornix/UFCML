"""Feature engineering for UFC fight winner prediction.

Builds *point-in-time* career features for each fighter: every statistic a
fighter carries into a fight is computed only from fights that happened
before it. The raw per-fight stat columns (F1_KD, F1_Sig_Landed, ...) are
outcomes of the fight itself and are never used directly as features.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

# Half-life in years for exponential time decay of rate/volume career stats.
# Set to None to disable decay (original behaviour).
DECAY_HALF_LIFE: float | None = 5.0

# CUM_COLS that carry win/loss record info — kept undecayed so record-based
# features (win_rate, ko_win_pct, etc.) stay unbiased counts.
_UNDECAYED_COLS = {'won', 'lost', 'ko_win', 'sub_win', 'dec_win', 'ko_loss', 'sub_loss'}

# Weight-class lookup: division name keyword -> weight limit in lbs.
# Women's divisions share the same weight limits as men's. Longest key
# first so "light heavyweight" is matched before "lightweight".
_DIVISION_LBS: list[tuple[str, float]] = sorted([
    ('super heavyweight', 265.0),
    ('light heavyweight', 205.0),
    ('strawweight',       115.0),
    ('flyweight',         125.0),
    ('bantamweight',      135.0),
    ('featherweight',     145.0),
    ('lightweight',       155.0),
    ('welterweight',      170.0),
    ('middleweight',      185.0),
    ('heavyweight',       265.0),
], key=lambda x: -len(x[0]))


def _weight_class_kg(wc) -> float:
    """Parse a Weight_Class string to the division weight limit in kg.

    Returns NaN for Catch Weight, Open Weight, or unrecognised strings.
    """
    if pd.isna(wc):
        return np.nan
    s = str(wc).lower()
    if 'catch' in s or 'open weight' in s:
        return np.nan
    for div, lbs in _DIVISION_LBS:
        if div in s:
            return lbs * 0.453592
    return np.nan


# Per-fight quantities accumulated into career totals (cum_*) per fighter.
CUM_COLS = [
    'won', 'lost', 'time_sec',
    'kd', 'opp_kd',
    'sig_l', 'sig_a', 'opp_sig_l', 'opp_sig_a',
    'td_l', 'td_a', 'opp_td_l', 'opp_td_a',
    'sub_att', 'ctrl',
    'ko_win', 'sub_win', 'dec_win', 'ko_loss', 'sub_loss',
]

# Point-in-time career features derived for each fighter.
CAREER_FEATURES = [
    'n_fights', 'win_rate', 'streak', 'recent3_win_rate',
    'slpm', 'str_acc', 'sapm', 'str_def',
    'td_per15', 'td_acc', 'td_def', 'opp_td_per15',
    'sub_per15', 'kd_per15', 'kd_abs_per15', 'ctrl_pct',
    'ko_win_pct', 'sub_win_pct', 'dec_win_pct', 'ko_loss_pct', 'sub_loss_pct',
    'avg_fight_min', 'layoff_days', 'age',
    'elo', 'opp_elo_avg', 'elo_change3',
]

# Elo configuration. Start/K follow the classic chess defaults; finishes move
# ratings more than decisions (a KO/sub is stronger evidence of a skill gap).
ELO_START = 1500.0
ELO_K = 32.0
ELO_FINISH_MULT = 1.25

# Opponent-quality weighting for career stat accumulation: each fight's
# contribution is scaled by (opp_pre_fight_elo / ELO_START) ** ELO_WEIGHT_K,
# discounting stats padded against weak opposition. 0 disables weighting.
ELO_WEIGHT_K = 1.0

PHYSICAL_FEATURES = ['height_cm', 'reach_cm', 'weight_kg', 'southpaw']

FIGHTER_FEATURES = CAREER_FEATURES + PHYSICAL_FEATURES


def _height_to_cm(h):
    m = re.match(r"(\d+)'\s*(\d+)\"", str(h))
    return int(m.group(1)) * 30.48 + int(m.group(2)) * 2.54 if m else np.nan


def _weight_to_kg(w):
    m = re.match(r"([\d.]+)\s*lbs", str(w))
    return float(m.group(1)) * 0.453592 if m else np.nan


def _reach_to_cm(r):
    m = re.match(r"([\d.]+)\"", str(r))
    return float(m.group(1)) * 2.54 if m else np.nan


def load_data(data_dir: Path | str = DATA_DIR):
    """Load and clean the fights and fighters tables."""
    data_dir = Path(data_dir)
    fights = pd.read_csv(data_dir / 'ufc_gold_dataset_final.csv')
    fighters = pd.read_csv(data_dir / 'ufc_fighters_final.csv')

    fights['Event_Date'] = pd.to_datetime(fights['Event_Date'])
    fights = fights.sort_values(['Event_Date'], kind='stable').reset_index(drop=True)

    fighters = fighters.drop_duplicates(subset='Fighter_Name', keep='first').copy()
    fighters['height_cm'] = fighters['Height'].map(_height_to_cm)
    fighters['weight_kg'] = fighters['Weight'].map(_weight_to_kg)
    fighters['reach_cm'] = fighters['Reach'].map(_reach_to_cm)
    fighters['southpaw'] = (fighters['Stance'] == 'Southpaw').astype(float)
    fighters.loc[fighters['Stance'].isna(), 'southpaw'] = np.nan
    fighters['DOB'] = pd.to_datetime(fighters['DOB'], errors='coerce')
    return fights, fighters


def _method_bucket(m: str) -> str:
    m = str(m).upper()
    if 'KO' in m or 'TKO' in m or 'DOCTOR' in m:
        return 'ko'
    if 'SUBMISSION' in m:
        return 'sub'
    if 'DECISION' in m:
        return 'dec'
    return 'other'


def compute_elo(fights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sequential Elo over all fights in chronological (index) order.

    Returns:
      per_fight: indexed like `fights`, with pre-fight values per side —
        {F1,F2}_elo, {F1,F2}_opp_elo_avg (mean Elo of prior opponents, i.e.
        strength of schedule) and {F1,F2}_elo_change3 (Elo gained over the
        last <=3 fights, a momentum/decline signal).
      current: indexed by fighter, the same three features *after* their
        latest fight — the state carried into their next bout.

    Draws on the scorecards count 0.5; no-contests/overturned results leave
    ratings untouched but still count toward strength of schedule.
    """
    elo: dict[str, float] = {}
    traj: dict[str, list[float]] = {}      # rating before fight 0, 1, 2, ...
    opp_hist: dict[str, list[float]] = {}  # opponents' pre-fight ratings

    def pre_feats(name):
        e = elo.get(name, ELO_START)
        opps = opp_hist.get(name, [])
        t = traj.get(name, [])
        sos = float(np.mean(opps)) if opps else np.nan
        change3 = e - t[max(0, len(t) - 3)] if t else np.nan
        return e, sos, change3

    cols = {f'{s}_{c}': np.empty(len(fights))
            for s in ('F1', 'F2') for c in ('elo', 'opp_elo_avg', 'elo_change3')}

    for i, (idx, row) in enumerate(fights.iterrows()):
        f1, f2, winner = row['Fighter_1'], row['Fighter_2'], row['Winner']
        e1, sos1, ch1 = pre_feats(f1)
        e2, sos2, ch2 = pre_feats(f2)
        cols['F1_elo'][i], cols['F1_opp_elo_avg'][i], cols['F1_elo_change3'][i] = e1, sos1, ch1
        cols['F2_elo'][i], cols['F2_opp_elo_avg'][i], cols['F2_elo_change3'][i] = e2, sos2, ch2

        method = _method_bucket(row['Method'])
        if winner == f1:
            s1 = 1.0
        elif winner == f2:
            s1 = 0.0
        elif method == 'dec':  # scorecard draw
            s1 = 0.5
        else:                  # NC / overturned: no rating change
            s1 = None

        k = ELO_K * (ELO_FINISH_MULT if method in ('ko', 'sub') else 1.0)
        if s1 is not None:
            exp1 = 1.0 / (1.0 + 10 ** ((e2 - e1) / 400.0))
            e1_new = e1 + k * (s1 - exp1)
            e2_new = e2 + k * ((1 - s1) - (1 - exp1))
        else:
            e1_new, e2_new = e1, e2

        for name, old, new, opp_e in ((f1, e1, e1_new, e2), (f2, e2, e2_new, e1)):
            traj.setdefault(name, []).append(old)
            opp_hist.setdefault(name, []).append(opp_e)
            elo[name] = new

    per_fight = pd.DataFrame(cols, index=fights.index)

    current = pd.DataFrame.from_dict(
        {name: (elo[name],
                float(np.mean(opp_hist[name])),
                elo[name] - traj[name][max(0, len(traj[name]) - 2)])
         for name in elo},
        orient='index', columns=['elo', 'opp_elo_avg', 'elo_change3'])
    current.index.name = 'fighter'
    return per_fight, current


def _elo_long(per_fight: pd.DataFrame) -> pd.DataFrame:
    """Reshape compute_elo per-fight output to one row per (fight, side).

    Also exposes opp_pre_elo: the opponent's pre-fight Elo for this row's
    fighter, used to weight career-stat accumulation by opponent quality.
    """
    parts = []
    for side, opp in (('F1', 'F2'), ('F2', 'F1')):
        parts.append(pd.DataFrame({
            'fight_idx': per_fight.index,
            'side': side,
            'elo': per_fight[f'{side}_elo'],
            'opp_elo_avg': per_fight[f'{side}_opp_elo_avg'],
            'elo_change3': per_fight[f'{side}_elo_change3'],
            'opp_pre_elo': per_fight[f'{opp}_elo'],
        }))
    return pd.concat(parts, ignore_index=True)


def _long_format(fights: pd.DataFrame) -> pd.DataFrame:
    """One row per (fight, fighter) with that fighter's stats and outcome."""
    rows = []
    for side, opp in (('F1', 'F2'), ('F2', 'F1')):
        name_col = 'Fighter_1' if side == 'F1' else 'Fighter_2'
        opp_col = 'Fighter_2' if side == 'F1' else 'Fighter_1'
        d = pd.DataFrame({
            'fight_idx': fights.index,
            'date': fights['Event_Date'],
            'fighter': fights[name_col],
            'side': side,
            'won': (fights['Winner'] == fights[name_col]).astype(int),
            'lost': (fights['Winner'] == fights[opp_col]).astype(int),
            'method': fights['Method'].map(_method_bucket),
            'time_sec': fights['Total_Fight_Time_Sec'],
            'kd': fights[f'{side}_KD'],
            'opp_kd': fights[f'{opp}_KD'],
            'sig_l': fights[f'{side}_Sig_Landed'],
            'sig_a': fights[f'{side}_Sig_Att'],
            'opp_sig_l': fights[f'{opp}_Sig_Landed'],
            'opp_sig_a': fights[f'{opp}_Sig_Att'],
            'td_l': fights[f'{side}_TD_Landed'],
            'td_a': fights[f'{side}_TD_Att'],
            'opp_td_l': fights[f'{opp}_TD_Landed'],
            'opp_td_a': fights[f'{opp}_TD_Att'],
            'sub_att': fights[f'{side}_Sub_Att'],
            'ctrl': fights[f'{side}_Ctrl_Sec'],
        })
        rows.append(d)
    long = pd.concat(rows, ignore_index=True)

    long['ko_win'] = ((long['method'] == 'ko') & (long['won'] == 1)).astype(int)
    long['sub_win'] = ((long['method'] == 'sub') & (long['won'] == 1)).astype(int)
    long['dec_win'] = ((long['method'] == 'dec') & (long['won'] == 1)).astype(int)
    long['ko_loss'] = ((long['method'] == 'ko') & (long['lost'] == 1)).astype(int)
    long['sub_loss'] = ((long['method'] == 'sub') & (long['lost'] == 1)).astype(int)

    # Chronological order; fight_idx breaks ties within an event night
    # (early UFC tournaments had fighters compete multiple times per night).
    return long.sort_values(['fighter', 'date', 'fight_idx'], kind='stable')


def _derive_career(d: pd.DataFrame) -> pd.DataFrame:
    """Turn cumulative totals (cum_*) into rate/ratio career features."""
    out = pd.DataFrame(index=d.index)
    n = d['cum_n']  # opponent-quality-weighted fight count
    minutes = d['cum_time_sec'] / 60.0

    def ratio(num, den):
        return np.where(den > 0, num / den, np.nan)

    # Raw fight count where available (train.py's debut filter and d_n_fights
    # need actual counts, not weighted ones).
    out['n_fights'] = d['cum_n_raw'] if 'cum_n_raw' in d.columns else n
    out['win_rate'] = ratio(d['cum_won'], n)
    out['slpm'] = ratio(d['cum_sig_l'], minutes)
    out['str_acc'] = ratio(d['cum_sig_l'], d['cum_sig_a'])
    out['sapm'] = ratio(d['cum_opp_sig_l'], minutes)
    out['str_def'] = 1 - ratio(d['cum_opp_sig_l'], d['cum_opp_sig_a'])
    out['td_per15'] = ratio(d['cum_td_l'] * 15, minutes)
    out['td_acc'] = ratio(d['cum_td_l'], d['cum_td_a'])
    out['td_def'] = 1 - ratio(d['cum_opp_td_l'], d['cum_opp_td_a'])
    out['opp_td_per15'] = ratio(d['cum_opp_td_l'] * 15, minutes)
    out['sub_per15'] = ratio(d['cum_sub_att'] * 15, minutes)
    out['kd_per15'] = ratio(d['cum_kd'] * 15, minutes)
    out['kd_abs_per15'] = ratio(d['cum_opp_kd'] * 15, minutes)
    out['ctrl_pct'] = ratio(d['cum_ctrl'], d['cum_time_sec'])
    out['ko_win_pct'] = ratio(d['cum_ko_win'], n)
    out['sub_win_pct'] = ratio(d['cum_sub_win'], n)
    out['dec_win_pct'] = ratio(d['cum_dec_win'], n)
    out['ko_loss_pct'] = ratio(d['cum_ko_loss'], n)
    out['sub_loss_pct'] = ratio(d['cum_sub_loss'], n)
    out['avg_fight_min'] = ratio(minutes, n)
    return out


def _streaks(won: np.ndarray) -> np.ndarray:
    """Streak entering each fight: +k after k straight wins, -k after k losses."""
    out = np.zeros(len(won))
    s = 0
    for i, w in enumerate(won):
        out[i] = s
        if w == 1:
            s = s + 1 if s >= 0 else 1
        else:
            s = s - 1 if s <= 0 else -1
    return out


def _opp_weights(long: pd.DataFrame) -> np.ndarray:
    """Per-row opponent-quality weights (opp_pre_elo / ELO_START) ** k."""
    opp_elo = long['opp_pre_elo'].fillna(ELO_START).to_numpy()
    return (opp_elo / ELO_START) ** ELO_WEIGHT_K


def _career_cumsums(long: pd.DataFrame, w: np.ndarray,
                    include_current: bool = False) -> pd.DataFrame:
    """Career totals per (fight, fighter) row, weighted and time-decayed.

    Each fight contributes its stats scaled by w (opponent-quality weight).
    Rate/volume columns additionally decay by 0.5 ** (years_to_current_fight
    / DECAY_HALF_LIFE), so stale performances fade; implemented incrementally
    by decaying the running sums by the time elapsed since the fighter's
    previous fight. Record columns (_UNDECAYED_COLS) and the weighted fight
    count cum_n stay undecayed so win_rate and finish percentages remain
    unbiased weighted shares. cum_n_raw is the plain fight count.

    Rows hold strictly-prior totals (point-in-time training features);
    include_current=True instead includes the row's own fight — the
    post-fight state used for the serving snapshot.
    """
    decay_cols = [c for c in CUM_COLS if c not in _UNDECAYED_COLS]
    plain_cols = [c for c in CUM_COLS if c in _UNDECAYED_COLS]
    secs_per_year = 365.25 * 24 * 3600

    n_rows = len(long)
    decay_out = np.zeros((n_rows, len(decay_cols)))
    plain_out = np.zeros((n_rows, len(plain_cols)))
    w_out = np.zeros(n_rows)
    n_out = np.zeros(n_rows)

    row_of = pd.Series(np.arange(n_rows), index=long.index)
    for _, grp in long.groupby('fighter', sort=False):
        rows = row_of.loc[grp.index].to_numpy()
        ts = grp['date'].to_numpy('datetime64[s]').astype(np.int64)
        gw = w[rows]
        d_vals = grp[decay_cols].to_numpy(dtype=float) * gw[:, None]
        p_vals = grp[plain_cols].to_numpy(dtype=float) * gw[:, None]

        d_run = np.zeros(len(decay_cols))
        p_run = np.zeros(len(plain_cols))
        w_run = 0.0
        prev_ts = None
        for i, r in enumerate(rows):
            if prev_ts is not None and DECAY_HALF_LIFE is not None:
                dt_years = (ts[i] - prev_ts) / secs_per_year
                d_run = d_run * 0.5 ** (dt_years / DECAY_HALF_LIFE)
            if not include_current:
                decay_out[r], plain_out[r] = d_run, p_run
                w_out[r], n_out[r] = w_run, i
            d_run = d_run + d_vals[i]
            p_run = p_run + p_vals[i]
            w_run += gw[i]
            if include_current:
                decay_out[r], plain_out[r] = d_run, p_run
                w_out[r], n_out[r] = w_run, i + 1
            prev_ts = ts[i]

    return pd.DataFrame(
        np.column_stack([decay_out, plain_out, w_out, n_out]),
        index=long.index,
        columns=[f'cum_{c}' for c in decay_cols + plain_cols] + ['cum_n', 'cum_n_raw'])


def build_fighter_history(fights: pd.DataFrame, fighters: pd.DataFrame) -> pd.DataFrame:
    """Per (fight, fighter) row: career features as of *before* that fight.

    Career stats accumulate opponent-quality weighted (ELO_WEIGHT_K) and
    time-decayed (DECAY_HALF_LIFE); see _career_cumsums.
    """
    per_fight_elo, _ = compute_elo(fights)
    elo_long = _elo_long(per_fight_elo)

    long = _long_format(fights)
    long = long.merge(elo_long[['fight_idx', 'side', 'opp_pre_elo']],
                      on=['fight_idx', 'side'], how='left')
    g = long.groupby('fighter', sort=False)

    cums = _career_cumsums(long, _opp_weights(long))

    hist = pd.concat([long[['fight_idx', 'date', 'fighter', 'side', 'won']], cums], axis=1)
    feat = _derive_career(hist)
    feat['streak'] = g['won'].transform(lambda s: pd.Series(_streaks(s.values), index=s.index))
    feat['recent3_win_rate'] = g['won'].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    prev_date = g['date'].shift(1)
    feat['layoff_days'] = (long['date'] - prev_date).dt.days

    hist = pd.concat([hist[['fight_idx', 'date', 'fighter', 'side']], feat], axis=1)
    hist = hist.merge(
        elo_long[['fight_idx', 'side', 'elo', 'opp_elo_avg', 'elo_change3']],
        on=['fight_idx', 'side'], how='left')
    hist = hist.merge(
        fighters[['Fighter_Name', 'height_cm', 'reach_cm', 'weight_kg', 'southpaw', 'DOB']],
        left_on='fighter', right_on='Fighter_Name', how='left')
    hist['age'] = (hist['date'] - hist['DOB']).dt.days / 365.25
    return hist.drop(columns=['Fighter_Name', 'DOB'])


def build_fight_matrix(fights: pd.DataFrame, fighters: pd.DataFrame,
                       seed: int = 42) -> pd.DataFrame:
    """Model-ready matrix: one row per decisive fight.

    Fighter_1 wins 64% of fights in the raw data (ufcstats ordering bias), so
    each fight's (A, B) orientation is randomized; target y = 1 iff A won.
    Features are A-minus-B differences plus symmetric context columns.
    """
    hist = build_fighter_history(fights, fighters)

    f1 = hist[hist['side'] == 'F1'].set_index('fight_idx')
    f2 = hist[hist['side'] == 'F2'].set_index('fight_idx')

    decisive = fights[
        (fights['Winner'] == fights['Fighter_1']) |
        (fights['Winner'] == fights['Fighter_2'])
    ]
    idx = decisive.index

    rng = np.random.default_rng(seed)
    swap = rng.random(len(idx)) < 0.5

    a = pd.DataFrame(
        np.where(swap[:, None], f2.loc[idx, FIGHTER_FEATURES], f1.loc[idx, FIGHTER_FEATURES]),
        index=idx, columns=FIGHTER_FEATURES)
    b = pd.DataFrame(
        np.where(swap[:, None], f1.loc[idx, FIGHTER_FEATURES], f2.loc[idx, FIGHTER_FEATURES]),
        index=idx, columns=FIGHTER_FEATURES)

    a_name = np.where(swap, decisive['Fighter_2'], decisive['Fighter_1'])
    y = (decisive['Winner'].values == a_name).astype(int)

    # Weight-class limit (kg) per fight — used by the aging interactions.
    wc_kg = decisive['Weight_Class'].map(_weight_class_kg).values

    X = build_matchup_features(a, b, wc_kg=wc_kg)
    X['y'] = y
    X['date'] = decisive['Event_Date'].values
    X['fighter_a'] = a_name
    X['fighter_b'] = np.where(swap, decisive['Fighter_1'], decisive['Fighter_2'])
    return X


def build_matchup_features(a: pd.DataFrame, b: pd.DataFrame,
                           wc_kg: np.ndarray | None = None) -> pd.DataFrame:
    """Difference + symmetric-context features for matchups A vs B.

    `a` and `b` must each contain FIGHTER_FEATURES columns, aligned by position.
    Used both in training and by the UI at prediction time.

    `wc_kg` is the bout's weight-class limit in kg (one value per row). When
    provided (training), it drives the weight-class-aware aging interaction;
    when absent (UI prediction path) or NaN (catch/open weight) the fighters'
    mean listed weight is used as a fallback.
    """
    X = pd.DataFrame(index=a.index if isinstance(a, pd.DataFrame) else None)
    for c in FIGHTER_FEATURES:
        X[f'd_{c}'] = a[c].values - b[c].values
    # Symmetric context: division proxy, combined experience, stance matchup.
    X['mean_weight_kg'] = (a['weight_kg'].values + b['weight_kg'].values) / 2
    X['total_fights'] = a['n_fights'].values + b['n_fights'].values
    X['mean_age'] = (a['age'].values + b['age'].values) / 2
    X['open_stance'] = np.where(
        np.isnan(a['southpaw'].values) | np.isnan(b['southpaw'].values),
        np.nan, (a['southpaw'].values != b['southpaw'].values).astype(float))

    # Weight-class-specific aging: heavier divisions are expected to punish
    # age gaps more. Scaled by 100 to keep magnitudes near other features.
    if wc_kg is not None:
        division_kg = np.where(np.isnan(wc_kg), X['mean_weight_kg'].values, wc_kg)
    else:
        division_kg = X['mean_weight_kg'].values
    X['d_age_x_wc'] = X['d_age'].values * division_kg / 100.0

    # Difference of squared ages (= d_age * 2*mean_age): non-linear,
    # accelerating physical decline with age.
    age_a = X['mean_age'].values + X['d_age'].values / 2.0
    age_b = X['mean_age'].values - X['d_age'].values / 2.0
    X['d_age_sq'] = age_a ** 2 - age_b ** 2

    return X


MODEL_FEATURES = [f'd_{c}' for c in FIGHTER_FEATURES] + [
    'mean_weight_kg', 'total_fights', 'mean_age', 'open_stance',
    'd_age_x_wc', 'd_age_sq',
]


def build_current_snapshot(fights: pd.DataFrame, fighters: pd.DataFrame,
                           as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Each fighter's career features *including* all fights to date.

    This is what the UI uses: the state a fighter would carry into their
    next fight. `layoff_days` and `age` are computed relative to `as_of`.
    """
    as_of = pd.Timestamp(as_of) if as_of is not None else fights['Event_Date'].max()
    per_fight_elo, current_elo = compute_elo(fights)
    elo_long = _elo_long(per_fight_elo)

    long = _long_format(fights)
    long = long.merge(elo_long[['fight_idx', 'side', 'opp_pre_elo']],
                      on=['fight_idx', 'side'], how='left')
    g = long.groupby('fighter', sort=False)

    # Same weighted/decayed accumulation as training, including each
    # fighter's latest fight (their state going into the next one).
    cums = _career_cumsums(long, _opp_weights(long), include_current=True)

    snap = pd.concat([long[['fighter', 'date', 'won']], cums], axis=1)
    feat = _derive_career(snap)

    # Streak *after* each fight = streak entering the next one.
    def post_streak(won):
        s = _streaks(np.append(won, 0))
        return pd.Series(s[1:], index=won.index)

    feat['streak'] = g['won'].transform(lambda s: post_streak(s))
    feat['recent3_win_rate'] = g['won'].transform(
        lambda s: s.rolling(3, min_periods=1).mean())

    snap = pd.concat([snap[['fighter', 'date']], feat], axis=1)
    snap = snap.groupby('fighter').tail(1).set_index('fighter')
    snap['layoff_days'] = (as_of - snap['date']).dt.days

    snap = snap.join(current_elo)

    fighters_idx = fighters.set_index('Fighter_Name')
    snap = snap.join(fighters_idx[['height_cm', 'reach_cm', 'weight_kg', 'southpaw',
                                   'DOB', 'Stance', 'Wins', 'Losses', 'Draws']])
    snap['age'] = (as_of - snap['DOB']).dt.days / 365.25
    snap['last_fight'] = snap['date']
    return snap.drop(columns=['date'])
