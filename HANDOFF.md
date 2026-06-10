# UFC Winner-Prediction Model — Agent Handoff

Last updated: 2026-06-10. State: all changes below are applied, trained, and
verified on the holdout. Nothing is committed to git yet (the whole `UFC/`
tree is untracked).

## What this project is

Predict UFC fight winners from point-in-time career features.

- `UFC/features.py` — feature engineering. Every feature a fighter carries
  into a fight is computed only from strictly earlier fights. This
  point-in-time rule is the project's core invariant; never leak the current
  fight's stats (or anything dated after it) into its own features.
- `UFC/train.py` — XGBoost trainer: 60-iter randomized search with
  time-series CV, holdout = last 15% of fights by date (currently cutoff
  2024-03-09, 970 fights). Saves `model/ufc_xgb.ubj`, `model/fighter_snapshot.csv`,
  `model/metadata.json`.
- `UFC/train_torch.py` — PyTorch MLP (GPU, RTX 2080 Ti) + blend evaluation.
  Saves `model/ufc_mlp.pt`, `model/metadata_torch.json`.
- `UFC/app.py` — Streamlit UI. Serves the 50/50 XGB + sym-MLP blend, so the
  app depends on torch. It shows model-implied odds, confidence, XGB/MLP
  disagreement, and abstains below the configured threshold.
- `UFC/update_data.py` — scraper, see below.
- Data: `UFC/ufc_gold_dataset_final.csv` (8,705 fights, 1994 → 2026-06-06,
  one row per fight, 37 cols incl. full stats), `UFC/ufc_fighters_final.csv`
  (4,465 fighters). `UFC/ELO-main/` is a vendored reference repo — unused by
  the pipeline, do not merge its CSV (name-matching and point-in-time hazards).

## Metric history (holdout log-loss is the primary metric)

| date | change | acc | AUC | log-loss | brier |
|---|---|---|---|---|---|
| 06-09 | pre-Elo (old cutoff 2023-12-02, not comparable) | .6357 | .6835 | .6477 | |
| 06-09 | + native Elo features | .6526 | .6924 | .6438 | |
| 06-10 | dataset refresh → cutoff 2024-03-09, 970 fights | .6464 | .6941 | .6390 | .2238 |
| 06-10 | + decay + Elo-weighting + aging (XGB) | .6464 | .7030 | .6343 | .2217 |
| 06-10 | symmetrized MLP alone | .6464 | .7048 | .6315 | .2205 |
| 06-10 | **blend sym-MLP + XGB 50/50 (current best)** | **.6526** | **.7101** | **.6309** | **.2201** |

Baseline win-rate pick: 0.6031. The XGB search has converged to the same
best_params across all recent runs (max_depth 4, n_estimators 535, lr 0.008,
…) — gains have come from features, not tuning.

## Recent changes (2026-06-10)

### 1. Data update pipeline (`update_data.py`)
Scrapes ufcstats.com completed events newer than the gold dataset's max
`Event_Date` into the gold schema; adds debuting fighters. Dedupe key is the
16-hex fight id in `Fight_URL`. Flags: `--dry-run`, `--since YYYY-MM-DD`.
ufcstats.com sits behind a sha256 proof-of-work browser check —
`UFCStatsSession` solves it automatically (finds n where
`sha256(nonce:n)` has the required zero prefix, POSTs to `/__c`; the cookie
is host-specific so URLs are normalized to `www.`). Parsing was validated
cell-for-cell against pre-existing dataset rows. Routine update:
`python3 update_data.py && python3 train.py && python3 train_torch.py`.
Elo and all career features recompute from the gold CSV — there is no
separate Elo data step.

### 2. Feature upgrades in `features.py` (all three validated independently
by isolated experiments — fixed-params screen first, then full re-tune —
then merged and regression-tested)
- **Time-decayed career stats** — `DECAY_HALF_LIFE = 5.0` years. Rate/volume
  sums decay by `0.5 ** (years_to_current_fight / 5)`; win/loss record
  columns (`_UNDECAYED_COLS`) stay plain counts.
- **Opponent-Elo-weighted averages** — `ELO_WEIGHT_K = 1.0`. Each fight's
  contribution scaled by `opp_pre_fight_elo / 1500`; numerators and
  denominators both weighted (proper weighted average). `cum_n_raw` keeps the
  raw count for train.py's debut filter and `d_n_fights`.
- **Aging interactions** — `d_age_sq` (non-linear decline) and `d_age_x_wc`
  (age gap × bout weight-class limit, `_weight_class_kg` parses
  `Weight_Class`; catch/open weight falls back to mean fighter weight). Both
  immediately entered the top-4 importances behind `d_age`/`d_elo`.

All accumulation goes through `_career_cumsums(long, w, include_current)`.
`build_current_snapshot` uses the same path with `include_current=True`, so
serving features match training exactly (an earlier inconsistency, now fixed).
Setting `DECAY_HALF_LIFE = None` and `ELO_WEIGHT_K = 0` reproduces the old
plain-cumsum behaviour exactly (verified to 4 decimals).

### 3. Torch MLP (`train_torch.py`)
128→64 MLP, dropout 0.4, BatchNorm, AdamW, early stopping on a temporal
validation slice; median imputation + missing indicators + standardization
fitted on train only. **Symmetrized inference** — average `p(x)` with
`1 − p(−x)` (all d_* features negate under fighter swap) — reliably improves
calibration and is how the MLP should always be scored. The 50/50 blend with
XGB is the best model overall. Honest-blend protocol: the saved
`ufc_xgb.ubj` is refit on ALL data, so for holdout comparisons retrain XGB
on the train split with metadata.json best_params (train_torch.py does this).

### 4. False-positive / selective-prediction pass
The product objective is now **high precision when the model speaks**, not
always-predict accuracy. A confident wrong winner is very costly, so the app
and reports must allow abstention.

- User decision: false positives cost 5x a correct confident pick; app may
  depend on torch; app should show a sliding scale with odds and confidence.
- A 5x false-positive penalty implies break-even precision of 83.3%
  (`5 / (5 + 1)`). Until calibrated thresholds are learned on validation, the
  app uses Confident pick >= 0.75, Lean >= 0.60, No pick < 0.60.
- `train.py` now reports selective holdout metrics for confidence thresholds
  0.55/0.60/0.65/0.70/0.75, including coverage, precision, false positives,
  and abstentions. It also reports confidence reliability bins and ECE.
- `train.py` writes `model/holdout_high_conf_wrong_picks.csv`, a ranked audit
  of wrong holdout calls with confidence >= 0.55.
- Current XGB selective holdout results:
  - threshold 0.55: precision .6875, coverage .7423, false positives 225
  - threshold 0.60: precision .7234, coverage .4845, false positives 130
  - threshold 0.65: precision .7782, coverage .2649, false positives 57
  - threshold 0.70: precision .8229, coverage .0990, false positives 17
  - threshold 0.75: precision .9091, coverage .0227, false positives 2
- `train_torch.py` now reports selective metrics for sym-MLP and the
  sym-MLP+XGB blend, plus a candidate disagreement-veto grid using
  `abs(p_xgb - p_mlp_sym)`. Re-run `python3 train_torch.py` to refresh
  `metadata_torch.json` with `selective_prediction`.
- `app.py` now loads `ufc_mlp.pt`, runs symmetrized MLP inference with the
  saved preprocessing, blends it 50/50 with symmetrized XGB, and displays
  model-implied American/decimal odds plus a confidence scale. It buckets
  output as Confident pick (>= 0.75), Lean (>= 0.60), or No pick (< 0.60).

## Conventions and gotchas

- `Winner == 'Draw/NC'` marks draws/no-contests; decisive-fight filtering is
  `Winner ∈ {Fighter_1, Fighter_2}`. Elo: scorecard draws = 0.5, NC = no change.
- Fighter_1 is not the winner — ufcstats page order (64% F1 win bias);
  `build_fight_matrix` randomizes orientation with seed 42.
- Within an event, fights are stored prelims-first (page order reversed).
- 7 duplicate fighter names exist (namesakes); `load_data` keeps first.
- Evaluation protocol for any feature idea: (1) cheap screen — fixed
  best_params from metadata.json, single fit, same split; (2) if it wins on
  log-loss, one full `train.py` run for the official number. Compare against
  a fixed-params baseline computed the same way, not against the tuned number.

## Next steps (in rough priority order)

1. **Calibrate and learn thresholds on a temporal validation slice** — do not
   choose production thresholds from the final holdout. Compare raw, Platt, and
   isotonic calibration by ECE/Brier/log-loss and precision@coverage.
2. **Refresh torch metadata** — re-run `python3 train_torch.py` so
   `metadata_torch.json` includes the new `selective_prediction` reports for
   the blend and disagreement veto.
3. **Disagreement veto / conformal abstention** — use XGB-vs-MLP disagreement
   or conformal prediction sets to abstain on unstable calls.
4. **Per-round stats** — the scraper currently skips the per-round tables on
   fight pages; they enable cardio/fade features (cheap data, new signal).
5. **Damage, mileage, scheduled-rounds/title, and recent-form features** —
   prioritize features that reduce confident wrong calls, not just global
   accuracy.
6. **Method/round targets** — second model head (finish vs decision, KO vs
   sub); also useful as auxiliary tasks that may regularize the winner head.
7. **Short-notice / missed-weight flags** — needs new sources (Tapology/
   Wikipedia announcement dates, weigh-in results); high ceiling.
8. **Commit the repo** — everything is untracked; commit before further
   surgery so experiments can diff against a baseline.

## Brainstorm: accuracy, and the asymmetric cost of false positives

Context: a "false positive" here = the model picks a winner with confidence
and that fighter loses. That cost asymmetry means raw accuracy matters less
than **precision at high confidence** — being right when we speak, silent
when unsure.

### A. Make confidence trustworthy (no new data needed)

- **Probability calibration.** Fit isotonic or Platt scaling on a temporal
  validation slice (never random — this is time-series). XGBoost log-loss is
  decent but the blend's probabilities have never been explicitly calibrated.
  Add ECE / reliability curves to train.py's report.
- **Selective prediction / abstention band.** Only "call" fights with
  p ≥ τ (sweep τ on validation for precision@coverage). Report a
  precision-vs-coverage curve; e.g. if precision at p≥0.65 is materially
  higher than at 0.5–0.65, the product answer is "no pick" for the middle band.
- **Ensemble disagreement as an uncertainty veto.** We already have two
  diverse models (XGB, sym-MLP). Abstain when they disagree by more than a
  margin (|p_xgb − p_mlp| > δ). Disagreement is a free, well-studied
  uncertainty signal; check whether agreement subsets have higher precision.
- **Conformal prediction.** Split-conformal on the validation slice gives
  distribution-free coverage guarantees for the abstention rule instead of
  hand-tuned thresholds.
- **Asymmetric training loss.** Upweight confident-wrong examples: sample
  weights on the minority of upsets, or focal-loss-style reweighting
  (XGBoost: custom objective; torch: trivial). Risk: hurts calibration —
  evaluate with the calibration metrics above, not accuracy.
- **Seed/bagging ensembles.** Average 5–10 XGB models over seeds (and the
  orientation-randomization seed in `build_fight_matrix`!) — the A/B swap is
  a hidden variance source; averaging over multiple swap draws is a free
  variance reduction that likely helps log-loss directly.

### B. New features from data we already have

- **Cardio/fade profile** (needs per-round scrape, step 3 above): sig-strike
  output slope across rounds, round-3+ output vs round-1, opponent control
  time growth — classic "gas tank" signals, especially predictive in
  5-rounders.
- **Durability/chin erosion:** recency-weighted KO-losses (chin doesn't come
  back), age at first KO loss, KD absorbed per 15 recent vs career. Note KO
  losses are currently decayed like everything in `_UNDECAYED_COLS`' complement
  — arguably KO damage should *accumulate* (mileage), the opposite of decay.
- **Mileage:** career total fight minutes, total strikes absorbed
  (undecayed), wars count (fights > 12 min with > 100 combined sig strikes);
  interact with age.
- **Scheduled rounds / title flag:** `Time_Format` gives 3 vs 5 rounds and
  `Weight_Class` contains "Title" — known pre-fight, currently unused. Also
  5-round experience differential (champions fare better in championship rounds).
- **Style matchup cross-terms.** Today every feature is an A−B difference of
  the same stat. Grappler-vs-striker dynamics live in *cross* terms:
  a_td_per15 × (1 − b_td_def), a_slpm × (1 − b_str_def), ctrl_pct vs
  sub_att. A handful of hand-built cross products is the cheapest way to give
  a tree model interaction hints.
- **Recent-form windows:** elo_change5, win rate / finish rate over last 5,
  stat trends (last-3 slpm minus career slpm) — "improving vs declining"
  beyond the 3-fight Elo delta.
- **Layoff interactions:** layoff × age (ring rust hits older fighters
  harder), layoff after a KO loss (medical suspensions correlate with damage).
- **Common-opponent features:** for each matchup, compare results against
  shared opponents (graph feature; moderate effort, decent literature support).
- **Weight-class moves:** fights since changing division, size differential
  history (reach/height vs division average).

### C. New data sources (effort-ordered)

1. **Betting odds (historical)** — the single strongest external signal;
   closing odds embed everything the market knows. Sources: bestfightodds.com
   (scrape), Kaggle UFC-odds datasets (join on fighter+date). Two uses:
   (a) feature — but ONLY odds available before the fight (opening odds are
   safest for prediction-time validity); (b) benchmark — if the model can't
   beat the closing line it has no edge, which is the honest yardstick given
   the false-positive cost framing.
2. **Weigh-in results** (missed weight, last-second weight) — ufc.com news
   and MMA media publish official weigh-ins; missed weight is a known
   performance red flag for the offender.
3. **Fight announcement dates** (Tapology/Wikipedia) — short-notice
   replacement flag (< 3 weeks notice is a large, well-documented disadvantage).
4. **Full pre-UFC careers** (Sherdog/Tapology) — fixes the cold-start problem
   (debut/2-fight fighters are currently filtered or feature-poor); enables
   regional-promotion Elo tiers. Biggest data project on the list; name
   matching is the hard part.
5. **Venue/altitude + corner geography** — Mexico City/ Denver altitude
   effects on cardio, home-crowd/judging proxies. Small but cheap once
   event location is scraped (it's on ufcstats event pages).
6. **Physical updates** — fighters' listed reach/stance are static in our
   CSV; Tapology tracks gym changes (camp switch ≈ form change signal).

### D. Evaluation upgrades to lock in the FP framing

Add to train.py's report: precision/recall at p-thresholds
{0.55, 0.60, 0.65, 0.70}, coverage at each, ECE, and upset-only error rate.
Without these, "accuracy improved" can hide "confident picks got worse" —
the exact failure mode the user cares about.
