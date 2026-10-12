# Reviewer feedback (mock Stage 2 judging)

Reviewed on 2026-10-12 against the Stage 2 brief. What I looked at: all code in `training/`, `app/server.py`, `tests/`, `exploration/05_feature_ideas.py`, `models/metrics.json`, README, CLAUDE.md, and both conversation logs. I ran `uv run pytest`: 4/4 passed in about 35 s. I did not launch the dashboard in a browser.

## Verdict in one paragraph

This is a technically strong submission. The leakage handling is better than most candidates will manage: `truncate_at` plus an equality test, a negative control, and a mutation test showing the test catches an injected leak. Feature selection is done on two validation windows with day-block bootstrap intervals, and the test set is scored once. The weak side is the **process signal**, which the brief calls the *primary* criterion. In the logs the AI does most of the critical thinking. It catches its own Luhansk mistake, mutation-tests its own test, and rejects the candidate's sin/cos, weather and "+5 min" ideas with evidence. The candidate mostly acts as product owner and asks good questions, but there are almost no moments where *they* catch an AI error. There is also one AI overclaim that went unchallenged (below). Presentation has gaps too: no commits at all, no screenshot, and a stale README section.

## Strengths

1. **Leakage discipline is real and tested, not just claimed.**
   - `truncate_at(alerts, t)` defines "knowable at t" in one place.
   - `tests/test_no_leakage.py` recomputes features on truncated data, and it includes alert start and end instants as edge cases.
   - The negative control proves the harness can detect look-ahead. The log also shows a mutation test with an injected 1 h leak (history-1, around line 1493).
2. **Honest data work.**
   - The official CSV stops on 2026-09-07. This was found, documented, and the volunteer feed was used instead so the test period could be the latest one (14 Sep – 12 Oct).
   - The raion-level structural break and the Luhansk/Crimea permanent alerts are explained.
3. **Sound evaluation protocol.** Train → 14-day validation (early stopping) → refit → 28-day test scored once.
4. **Rigorous feature selection.**
   - `exploration/05_feature_ideas.py` repeats each test on a second window and uses a day-block bootstrap, because alerts come in waves.
   - It measures seed noise (≈0.0001) against day-to-day noise (≈0.001–0.002).
   - A feature is adopted only if it wins in both windows. This is a mature standard for a one-day project.
5. **Negative results are kept and documented.** Weather at oblast centroids and at drone launch sites, sin/cos encoding, and the "+5 min" variant all have scripts and numbers. That is strong material for the Stage 3 interview.
6. **Sensible model choices for the compute budget.**
   - Start model: LightGBM with calibrated probabilities. The calibration table is good in the middle bins.
   - Duration model: a log-normal survival model, with closed-form quantiles conditioned on elapsed time. The closed form is checked against lifelines at train time.
   - Survival analysis is the right framing for "how long will it last".
7. **Working product.** The dashboard reuses `training.features`, so there is no train/serve skew. Trained artifacts are committed, so judges don't need to retrain. Training takes about 20 s.
8. **The candidate's own ideas drove iterations.** Neighbour features, weather, previous duration and "+5 min" all came from the candidate and were then tested.

## Weaknesses (most important first)

### 1. The process log shows little candidate-driven error correction
The brief asks for candidates who "challenge model hallucinations, correct errors". Most correction moments in the logs are the **AI correcting itself**:
- the Luhansk claim in CLAUDE.md;
- the mutation test of its own leakage test;
- the OOM kill;
- `pkill` killing its own shell.

The candidate's challenges are mostly *questions* ("prove me wrong" about LSTM, "what is BFGS"). Those show curiosity, but a judge can't credit the candidate with steering the result. The one factual mix-up in log 2 (LSTM vs LightGBM) was the candidate's, and the AI corrected it.

**Unchallenged AI overclaim.** At the end of history-2 the AI says *"The duration model is now clearly better than the per-oblast median baseline."* The numbers are an MAE of 18.8 vs 19.5 min (−0.7 min, no confidence interval). That is not "clearly", and the README itself says "barely better". A judge reading the log will notice that the candidate let this pass.

### 2. The start-model baseline is weak, so the headline gain is overstated
I checked this in a scratchpad script; the repo is unchanged. Same split, calm hours only, test window:

| Model | Log loss ↓ | ROC-AUC ↑ | PR-AUC ↑ |
|---|---|---|---|
| README baseline (train-period rate per oblast × hour) | 0.525 | 0.730 | 0.479 |
| Logistic regression on **one feature**, `log1p(starts_7d)` | 0.466 | 0.764 | 0.508 |
| Logistic regression, 5 recency features + hour | 0.463 | 0.768 | 0.515 |
| LightGBM (submitted) | **0.455** | **0.784** | **0.529** |

About 85% of the log-loss gain over the stated baseline comes from "how many alerts did this oblast have in the last week". The reason is drift: the positive rate is **16.2%** in train+valid but **24.5%** in test. A rate learned over a whole year is stale, and any recency feature fixes most of that.

The model is still the best of the four. But the honest gain over a sensible baseline is about 0.01 log loss, not 0.07. Interviewers may well ask about this. It's better to report it yourself.

### 3. The training window was never validated
`TRAIN_START = 2025-10-01` is commented "older periods behave differently (see CLAUDE.md)", but CLAUDE.md says nothing about it. No experiment compares training windows of 3, 6, 12 or 24 months. Given the drift above, this is probably the most important untested hyperparameter.

### 4. Calibration under drift
The test calibration table under-predicts in the busiest bins: predicted 0.15 → observed 0.195, and predicted 0.03 → observed 0.043. That matches the base-rate jump. It isn't discussed anywhere.

### 5. Loose ends on the duration model
- Weibull beats log-normal on validation (−4.295 vs −4.338 log-likelihood), but log-normal ships. "The prediction code only supports log-normal" is a tooling reason, not a modelling one. Weibull's closed form is equally simple.
- The c-index is 0.62. The point forecast is close to the baseline, so the value of this model is in its calibrated intervals (81% coverage at the 80% level). Lead with that, not with MAE.
- Alerts with `naive=True` (unknown end) are dropped from training *and* test. Those alerts are not random, so this introduces selection bias. At least state it.

### 6. "Now" mode inconsistencies (`app/server.py`, `/api/now`)
- The live feed overrides `active_now` and `n_active_oblasts`, but **`nb_active_frac` and `nb_starts_1h` still come from the stale daily dump**. So the neighbour features disagree with the map shown on the same page.
- Live alerts are treated as starting now (`elapsed = 0`). The expected end is therefore systematically too late for alerts that have already been running. The README says the curve isn't conditioned, but not that the median is biased.

### 7. The "time series analysis" part is thin
The project is entirely forecasting. There is no descriptive time-series analysis anywhere in the repo:
- daily/weekly counts and trend;
- hour × weekday heatmap;
- autocorrelation or clustering of starts;
- a figure of the structural break or of the drift.

The brief literally says "Time Series Analysis". One notebook or script with 4–5 figures would cover this and make the README much easier to read.

### 8. The optional text/Telegram data was scoped out
Telegram was dropped early, which was a reasonable call for a one-day MVP. But CLAUDE.md already found that `t.me/s/kpszsu` is scrapeable and that posts carry timestamps. Even a simple feature like "Air Force posts mentioning drones/missiles in the last hour" would have touched the optional track and given a natural "text helped / didn't help" result.

## Presentation issues

1. **The repo has zero commits** (`git log`: "does not have any commits yet"), and the branch is `master` while the expected main branch is `main`. Reviewers who open GitHub will see one commit at most. They lose the timeline that would back up the iteration story. Commit now in logical chunks if you can (data/features → models → app → feature study → docs), and push to `main`.
2. **There is no screenshot or GIF of the dashboard in the README.** Many judges won't run `uv sync`. One image of the dark ops map makes the product visible in 5 seconds.
3. **The README "Layout" section is stale.** It says `exploration/` holds "initial data profiling", but it now contains five studies, including the key feature-selection evidence.
4. **The conversation logs are hard to audit.**
   - `history-1` was restored after a fork, and its header says so. That's honest, but add the raw session files (`~/.claude/projects/.../*.jsonl`) or explain exactly how the log was restored, so authenticity can't be questioned.
   - `history-2` is a terminal copy in which most tool outputs are collapsed ("ctrl+o to expand"). A reviewer can't see the numbers the AI was looking at.
   - Add a short `claude_conversation/README.md` that indexes the key moments with line numbers. Judges skim, so point them to: leakage mutation test (history-1 ~1493); Luhansk self-correction (history-1 ~2811–2843); weather rejection (history-2 ~170–470); the user's feature ideas and their outcome (history-2 ~591–end).
5. **The README results table has no uncertainty.** Add a day-block bootstrap CI on the test deltas, since the script already exists for validation. Also add the stronger recency baseline.
6. **Small hygiene issues.**
   - The test suite downloads ~9 MB from GitHub on its first run. Mention this in the README.
   - There are NumPy deprecation warnings in the tests (`'generic' unit for timedelta`).
   - `.env` exists locally. It is gitignored, and I confirmed the W&B key does not appear in either log. Keep it that way when committing.

## How to improve, by priority (given the deadline is tonight)

**Must do (≈30 min)**
1. Commit in a few logical commits, push to `main`, and check that a fresh clone runs: `uv sync && uv run pytest && uv run python -m app`.
2. Add a dashboard screenshot to the README and fix the Layout section.
3. Write the ~100-word reflection around a *real* course correction. For example: "My baseline was too weak: a one-feature recency model gets 85% of the gain. I added it, and here's what is left." Or: "the AI called a 0.7-minute gain 'clearly better' and I pushed back." Real errors you caught are worth more than a polished narrative.

**Should do (≈1–2 h), in a new logged session where *you* lead the critique**
4. Ask the AI to add the recency logistic-regression baseline to `train.py` and the README. Challenge the "clearly better" wording and fix it in the log.
5. Run a training-window ablation (3/6/12/24 months, validation only) and either justify `TRAIN_START` or change it.
6. Add bootstrap CIs to the test metrics.
7. Fix the `nb_active_frac` live override in Now mode.

**Nice to have**
8. One EDA script or notebook with figures: trend, hour × weekday heatmap, base-rate drift, structural break.
9. Switch the duration model to Weibull, or explain with numbers why log-normal is kept.
10. Add a self-exciting point-process baseline (Hawkes). Alerts cluster in waves, so it's the textbook model for this data and a strong Stage 3 talking point.
11. Run a minimal Telegram (`kpszsu`) count feature through the existing two-window evaluation, even if it's rejected.

## Questions to prepare for Stage 3

- How much of your model's gain is just "recent activity"? What would a 7-day-rate baseline score?
- Why does training start on 2025-10-01?
- The test positive rate is 24.5% vs 16% in training. How does your model handle drift, and how would you retrain in production?
- Your logs show the AI catching its own mistakes. Where did *you* catch the AI being wrong?
- Why log-normal when Weibull won on validation?
- Is the volunteer (eTryvoga) dataset ever revised after the fact? If so, could revisions leak future information into historical features?
- How would Now mode behave when the daily dump is 20 hours behind?

## Scorecard (subjective, out of 5)

| Criterion | Score | Note |
|---|---|---|
| No leakage / evaluation rigour | 5 | Best part of the project |
| Uses the latest data | 5 | Handled the stalled official file correctly |
| Modelling quality | 3.5 | Good, but the baseline is weak and the training window untested |
| Working code / reproducibility | 4 | Tests pass and artifacts are committed; no git history |
| Optional text/media | 1 | Explored, not implemented |
| **Process: candidate steering and challenging the AI** | **3** | Good questions and ideas, but few errors caught by the candidate |
| Presentation | 3 | Clear README; no visuals, stale section, logs hard to audit |
