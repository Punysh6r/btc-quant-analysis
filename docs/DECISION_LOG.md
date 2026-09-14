# Decision Log — Bitcoin Quantitative Analysis (BTC-USD / BTC-EUR)

> Portfolio artifact generated from Engram memory (project: `mi-proyecto`).
> Purpose: show **why** each engineering decision was made, not just the code.
> Stack: Python, Streamlit 1.63, yfinance, pandas, scikit-learn (TF-IDF + LogisticRegression), free RSS.

## 1. Snapshot (2026-09-14)

- News (RSS, keyless): **25 headlines** — CoinDesk / Cointelegraph / Bitcoin Magazine
- Prices (BTC-USD): **4381 daily rows**, updated to 2026-09-14
- Labeled dataset: **16 rows, all HOLD** (24h forward-return labels, ±2% thresholds)
- Model training: **blocked by design** until 200+ rows with 2+ classes (`MIN_ROWS=200`)
- App: runs locally, News Sentiment Tier 2 verified OK
- Regime panel: **Short-term bullrun — Bull 80% / Bear 20%** on real data

Run it:

```bash
pip install -r requirements.txt
# Double-click on Windows:
ver_app.bat
# Weekly refresh (double-click):
actualizar_datos.bat
```

## 2. Decisions

### D1 — Drop the API key, switch news to free RSS

- **Context:** CryptoCompare / CoinDesk news now requires a key (401) and the key page required login + email verification. The pipeline was blocked.
- **Decision:** Build `download_news_rss.py` with the same CSV schema as `download_news.py`, sourcing CoinDesk / Cointelegraph / Bitcoin Magazine RSS.
- **Alternatives discarded:** keep chasing the key, pay for crypto-news API (x402 / bot-blocked), scrape HTML (fragile, ToS risk).
- **Result:** 25 items → 4381 prices → 16 HOLD rows. Pipeline unblocked for validation, but RSS only covers recent weeks.
- **Lesson:** free RSS validates the pipeline end-to-end, but it cannot provide BUY/SELL history. Training needs weekly accumulation. Documented as partial unblock, not a full fix.

Files: `download_news_rss.py`, `data/news_btc.csv`, `build_dataset.py`, `train_model.py`

### D2 — Make the Streamlit app 100% keyless with a 4-tier sentiment panel

- **Context:** user definitively discarded the API key. The app had to be useful with 16 HOLD rows and no trained model.
- **Decision:** `render_news_model_signal()` with 4 tiers:
  1. trained model (if `artifacts/model.joblib` exists),
  2. dataset forward-return labels,
  3. raw headlines,
  4. RSS instructions.
  `TRAIN_COMMANDS` now points to `download_news_rss.py`. Added `--append` with id-based dedupe for weekly growth.
- **Alternatives discarded:** hide the panel until the model exists (worse UX), fake probabilities (dishonest).
- **Result:** Tier 2 renders metrics + bar chart + table + ML hint with real data.
- **Lesson:** `wc -l` lies on CSVs with multiline bodies (16 real rows vs 33 physical lines). Always count with pandas, not lines.

Files: `app.py` (`render_news_model_signal`, `load_labeled_dataset`), `download_news_rss.py` (`--append`)

### D3 — One double-click weekly refresh (`actualizar_datos.bat`)

- **Context:** user asked for a non-technical weekly update.
- **Decision:** `actualizar_datos.bat`: RSS `--append` + prices + dataset + train attempt, with friendly messages and non-blocking warnings.
- **Bug fixed:** parentheses inside `echo` within `if (...)` blocks break `cmd` parsing even on the untaken branch. Removed all parentheses from messages inside blocks.
- **Result:** tested end-to-end via `cmd`: 25 news, 4381 prices, 16 HOLD, train correctly pending.
- **Lesson:** in `.bat`, never use `(` `)` in text inside `if/else` parenthesised blocks.

File: `actualizar_datos.bat`

### D4 — Verify without a browser (stub Streamlit)

- **Context:** user reported “app.py closes when I open it” (double-clicking a Streamlit script).
- **Decision:** clarify that `app.py` never opens directly — added `ver_app.bat` (`python -m streamlit run app.py`). For verification, boot Streamlit 1.63 (health OK on :8502) **plus** a stub test: inject a fake `streamlit` module into `sys.modules` and call the render function directly.
- **Result:** Tier 2 smoke test passes (16 HOLD, metrics + chart + table + hint). No leftover test server or temp script.
- **Lesson:** Streamlit health OK is not enough — the script only runs on session connect. Stubbing the module is the fastest way to test panels headlessly.

Files: `app.py`, `ver_app.bat`

### D5 — Short-term Bullrun vs Bear regime panel (technical, explainable)

- **Context:** user asked for “bullrun chances and the opposite”.
- **Choice:** simple technical regime, new panel, short-term horizon.
- **Decision:** new module `bull_bear.py`:
  `get_bull_bear_regime()` scores 0–100 from 50 neutral:
  trend SMA20/SMA50 (±30) + price vs SMA20 (±20) + RSI-14 (+15/−20) + fresh crossover (±15 or ±5 within 5d) + distance to 90-day high (+10/−20), clamped 5–95. `Bear = 100 − Bull`.
  Labels: ≥70 bullrun, 55–69 moderately bullish, 45–54 sideways, 31–44 correction, ≤30 bear.
  `render_bull_bear_regime()` shows Regime + Bull% + Bear% + progress bar + reasons + “How is this score built?” expander.
- **Alternatives discarded:** ML probability (no data), long-term SMA200 cycle (deferred, user chose short-term only), mixing news sentiment into the score (kept separate for clarity).
- **Result:** bull logic 95/5 on synthetic uptrend, bear logic 5/95 on synthetic downtrend, **80/20 on real 2026-09-14 data** (uptrend gap +9.9%, RSI 49.5 healthy, −3.5% from 90d high, price −0.1% below SMA20).
- **Lesson:** keep pure logic in its own module — testable without a browser and easy to defend in a TFM.

Files: `bull_bear.py`, `app.py` (import + render after verdict)

## 3. What is intentionally NOT done yet

- `train_model.py` still refuses to train until 200+ rows with 2+ classes. This is by design (temporal 80/20 split needs both regimes in train).
- Path forward: run `actualizar_datos.bat` weekly; RSS history accumulates via `--append` dedupe.
- Optional next step (deferred): add SMA200 long-cycle view for multi-month bull/bear context.

### D6 — Threshold bot on the bull score (hysteresis, backtest only)

- **Context:** user asked for a bot that buys/sells with a high and a low threshold, both tunable.
- **Choice:** bull/bear score 0-100 as signal, Sidebar + CLI tuning, backtest only (no real money).
- **Decision:** new module `bot.py`:
  `compute_bull_scores()` recomputes the regime score causally for every day (no lookahead),
  `backtest(df, buy, sell, capital)` buys all-in when score >= buy and sells all-out when score <= sell.
  Defaults `buy=70` (bullrun) / `sell=30` (bear), matching the regime labels. Requires `0 < sell < buy < 100`.
  CLI: `python bot.py --buy 70 --sell 30 --capital 10000 --out data/bot_trades.csv`.
  App: `render_bot_backtest(df)` adds Sidebar sliders + capital input and a panel with final value vs buy&hold, trades, win rate, max drawdown, equity chart and trades table.
- **Alternatives discarded:** RSI thresholds (kept as explanatory indicator, not bot signal), 24h-return thresholds (same as labels, less reactive), live/paper trading (out of scope for a TFM demo).
- **Result:** full history (4381 days) 70/30 gives 43 buys / 42 sells, bot +15571% vs buy&hold +17046%, max DD -72%. Last 365 days 70/30 gives 5 buys, bot +1.2% vs buy&hold -32.1% — defensive behaviour in a down year (ends long).
- **Lesson:** hysteresis (buy above, sell below) avoids whipsaws from a single threshold; causal recomputation is mandatory or the backtest lies.

Files: `bot.py`, `app.py` (import + `render_bot_backtest(df)` after regime panel), `data/bot_trades.csv`

### D7 — Regime bands on the price chart (visual wow)

- **Context:** user asked for one more eye-catching element for the hiring portfolio.
- **Choice:** bull/bear bands over the price chart (vs threshold heatmap / what-if simulator).
- **Decision:** `bull_bear.py` now exposes `simplify_regime()` (>=55 bull / <45 bear / else sideways),
  `regime_blocks()` to merge consecutive same-regime days, plus
  `add_regime_bands_plotly()` (`add_vrect`, green/gray/red at 10% opacity, below traces) and
  `add_regime_bands_matplotlib()` (`axvspan` fallback). `app.py` computes
  `compute_bull_scores(df)` once for the selected window and tints the Price and Signals chart.
- **Result:** last 365 days render 38 bands (15 bull / 13 sideways / 10 bear), 38 Plotly shapes, no lookahead.
- **Lesson:** merge into blocks instead of one shape per day — same visual impact with ~100x fewer shapes.

Files: `bull_bear.py`, `app.py` (Price and Signals section)

### D8 — Train the model and let it grade itself (demo with honest failure)

- **Context:** user asked to train the model and have it tested to see whether it hits or misses.
- **Decision:** parameterize both scripts instead of hardcoding: `build_dataset.py --threshold`
  (default 0.02) and `train_model.py --min-rows` (default 200) + `--threshold-pct` metadata.
  Production data untouched; demo runs on separate files.
- **Demo run:** `build_dataset.py --threshold 0.004 --out data/dataset_demo.csv` gives
  16 rows (6 BUY / 10 SELL / 0 HOLD). `train_model.py --data data/dataset_demo.csv
  --out artifacts/model_demo.joblib --min-rows 10 --threshold-pct 0.4` trains on the first
  12 rows and tests on the last 4 (strict temporal split, no shuffle).
- **Result — the model FAILS, and that is the point:** accuracy 25%, macro-F1 0.20,
  confusion matrix [[0, 2], [1, 1]] — 1 hit out of 4 (only one SELL caught, both BUYs missed).
  With 12 training headlines TF-IDF cannot generalize; the evaluation proves the harness works.
- **Lesson (portfolio-grade):** a failing evaluation with correct methodology beats no evaluation.
  The real 2%-threshold model unlocks after weekly `actualizar_datos.bat` runs reach 200+ rows
  with 2+ classes. Demo files: `data/dataset_demo.csv`, `artifacts/model_demo.joblib`,
  `artifacts/metrics_demo.json`.

### D9 — Plain-language indicator guide (every acronym explained)

- **Context:** user asked what each acronym means and how to read it (e.g. is RSI 46.9 good?).
- **Decision:** new module `glossary.py` with pure, tested categorizers plus
  `render_indicator_guide(df, last_row)` shown after the bot panel: a live table translating
  the current SMA trend, momentum, RSI-14, signal, bull score and 90-day distance into
  Good (green) / Neutral (gray) / Caution (yellow) / Bad (red), plus six expanders with the
  full scales (SMA, RSI, signals, bull score, bot words, news-model words incl. TF-IDF and F1).
  All bands match the rules in `app.py` / `bull_bear.py` / `bot.py`; e.g. RSI 46.9 reads as
  Healthy pullback (green, constructive).
- **Result:** helpers unit-checked (RSI zones, trend/momentum/drawdown bands, regime, F1);
  live data reads Short-term bullrun 80/20 with RSI 49.5 Healthy pullback.
- **Lesson:** one source of truth for bands prevents the guide drifting from the logic.

Files: `glossary.py`, `app.py` (guide after bot panel)

### D10 — Backend review fixes (single source of truth + tests)

- **Context:** architecture review found 4 real issues: shared metrics filename, duplicated
  indicator logic, unpinned requirements, no automated tests.
- **Decision:** (1) new `indicators.py` (RSI/SMA/signals) imported by both `app.py` and `bot.py` —
  verified bit-identical to the old app logic before deleting it; (2) `train_model.py` writes
  metrics next to the model (`model_demo.joblib` -> `metrics_demo.json`, legacy `model.joblib` ->
  `metrics.json`) and sets `random_state=42` for full reproducibility; (3) `requirements.txt`
  pinned from the working environment; (4) new `tests/` (19 tests, pytest) covering indicator math,
  regime extremes/clamping/block merging, bot hysteresis/validation/causality and glossary bands.
- **Result:** `pytest tests/` 19 passed. Proof the duplication had already drifted: the 365-day 70/30
  backtest moved from -6.4% to +1.2% after unifying (full-history numbers unchanged). The old
  misleading `artifacts/metrics.json` (demo content under a production name) was removed.
- **Lesson:** duplicated math always drifts; failing-then-passing tests are the portfolio proof.

Files: `indicators.py`, `train_model.py`, `requirements.txt`, `tests/`, `app.py`, `bot.py`

### D11 — Dark trading-terminal design

- **Context:** user asked to improve the page design; chose the dark terminal style.
- **Decision:** `.streamlit/config.toml` (base dark, near-black background, market-green primary)
  plus `assets/terminal.css` (1200px column, green-ruled title, card-style KPI metrics with
  uppercase muted labels and tabular numbers, green sidebar titles, compact tables), injected
  once via `_load_terminal_css()` in `app.py` (no-op if the file is missing).
- **Result:** headless server boots clean on a test port (health + page 200, no errors),
  `pytest tests/` 22 passed (incl. new `test_design.py`: theme parses, CSS hooks exist).
- **Lesson:** theme in config + one stylesheet keeps design out of the Python logic.

Files: `.streamlit/config.toml`, `assets/terminal.css`, `app.py`, `tests/test_design.py`

### D12 — GitHub launch kit (README, MIT, weekly Action)

- **Context:** user wants the project to have a future on GitHub as a hiring portfolio.
- **Decision:** `README.md` (badges, real result tables incl. the honest demo failure, 2-minute
  quickstart, pipeline diagram, structure, roadmap, disclaimer), MIT `LICENSE`, and
  `.github/workflows/weekly-update.yml` (Mondays 07:00 UTC: RSS --append, prices, dataset,
  pytest gate, train retry allowed to fail, auto-commit of data + artifacts). Screenshot slots
  under `docs/screenshots/` are referenced as openly-marked placeholders with capture steps.
- **Result:** workflow YAML structurally validated, `pytest tests/` 22 passed.
- **Lesson:** a repo that updates itself weekly signals life to recruiters for free.

Files: `README.md`, `LICENSE`, `.github/workflows/weekly-update.yml`

## 4. How an employer can verify this in 5 minutes

1. `ver_app.bat` → check top metrics, “Should you buy now?”, then **Bullrun vs Bear Regime**, then **Threshold Bot**, then **Price and Signals with regime bands**, then **News Sentiment (RSS)**.
2. Move Sidebar sliders **Buy threshold (high)** / **Sell threshold (low)** — metrics, equity curve and trades update. Try 70/30 vs 80/20 vs 60/40.
3. Run `actualizar_datos.bat` → see the 4-step keyless refresh.
4. `python -m py_compile app.py bull_bear.py bot.py glossary.py indicators.py` → compile OK.
5. `python bot.py --buy 70 --sell 30` → CLI backtest with trades CSV.
6. `python -m pytest tests/ -q` → 22 passed.

*Academic demo, not investment advice. All data via yfinance + free RSS; deterministic logic, no network calls at render time.*
