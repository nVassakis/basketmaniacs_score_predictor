# Pipeline Architecture

## Execution Order

Run these in sequence to build everything from scratch:

```
1. src/data/scraper.py
2. src/data/parser.py
3. src/data/ingest.py
4. src/features/logic.py     
5. src/models/train.py

# Team model (separate grain):
6. src/features/team_logic.py
7. src/models/team_train.py
```

---

## Data Layer (`src/data/`)

**`scraper.py`** — Iterates over team slugs in `config/teams.txt` and downloads each team page and game page from basketmaniacs.com. Already-downloaded files are skipped; randomised delays are applied between requests to be polite to the server.

**`parser.py`** — Reads raw HTML, extracts one row per player per game, and merges with team-level metadata (date, result, season). Calls into `cleaners.py` and outputs `data/processed/full_stats_master.csv`.

**`cleaners.py`** — Handles three concerns: Greek-to-Latin character transliteration (player names are stored in Greek on the site), opponent resolution from the matchup string, and cross-validation of player point totals against the game scoreline to catch bad parses before they reach the model.

**`ingest.py`** — Loads the processed CSV into a SQLite database as the `raw_stats` table.

---

## Feature Engineering (`src/features/`)

All features are built using only historical data — no current-game stats are visible to the model at prediction time. Features are computed at two grains:

**`logic.py`** builds the **player-grain** feature table (`ml_features`). Key feature groups:

| Group | Examples |
|---|---|
| Short-term form | 3-game rolling averages for PTS, EFF, USG, AST, BLK, STL, REB, TS% |
| Season baseline | Expanding season averages for the same stats |
| Matchup history | Player's historical average against this specific opponent |
| Opponent defence | How many points this opponent typically concedes per player |
| Lineup context | `SCORING_VACUUM` — gap between team's usual PPG and tonight's active roster scoring power |
| Role | `TEAM_USG_RANK` — player's usage rank within tonight's active lineup |

**`team_logic.py`** builds the **team-grain** feature table (`ml_team_features`) by aggregating player rows into one row per team per game, then layering on team form, opponent form, and lineup context features. Full feature definitions are in [features/features.md](features/features.md).

---

## Models (`src/models/`)

**`train.py`** trains a player points predictor using a chronological train-test split, so the model is always evaluated on games it has never seen. Performance is reported against a naive rolling-average baseline and broken out by player tier (Bench / Role / Star) to surface where the model adds real value.

**`team_train.py`** trains a team score predictor using walk-forward cross-validation — each fold trains only on games from before its test window, mirroring real-world deployment. Two baselines (global mean and per-team historical mean) are evaluated alongside the model per fold. Experiment metrics and artifacts are tracked with MLflow.