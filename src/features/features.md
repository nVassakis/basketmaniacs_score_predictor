# Features

Two feature tables are built from raw game data: one at **player grain** (`ml_features`) used by the player scoring model, and one at **team grain** (`ml_team_features`) used by the team score model.

All features are computed using only games that occurred **before** the target game — no current-game stats are ever visible during training or prediction.

---

## Player Features — `ml_features`

### Short-Term Form
3-game rolling averages. Each series is shifted by one game before rolling so the current game is never included.

| Feature | Description |
|---|---|
| `PTS_last_3` | Average points scored over the last 3 games |
| `EFF_last_3` | Average efficiency rating over the last 3 games |
| `USG_last_3` | Average shot attempts (2FG + 3FG + FT) over the last 3 games |
| `AST_last_3` | Average assists over the last 3 games |
| `REB_last_3` | Average total rebounds over the last 3 games |
| `BLK_last_3` | Average blocks over the last 3 games |
| `STL_last_3` | Average steals over the last 3 games |
| `TS_PCT_last_3` | Average true shooting percentage over the last 3 games |

### Season Baseline
Expanding averages from game 1 of the current season up to (but not including) the current game. Reset at the start of each new season.

| Feature | Description |
|---|---|
| `PTS_season_avg` | Season scoring average |
| `EFF_season_avg` | Season efficiency average |
| `REB_season_avg` | Season rebounding average |
| `AST_season_avg` | Season assist average |
| `BLK_season_avg` | Season block average |
| `STL_season_avg` | Season steal average |
| `USG_season_avg` | Season usage average (shot attempts) |
| `TS_PCT_season_avg` | Season true shooting percentage |

### Matchup History
Expanding averages grouped by player + opponent, capturing how a player performs against a specific team across all prior meetings.

| Feature | Description |
|---|---|
| `PTS_vs_OPP_hist` | Player's historical scoring average against this opponent |
| `EFF_vs_OPP_hist` | Player's historical efficiency average against this opponent |

### Opponent Vulnerability
How well (or poorly) the opponent defends, measured at player grain.

| Feature | Description |
|---|---|
| `OPP_PTS_ALLOWED_PER_PLAYER` | Season average points this opponent concedes per individual player |
| `OPP_REB_ALLOWED_PER_PLAYER` | Season average rebounds this opponent concedes per individual player |
| `OPP_PPG_ALLOWED` | Average total points this opponent has conceded per game |

### Lineup Context

| Feature | Description |
|---|---|
| `SCORING_VACUUM` | Gap between the team's historical PPG and tonight's active roster combined scoring average. Captures the impact of missing players — a positive vacuum means the team is under-strength tonight. |
| `TEAM_USG_RANK` | Player's usage rank within tonight's active lineup (1 = highest usage). Captures role context: a star player stepping into the top role scores differently than when they're the third option. |

### Calendar

| Feature | Description |
|---|---|
| `MONTH` | Month of the game — captures seasonal trends such as playoff intensity |
| `DAY_OF_WEEK` | Day of the week — captures schedule-based fatigue patterns |

---

## Team Features — `ml_team_features`

### Lineup Context

| Feature | Description |
|---|---|
| `ROSTER_COUNT` | Total number of players active for the game |
| `SOLID_PLAYER_COUNT` | Number of active players with a historical EFF above 11 |
| `BIG_3_EFF_SUM` | Combined season efficiency of the top 3 players active tonight |
| `ACTIVE_ROSTER_PTS` | Sum of season scoring averages of all active players |
| `ACTIVE_ROSTER_EFF` | Sum of season efficiency averages of all active players |
| `ACTIVE_ROSTER_STL` | Sum of season steal averages of all active players (lineup defence) |
| `ACTIVE_ROSTER_BLK` | Sum of season block averages of all active players (lineup defence) |
| `TEAM_SEASON_AVG_PTS` | Team's overall scoring average for the current season |
| `ROSTER_SCORING_VARIANCE` | Difference between tonight's active lineup scoring power and the team's season average |

### Team Form

| Feature | Description |
|---|---|
| `TEAM_PTS_last_5` / `_season` | Average points scored over the last 5 games / full season |
| `TEAM_AST_last_5` / `_season` | Average assists over the last 5 games / full season |
| `TEAM_REB_last_5` | Average total rebounds over the last 5 games |
| `TEAM_3PT_PCT_last_5` / `_season` | True 3-point percentage over the last 5 games / full season |
| `TEAM_PTS_ALLOWED_last_5` / `_season` | Average points conceded over the last 5 games / full season |

### Opponent Form

| Feature | Description |
|---|---|
| `OPP_PTS_ALLOWED_last_5` / `_season` | Average points the opponent gives up to teams (defensive weakness) |
| `OPP_REB_ALLOWED_last_5` / `_season` | Average rebounds the opponent allows |
| `OPP_3PT_PCT_ALLOWED_season` | Historical 3-point percentage allowed by the opponent's defence |
| `OPP_TEAM_PTS_last_5` | Opponent's recent scoring form |
| `OPP_TEAM_3PT_PCT_last_5` / `_season` | Opponent's recent and long-term 3-point shooting threat |

### Opponent Lineup Context

| Feature | Description |
|---|---|
| `OPP_ROSTER_COUNT` | Total players active for the opponent tonight |
| `OPP_SOLID_PLAYER_COUNT` | Number of high-efficiency players active for the opponent |
| `OPP_BIG_3_EFF_SUM` | Star power of the opponent's top 3 players tonight |
| `OPP_ROSTER_SCORING_VARIANCE` | Whether the opponent is missing key scorers tonight |
| `OPP_ACTIVE_ROSTER_STL` / `_BLK` | Defensive disruption potential of the opponent's active lineup |

### Matchup & Calendar

| Feature | Description |
|---|---|
| `H2H_PTS_season` | Average points this team scores against this specific opponent in the current season |
| `MONTH` | Month of the game |
| `DAY_OF_WEEK` | Day of the week |
