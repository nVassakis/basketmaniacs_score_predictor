import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle, Rectangle, Arc

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHOTS_CSV = os.path.join(BASE_DIR, 'data', 'processed', 'shots_master.csv')

# Mapping regions to numbers for the meshgrid
REGION_NUMBERS = {
    "Box": 0, "Left Side": 1, "Right Side": 2, "Top Mid-Range": 3,
    "Left Corner 3": 4, "Right Corner 3": 5,
    "Left-45 3pt": 6, "Right-45 3pt": 7, "Top 3pt": 8,
}

# Where the FG% circle sits inside each region
REGION_CENTERS = {
    "Box":            (0,    28),
    "Left Side":      (-140, 60),
    "Right Side":     (140,  60),
    "Top Mid-Range":  (0,    140),
    "Left Corner 3":  (-247, 35),
    "Left-45 3pt":    (-170, 200),
    "Top 3pt":        (0,    260),
    "Right-45 3pt":   (170,  200),
    "Right Corner 3": (247,  35),
}

# Where the attempt count sits
REGION_ATTEMPT_OFFSETS = {
    "Box":            (53,  -43),
    "Left Side":      (-67, -43),
    "Right Side":     (213, -43),
    "Top Mid-Range":  (53,   74),
    "Top 3pt":        (53,  237),
    "Left-45 3pt":    (-225, 96),
    "Right-45 3pt":   (268,  96),
    "Left Corner 3":  (-225, -43),
    "Right Corner 3": (268, -43),
}

# Cap on the vs-league colour scale
VS_LEAGUE_LIMIT = 15


def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    if ax is None:
        ax = plt.gca()

    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color, fill=False)
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180, linewidth=lw, color=color, fill=False)
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0, linewidth=lw, color=color, linestyle='dashed')
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw, color=color)
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw, color=color)
    corner_three_b = Rectangle((219.5, -47.5), 0, 140, linewidth=lw, color=color)
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw, color=color)
    center_outer_arc = Arc((0, 375), 120, 120, theta1=180, theta2=0, linewidth=lw, color=color)

    court_elements = [hoop, backboard, inner_box, top_free_throw, bottom_free_throw,
                      restricted, corner_three_a, corner_three_b, three_arc, center_outer_arc]

    if outer_lines:
        court_elements.append(Rectangle((-275, -47.5), 550, 422.5, linewidth=lw, color=color, fill=False))

    for element in court_elements:
        ax.add_patch(element)

    return ax

# Boundaries are duplicated in _build_region_grid — change both if needed.
def get_region(court_x, court_y, shot_type):
    if shot_type == "2PT":
        if -60 <= court_x <= 60 and court_y < 70:
            return "Box"
        elif -60 <= court_x <= 60 and court_y >= 70:
            return "Top Mid-Range"
        elif court_x < -60:
            return "Left Side"
        elif court_x > 60:
            return "Right Side"
        else:
            return "Undefined"
    else:  # 3PT
        if court_x < -220 and court_y < 92.5:
            return "Left Corner 3"
        elif court_x > 220 and court_y < 92.5:
            return "Right Corner 3"
        elif court_x < -80:
            return "Left-45 3pt"
        elif court_x > 80:
            return "Right-45 3pt"
        elif -80 <= court_x <= 80 and court_y >= 92.5:
            return "Top 3pt"
        else:
            return "Undefined"


def load_shots(csv_path=SHOTS_CSV):
    """Load shots_master.csv with a REGION column added, minus unclassifiable shots."""
    shots_df = pd.read_csv(csv_path)
    shots_df["REGION"] = shots_df.apply(
        lambda row: get_region(row["COURT_X"], row["COURT_Y"], row["SHOT_TYPE"]), axis=1
    )
    return shots_df[shots_df["REGION"] != "Undefined"]


def region_summary(df, prefix=""):
    """Attempts / makes / FG% per region for the given shots."""
    results = []
    for region, group in df.groupby("REGION"):
        attempts = len(group)
        made = int(group["MADE"].sum())
        pct = made / attempts * 100 if attempts else 0
        results.append({
            "REGION": region,
            f"{prefix}ATTEMPTS": attempts,
            f"{prefix}MADE": made,
            f"{prefix}PCT": round(pct, 1),
        })
    return pd.DataFrame(results).sort_values(f"{prefix}ATTEMPTS", ascending=False)


def _style_court(ax, title):
    ax.set_xlim(-300, 300)
    ax.set_ylim(-100, 400)
    ax.set_title(title, fontsize=22, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _build_region_grid(X, Y):
    """Label every point of the meshgrid with its region id (-1 = off-court)."""
    dist = np.sqrt(X ** 2 + Y ** 2)
    arc_radius = 237.5

    in_corner_left = (X < -220) & (Y < 92.5)
    in_corner_right = (X > 220) & (Y < 92.5)
    outside_arc = (dist > arc_radius) | in_corner_left | in_corner_right
    inside_arc = ~outside_arc
    outer_limit = dist <= arc_radius * 1.25

    grid = np.full(X.shape, -1)

    grid[inside_arc & (X >= -60) & (X <= 60) & (Y < 70)] = REGION_NUMBERS["Box"]
    grid[inside_arc & (X >= -60) & (X <= 60) & (Y >= 70)] = REGION_NUMBERS["Top Mid-Range"]
    grid[inside_arc & (X < -60)] = REGION_NUMBERS["Left Side"]
    grid[inside_arc & (X > 60)] = REGION_NUMBERS["Right Side"]

    outside_not_corner = outside_arc & ~in_corner_left & ~in_corner_right & outer_limit
    grid[in_corner_left] = REGION_NUMBERS["Left Corner 3"]
    grid[in_corner_right] = REGION_NUMBERS["Right Corner 3"]
    grid[outside_not_corner & (X < -60)] = REGION_NUMBERS["Left-45 3pt"]
    grid[outside_not_corner & (X > 60)] = REGION_NUMBERS["Right-45 3pt"]
    grid[outside_not_corner & (X >= -60) & (X <= 60) & (Y >= 92.5)] = REGION_NUMBERS["Top 3pt"]

    return grid


def plot_simple(player_df, player_name, season):
    """Made/missed scatter of every shot."""
    made = player_df[player_df["MADE"]]
    missed = player_df[~player_df["MADE"]]

    fig, ax = plt.subplots(figsize=(12, 11))
    draw_court(ax, outer_lines=True)

    ax.scatter(missed["COURT_X"], missed["COURT_Y"], facecolors="none", edgecolors="red",
               s=140, linewidths=2, alpha=0.5, label="Missed")
    ax.scatter(made["COURT_X"], made["COURT_Y"], facecolors="limegreen", edgecolors="darkgreen",
               s=140, linewidths=2, alpha=0.5, label="Made")

    _style_court(ax, f"Shot chart - {player_name} ({season})")
    ax.legend(loc="upper right")
    return fig


def plot_heat(player_df, league_df, player_name, season, show_table=False):
    """Court shaded by the player's FG% against the league average for that region."""
    comparison = region_summary(player_df).merge(
        region_summary(league_df, prefix="LEAGUE_")[["REGION", "LEAGUE_PCT"]], on="REGION"
    )
    comparison["VS_LEAGUE"] = comparison["PCT"] - comparison["LEAGUE_PCT"]
    if show_table:
        print(comparison.to_string(index=False))


    x_vals = np.linspace(-275, 275, 400)
    y_vals = np.linspace(-50, 400, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    region_grid = _build_region_grid(X, Y)

    value_grid = np.full(X.shape, np.nan)
    for _, row in comparison.iterrows():
        value_grid[region_grid == REGION_NUMBERS[row["REGION"]]] = row["VS_LEAGUE"]
    value_grid[Y < -47.5] = np.nan

    fig, ax = plt.subplots(figsize=(12, 11))
    color_scale = TwoSlopeNorm(vcenter=0, vmin=-VS_LEAGUE_LIMIT, vmax=VS_LEAGUE_LIMIT)
    ax.pcolormesh(X, Y, value_grid, cmap=plt.cm.RdYlGn, norm=color_scale, alpha=0.6, zorder=0)
    draw_court(ax, outer_lines=True)

    max_attempts = comparison["ATTEMPTS"].max()
    for _, row in comparison.iterrows():
        cx, cy = REGION_CENTERS[row["REGION"]]
        circle_size = 550 + 3700 * (row["ATTEMPTS"] / max_attempts)

        ax.scatter(cx, cy, s=circle_size, color="white", edgecolors="black",
                   linewidths=1.5, zorder=5, alpha=0.7)
        ax.text(cx, cy, f"{row['PCT']:.0f}%", ha="center", va="center",
                fontsize=12, fontweight="bold", color="black", zorder=6)

        ox, oy = REGION_ATTEMPT_OFFSETS[row["REGION"]]
        ax.text(ox, oy, f"{int(row['ATTEMPTS'])}", ha="center", va="center",
                fontsize=8, fontweight="bold", color="black", zorder=7)

    _style_court(ax, f"{player_name} ({season})")
    ax.text(0, -80,
            "Region color: green = above league avg, red = below  |  "
            "Circle size = shot volume  |  Bottom-right number = attempts",
            ha="center", va="center", fontsize=10, color="gray")
    return fig


def plot_player_shots(player_name, season, chart="simple", csv_path=SHOTS_CSV, save_to=None, show_table=False):
    shots_df = load_shots(csv_path)

    season_df = shots_df[shots_df["SEASON"] == season]
    player_df = season_df[season_df["PLAYER"] == player_name]

    if player_df.empty:
        print(f"No shots found for {player_name} in {season}.")
        return None

    if chart == "simple":
        fig = plot_simple(player_df, player_name, season)
    else:
        fig = plot_heat(player_df, season_df, player_name, season, show_table=show_table)

    if save_to:
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_to}")
    else:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot a player's shot chart for a season.")
    parser.add_argument("player", help="Player name as it appears in shots_master.csv")
    parser.add_argument("season", help="Season, e.g. 2025-26")
    parser.add_argument("--chart", choices=["simple", "heat"], default="simple",
                        help="simple: made/missed scatter. heat: FG%% by region vs league average.")
    parser.add_argument("--save", metavar="PATH", help="Write the chart to a file instead of showing it")
    parser.add_argument("--csv", default=SHOTS_CSV, help="Path to shots_master.csv")
    parser.add_argument("--table", action="store_true", help="Print the region comparison table")


    args = parser.parse_args()

    plot_player_shots(args.player, args.season, chart=args.chart, csv_path=args.csv,
                      save_to=args.save, show_table=args.table)


if __name__ == "__main__":
    main()
