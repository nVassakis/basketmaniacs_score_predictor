import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHOTS_CSV = os.path.join(BASE_DIR, 'data', 'processed', 'shots_master.csv')


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


def plot_player_shots(player_name, season):
    """Plot the shot chart for a player in a given season.

    Args as they appear in the shots_master.csv
    """
    shots_df = pd.read_csv(SHOTS_CSV)

    player_df = shots_df[(shots_df["PLAYER"] == player_name) & (shots_df["SEASON"] == season)]

    if player_df.empty:
        print(f"No shots found for {player_name} in {season}.")
        return

    made = player_df[player_df["MADE"]]
    missed = player_df[~player_df["MADE"]]

    fig, ax = plt.subplots(figsize=(12, 11))
    draw_court(ax, outer_lines=True)

    ax.scatter(missed["COURT_X"], missed["COURT_Y"], facecolors="none", edgecolors="red",
               s=140, linewidths=2, alpha=0.5, label="Missed")
    ax.scatter(made["COURT_X"], made["COURT_Y"], facecolors="limegreen", edgecolors="darkgreen",
               s=140, linewidths=2, alpha=0.5, label="Made")

    ax.set_xlim(-300, 300)
    ax.set_ylim(-100, 400)
    ax.set_title(f"Shot chart - {player_name} ({season})", fontsize=22, fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.show()


if __name__ == "__main__":
    plot_player_shots("Nikolaos Vassakis", "2025-26")
