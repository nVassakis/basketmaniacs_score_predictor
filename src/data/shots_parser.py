import os
import json
import pandas as pd
from bs4 import BeautifulSoup

from cleaners import greek_to_latin

# --- CONFIGURATION ---
BASE_DIR = os.getcwd()
INDEX_FILE = os.path.join(BASE_DIR, 'data', 'raw', 'master_games_index.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'shots_master.csv')
MAPPING_FILE = os.path.join(BASE_DIR, 'config', 'team_mapping.json')

with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
    TEAM_MAPPING = json.load(f)

HOOP_X_AWAY = 944
HOOP_X_HOME = 1000 - HOOP_X_AWAY  # 56
HOOP_Y = 250

# Scaling factors to transorm the coordinates to the new half-court
SCALE_X = 1.16
SCALE_Y = 1.16


# --- HELPERS ---
def convert_player_name(raw_name):
    if ',' in raw_name:
        last, first = raw_name.split(',', 1)
        raw_name = f"{first.strip()} {last.strip()}"
    return greek_to_latin(raw_name)


def get_roster(soup, team_side):
    """
    Return {player_id: player_name} for one team.
    """
    roster = {}
    for label in soup.select("label.player-check"):
        is_away = "away" in label.get("class", [])
        if (team_side == "away") != is_away:
            continue

        player_id = label.find("input")["id"].replace("player-", "")
        name_div = label.find_parent("div", class_="player-name")
        roster[player_id] = convert_player_name(name_div.get_text(strip=True))

    return roster


def get_player_shots(soup, player_id):
    """Return a list of raw shot dicts for one player, in the order they
    appear in the SVG."""
    shots = []
    for index, g in enumerate(soup.select(f'g[data-user="{player_id}"]')):
        circle = g.find("circle")
        stat = g["data-stat"]
        shots.append({
            "shot_index": index,
            "period": int(g["data-period"]) + 1,
            "shot_type": "2PT" if stat.startswith("2") else "3PT",
            "made": stat.endswith("S"),
            "raw_x": int(circle["cx"]),
            "raw_y": int(circle["cy"]),
        })
    return shots


def convert_coordinates(raw_x, raw_y, team_side):
    """
    Convert raw SVG pixel coordinates into court coordinates.
    """
    if team_side == "away":
        court_x = (HOOP_Y - raw_y) * SCALE_X
        court_y = (HOOP_X_AWAY - raw_x) * SCALE_Y
    else:
        court_x = (raw_y - HOOP_Y) * SCALE_X
        court_y = (raw_x - HOOP_X_HOME) * SCALE_Y

    return court_x, court_y


def get_match_date(soup):
    date_element = soup.find('time', itemprop='startDate')
    if not date_element:
        return None
    return date_element.text.strip().split()[0]


def get_team_names(soup):
    """Return (home_team, away_team) using the box score table captions,
    mapped to canonical team names via team_mapping.json."""
    teams = {}
    for table_id, side in [('homeDataTable', 'home'), ('awayDataTable', 'away')]:
        table = soup.find('table', id=table_id)
        if not table:
            return None, None

        caption = table.find_previous('h4', class_='sp-table-caption')
        raw_name = caption.text.strip()
        teams[side] = TEAM_MAPPING.get(raw_name, raw_name)

    return teams['home'], teams['away']


# --- MAIN PARSER ---
def parse_shot_data(filepath):
    """Parse one match.html file into a DataFrame with one row per shot,
    for every player on both teams."""
    with open(filepath, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'lxml')

    match_date = get_match_date(soup)
    home_team, away_team = get_team_names(soup)
    if match_date is None or home_team is None:
        return pd.DataFrame()

    teams = [(home_team, away_team, "home"), (away_team, home_team, "away")]

    rows = []
    for team, opponent, team_side in teams:
        roster = get_roster(soup, team_side)

        for player_id, player_name in roster.items():
            for shot in get_player_shots(soup, player_id):
                court_x, court_y = convert_coordinates(shot["raw_x"], shot["raw_y"], team_side)
                rows.append({
                    "DATE": match_date,
                    "TEAM": team,
                    "OPPONENT": opponent,
                    "TEAM_SIDE": team_side,
                    "PLAYER_ID": player_id,
                    "PLAYER": player_name,
                    "PERIOD": shot["period"],
                    "SHOT_TYPE": shot["shot_type"],
                    "MADE": shot["made"],
                    "SHOT_INDEX": shot["shot_index"],
                    "RAW_X": shot["raw_x"],
                    "RAW_Y": shot["raw_y"],
                    "COURT_X": court_x,
                    "COURT_Y": court_y,
                })

    return pd.DataFrame(rows)


# --- PIPELINE ---
def run_pipeline():
    index_df = pd.read_csv(INDEX_FILE, header=None, names=['URL', 'Team_Slug', 'File_Path'])

    processed_urls = set()
    all_games = []

    for _, row in index_df.iterrows():
        url, file_path = row['URL'], row['File_Path']

        # Each match is listed twice (once per team's index entry). Both
        # copies contain shots for both teams, so only parse it once.
        if url in processed_urls:
            continue
        if not os.path.exists(file_path):
            continue

        game_df = parse_shot_data(file_path)
        if not game_df.empty:
            all_games.append(game_df)
            processed_urls.add(url)

    if not all_games:
        print("No shot data found.")
        return

    shots_df = pd.concat(all_games, ignore_index=True)
    shots_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote {len(shots_df)} shots from {len(all_games)} games to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_pipeline()
