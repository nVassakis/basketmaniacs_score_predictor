import os
import json
import pandas as pd
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
from cleaners import greek_to_latin, get_opponent, validate_team_points

BASE_DIR = os.getcwd()
INDEX_FILE = os.path.join(BASE_DIR, 'data', 'raw', 'master_games_index.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'full_stats_master.csv')
MAPPING_FILE = os.path.join(BASE_DIR, 'config', 'team_mapping.json')

with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
    TEAM_MAPPING = json.load(f)

# --- PARSERS ---
def get_team_metadata(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            dfs = pd.read_html(f.read())
        for df in dfs:
            if {'Date', 'Match', 'Result', 'Season'}.issubset(df.columns):
                df['Date'] = df['Date'].astype(str).str.split().str[0]
                return df[['Date', 'Match', 'Result', 'Season']]
    except Exception:
        pass
    return pd.DataFrame()

def parse_game_stats(filepath, team_slug, url):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            
        date_element = soup.find('time', itemprop='startDate')
        if not date_element: return None
        match_date = date_element.text.strip().split()[0] 
        
        player_data = []
        for table_id in ['homeDataTable', 'awayDataTable']:
            table = soup.find('table', id=table_id)
            if not table: continue
            
            actual_team_name = table.find_previous('h4', class_='sp-table-caption').text.strip()
            for row in table.find('tbody').find_all('tr', class_='sp-total-row'):
                cols = [c.text.strip() for c in row.find_all(['td', 'th'])]
                if len(cols) >= 17 and "Σύνολα" not in cols[0]:
                    player_data.append({
                        'Date': match_date, 'Team': actual_team_name,
                        'Player': cols[0], 'EFF': cols[1], 'PTS': cols[2],
                        '2FG_MA': cols[3], '2FG_PCT': cols[4], '3FG_MA': cols[5], '3FG_PCT': cols[6],
                        'FT_MA': cols[7], 'FT_PCT': cols[8], 'AST': cols[9], 'STL': cols[10], 
                        'BLK': cols[11], 'REB_TOT': cols[12], 'REB_OFF': cols[13], 'REB_DEF': cols[14],
                        'TO': cols[15], 'FLS': cols[16]
                    })
        return pd.DataFrame(player_data)
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def transform_and_clean(df, team_meta_df):
    df = pd.merge(df, team_meta_df, on='Date', how='inner')
    
    df['Player'] = df['Player'].apply(
        lambda x: f"{x.split(',')[1].strip()} {x.split(',')[0].strip()}" if isinstance(x, str) and ',' in x else str(x).strip()
    )
    df['Player'] = df['Player'].apply(greek_to_latin)

    # Build opponent directly from Match relative to each player's Team row.
    team_raw = df['Team'].astype(str).str.strip()
    match_split = df['Match'].astype(str).str.split('-', n=1, expand=True)
    left = match_split[0].astype(str).str.strip()
    right = match_split[1].astype(str).str.strip() if match_split.shape[1] > 1 else pd.Series('', index=df.index)
    team_norm = team_raw.str.lower().str.replace(r'[\s\-]+', '', regex=True)
    left_norm = left.str.lower().str.replace(r'[\s\-]+', '', regex=True)
    right_norm = right.str.lower().str.replace(r'[\s\-]+', '', regex=True)

    team_is_left = team_norm == left_norm
    team_is_right = team_norm == right_norm

    df['Opponent'] = df.apply(get_opponent, axis=1)

    df['Team'] = team_raw.replace(TEAM_MAPPING)
    df['Opponent'] = df['Opponent'].replace(TEAM_MAPPING)

    for col in ['2FG_MA', '3FG_MA', 'FT_MA']:
        if col in df.columns:
            prefix = col.replace('_MA', '')
            split_df = df[col].astype(str).str.split(r'[/]', expand=True)
            df[f'{prefix}_M'] = pd.to_numeric(split_df[0], errors='coerce').fillna(0).astype(int)
            df[f'{prefix}_A'] = pd.to_numeric(split_df[1], errors='coerce').fillna(0).astype(int)
    df = df.drop(columns=['2FG_MA', '3FG_MA', 'FT_MA'], errors='ignore')

    for col in ['2FG_PCT', '3FG_PCT', 'FT_PCT']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', ''), errors='coerce').fillna(0.0)

    df['Date_Obj'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df['YEAR'], df['MONTH'], df['DAY'] = df['Date_Obj'].dt.year, df['Date_Obj'].dt.month, df['Date_Obj'].dt.day
    return df

def run_pipeline():
    processed_urls = set()
    index_df = pd.read_csv(INDEX_FILE, header=None, names=['URL', 'Team_Slug', 'File_Path'])
    team_meta_cache = {}
    all_data = []

    for _, row in index_df.iterrows():
        url, team_slug, file_path = row['URL'], row['Team_Slug'], row['File_Path']

        if url in processed_urls: continue
        if not os.path.exists(file_path): continue

        if team_slug not in team_meta_cache:
            team_page = os.path.join(BASE_DIR, 'data', 'raw', 'teams', team_slug, 'team_page.html')
            team_meta_cache[team_slug] = get_team_metadata(team_page)

        game_df = parse_game_stats(file_path, team_slug, url)
        if game_df is not None and not game_df.empty and not team_meta_cache[team_slug].empty:
            cleaned_df = transform_and_clean(game_df, team_meta_cache[team_slug])
            cleaned_df = validate_team_points(cleaned_df)
            all_data.append(cleaned_df)
            processed_urls.add(url)

    if all_data:
        master_df = pd.concat(all_data, ignore_index=True)
        cols_to_keep = [
            'Season', 'Date', 'YEAR', 'MONTH', 'DAY', 'Player', 'Match', 'Result',
            'Team', 'Opponent', 'EFF', 'PTS', 'FT_M', 'FT_A', 'FT_PCT', '2FG_M', '2FG_A',
            '2FG_PCT', '3FG_M', '3FG_A', '3FG_PCT', 'AST', 'STL', 'BLK', 'REB_TOT',
            'REB_OFF', 'REB_DEF', 'TO', 'FLS'
        ]
        master_df = master_df[[c for c in cols_to_keep if c in master_df.columns]]
        master_df.columns = master_df.columns.str.upper()
        if os.path.exists(OUTPUT_FILE):
            existing_df = pd.read_csv(OUTPUT_FILE)
            master_df = pd.concat([existing_df, master_df], ignore_index=True)
            master_df = master_df.drop_duplicates(subset=['DATE', 'PLAYER', 'TEAM', 'MATCH'])

        master_df.to_csv(OUTPUT_FILE, mode='w', header=True, index=False)
        print(f"Wrote {len(all_data)} games to {OUTPUT_FILE}")
    else:
        print("No data to process.")

if __name__ == "__main__":
    run_pipeline()