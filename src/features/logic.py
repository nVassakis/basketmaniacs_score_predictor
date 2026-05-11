import pandas as pd
import sqlite3

def engineer_features():
    db_path = 'data/basketball.db'
    conn = sqlite3.connect(db_path)

    print("Loading raw stats...")
    df = pd.read_sql("SELECT * FROM raw_stats", conn)

    # Sort the data chronologically
    df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y')
    df = df.sort_values(by=['DATE']).reset_index(drop=True)

    # Calendar Features
    print("Extracting Calendar Features...")
    df['MONTH'] = df['DATE'].dt.month
    df['DAY_OF_WEEK'] = df['DATE'].dt.dayofweek

    # Calculate 'USG' (Usage) - Total shots taken
    df['USG'] = df['2FG_A'] + df['3FG_A'] + df['FT_A']

    # ---------------------------------------------------------
    # True Shooting Percentage (Raw)
    # ---------------------------------------------------------
    print("Calculating True Shooting Percentage...")
    # TSA = True Shooting Attempts
    df['TSA'] = df['2FG_A'] + df['3FG_A'] + (0.44 * df['FT_A'])
    
    # Calculate raw TS% (Multiply TSA by 2 for the standard formula)
    df['TS_PCT_raw'] = (df['PTS'] / (2 * df['TSA'])).fillna(0)

    # ---------------------------------------------------------
    # Step 1: Short-Term Form -> Last 3 Games
    # ---------------------------------------------------------
    print("Calculating Short-Term Form...")

    # .shift(1) hides today's game. .rolling(3) looks at the last 3 games.
    def get_last_3_avg(series):
        return series.shift(1).rolling(window=3, min_periods=1).mean()
    
    # Apply the formula to each player individually using groupby
    df['PTS_last_3'] = df.groupby('PLAYER')['PTS'].transform(get_last_3_avg)
    df['EFF_last_3'] = df.groupby('PLAYER')['EFF'].transform(get_last_3_avg)
    df['USG_last_3'] = df.groupby('PLAYER')['USG'].transform(get_last_3_avg)
    df['AST_last_3'] = df.groupby('PLAYER')['AST'].transform(get_last_3_avg)
    df['BLK_last_3'] = df.groupby('PLAYER')['BLK'].transform(get_last_3_avg)
    df['STL_last_3'] = df.groupby('PLAYER')['STL'].transform(get_last_3_avg)
    df['REB_last_3'] = df.groupby('PLAYER')['REB_TOT'].transform(get_last_3_avg)
    df['TS_PCT_last_3'] = df.groupby('PLAYER')['TS_PCT_raw'].transform(get_last_3_avg)


    # ---------------------------------------------------------
    # Step 2: Long-Term Form -> Season Average
    # ---------------------------------------------------------
    print("Calculating Season Baselines...")
    
    # .expanding() means "take the average of ALL previous games in the group"
    def get_season_avg(series):
        return series.shift(1).expanding(min_periods=1).mean()

    # Group by both Player AND Season (so the math resets every new year)
    df['PTS_season_avg'] = df.groupby(['PLAYER', 'SEASON'])['PTS'].transform(get_season_avg)
    df['REB_season_avg'] = df.groupby(['PLAYER', 'SEASON'])['REB_TOT'].transform(get_season_avg)
    df['EFF_season_avg'] = df.groupby(['PLAYER', 'SEASON'])['EFF'].transform(get_season_avg)
    df['AST_season_avg'] = df.groupby(['PLAYER', 'SEASON'])['AST'].transform(get_season_avg)
    df['BLK_season_avg'] = df.groupby(['PLAYER', 'SEASON'])['BLK'].transform(get_season_avg)
    df['STL_season_avg'] = df.groupby(['PLAYER', 'SEASON'])['STL'].transform(get_season_avg)
    df['TS_PCT_season_avg'] = df.groupby(['PLAYER', 'SEASON'])['TS_PCT_raw'].transform(get_season_avg)
    df['USG_season_avg'] = df.groupby(['PLAYER', 'SEASON'])['USG'].transform(get_season_avg)

    # ---------------------------------------------------------
    # Step 3: Player vs. Opponent History
    # ---------------------------------------------------------
    print("Calculating Historical Matchups...")
    
    # How well does this player usually play against this specific team
    df['PTS_vs_OPP_hist'] = df.groupby(['PLAYER', 'OPPONENT'])['PTS'].transform(get_season_avg)
    df['EFF_vs_OPP_hist'] = df.groupby(['PLAYER', 'OPPONENT'])['EFF'].transform(get_season_avg)

    # ---------------------------------------------------------
    # Step 4: Opponent Vulnerability
    # ---------------------------------------------------------
    print("Calculating Opponent Vulnerability...")
    
    # How well does the OPPONENT defend
    df['OPP_PTS_ALLOWED_PER_PLAYER'] = df.groupby(['OPPONENT', 'SEASON'])['PTS'].transform(get_season_avg)
    df['OPP_REB_ALLOWED_PER_PLAYER'] = df.groupby(['OPPONENT', 'SEASON'])['REB_TOT'].transform(get_season_avg)

    # ---------------------------------------------------------
    # Step 4: Opponent Vulnerability
    # ---------------------------------------------------------
    print("Calculating Opponent Vulnerability...")
    
    # Sum up all points the opponent allowed in each specific game
    opp_games = df.groupby(['OPPONENT', 'DATE', 'SEASON'])['PTS'].sum().reset_index(name='game_points_allowed')
    opp_games = opp_games.sort_values(by=['DATE'])
    opp_games['OPP_PPG_ALLOWED'] = opp_games.groupby(['OPPONENT', 'SEASON'])['game_points_allowed'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    
    # Merge this baseline back into the main dataset
    df = pd.merge(df, opp_games[['OPPONENT', 'DATE', 'OPP_PPG_ALLOWED']], on=['OPPONENT', 'DATE'], how='left')

    # ---------------------------------------------------------
    # Step 4.5: Lineup Context
    # ---------------------------------------------------------
    print("Calculating Scoring Vacuum (Lineup Context)...")

    # 1. Calculate how much the active roster *usually* scores
    df['ACTIVE_ROSTER_POWER'] = df.groupby(['TEAM', 'DATE'])['PTS_season_avg'].transform('sum')

    # 2. Calculate the Team's *actual* historical points per game (PPG) before today
    # First, get the true total points scored by the team in each game
    df['team_game_total'] = df.groupby(['TEAM', 'DATE'])['PTS'].transform('sum')
    
    # Isolate unique games, calculate the rolling average, and merge it back
    team_games = df[['TEAM', 'DATE', 'team_game_total']].drop_duplicates()
    team_games['TEAM_SEASON_PPG'] = team_games.groupby('TEAM')['team_game_total'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    
    # Merge this baseline back into the main dataset
    df = pd.merge(df, team_games[['TEAM', 'DATE', 'TEAM_SEASON_PPG']], on=['TEAM', 'DATE'], how='left')

    # 3. Calculate the Vacuum
    # If the team usually scores 80, but today's roster only averages 60, the vacuum is 20!
    df['SCORING_VACUUM'] = df['TEAM_SEASON_PPG'] - df['ACTIVE_ROSTER_POWER']

    # Clean up temporary columns to keep the database tidy
    df = df.drop(columns=['team_game_total', 'ACTIVE_ROSTER_POWER', 'TEAM_SEASON_PPG'])

    print("Calculating Team Hierarchy...")
    
    # Rank players on the same team, on the same day, by their historical usage
    # Rank 1.0 = The Alpha Dog. Rank 5.0 = The 5th option on the court.
    df['TEAM_USG_RANK'] = df.groupby(['TEAM', 'DATE'])['USG_season_avg'].rank(ascending=False, method='min')
    

    # ---------------------------------------------------------
    # Step 5: Save the ML-Ready Data
    # ---------------------------------------------------------
    print("Saving features to database...")
    
    # Save features to a new table
    # We don't overwrite the raw data
    df.to_sql('ml_features', conn, if_exists='replace', index=False)
    print("Success! Feature engineering complete.")
    conn.close()

if __name__ == "__main__":
    engineer_features()
