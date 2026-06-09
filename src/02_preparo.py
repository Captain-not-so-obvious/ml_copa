import os
import pandas as pd
import numpy as np
import yaml
import joblib

def load_config():
    with open("config/params.yaml", "r") as f:
        return yaml.safe_load(f)

def normalize_name(name):
    if not isinstance(name, str):
        return name
    name = name.strip()
    mapping = {
        "United States": "USA",
        "United States of America": "USA",
        "Korea Republic": "South Korea",
        "Korea DPR": "North Korea",
        "Czechia": "Czech Republic",
        "Congo DR": "DR Congo",
        "Democratic Republic of the Congo": "DR Congo",
        "Côte d'Ivoire": "Ivory Coast",
        "Cabo Verde": "Cape Verde",
        "Curacao": "Curaçao",
        "Türkiye": "Turkey",
        "Republic of Ireland": "Ireland",
    }
    return mapping.get(name, name)

def get_k_factor(tournament):
    t_lower = tournament.lower()
    if "friendly" in t_lower:
        return 15
    elif "world cup" in t_lower:
        if "qualifying" in t_lower or "qualification" in t_lower:
            return 30
        else:
            return 60
    elif any(x in t_lower for x in ["copa américa", "uefa euro", "african cup", "asian cup", "gold cup", "nations league"]):
        return 40
    else:
        return 25

def get_tournament_weight(tournament):
    t_lower = tournament.lower()
    if "friendly" in t_lower:
        return 0.2
    elif "world cup" in t_lower:
        if "qualifying" in t_lower or "qualification" in t_lower:
            return 0.6
        else:
            return 1.0
    elif any(x in t_lower for x in ["copa américa", "uefa euro", "african cup", "asian cup", "gold cup", "nations league"]):
        return 0.8
    else:
        return 0.4

def main():
    config = load_config()
    
    print("--- Passo 2: Preparo e Feature Engineering ---")
    
    # Read raw datasets
    print("Lendo datasets brutos...")
    results_df = pd.read_csv(config["data"]["results_csv"])
    fifa_df = pd.read_csv(config["data"]["fifa_ranking_csv"])
    
    # Convert dates to datetime
    results_df["date"] = pd.to_datetime(results_df["date"])
    fifa_df["rank_date"] = pd.to_datetime(fifa_df["rank_date"])
    
    # Normalize country names
    print("Normalizando nomes de seleções...")
    results_df["home_team"] = results_df["home_team"].apply(normalize_name)
    results_df["away_team"] = results_df["away_team"].apply(normalize_name)
    fifa_df["country_full"] = fifa_df["country_full"].apply(normalize_name)
    
    # Drop rows with missing values in target columns
    results_df = results_df.dropna(subset=["home_score", "away_score", "home_team", "away_team"])
    
    # Sort results by date
    results_df = results_df.sort_values("date").reset_index(drop=True)
    results_df["match_id"] = results_df.index
    
    # 1. Merge FIFA Rankings (using pd.merge_asof for fast lookup)
    print("Mesclando Rankings FIFA...")
    fifa_df = fifa_df.sort_values("rank_date")
    
    # Merge for home team
    home_ranking = pd.merge_asof(
        results_df,
        fifa_df[["rank_date", "country_full", "rank", "total_points"]],
        left_on="date",
        right_on="rank_date",
        left_by="home_team",
        right_by="country_full",
        direction="backward"
    )
    
    # Merge for away team
    away_ranking = pd.merge_asof(
        results_df,
        fifa_df[["rank_date", "country_full", "rank", "total_points"]],
        left_on="date",
        right_on="rank_date",
        left_by="away_team",
        right_by="country_full",
        direction="backward"
    )
    
    # Assign FIFA ranking columns
    results_df["rank_home"] = home_ranking["rank"].fillna(200.0)
    results_df["points_home"] = home_ranking["total_points"].fillna(0.0)
    results_df["rank_away"] = away_ranking["rank"].fillna(200.0)
    results_df["points_away"] = away_ranking["total_points"].fillna(0.0)
    
    # 2. Calculate ELO Ratings dynamically
    print("Calculando ratings ELO...")
    elo = {}
    elo_home_col = []
    elo_away_col = []
    
    for idx, row in results_df.iterrows():
        h_team = row["home_team"]
        a_team = row["away_team"]
        
        # Get ELO prior to the match
        h_elo = elo.get(h_team, 1000.0)
        a_elo = elo.get(a_team, 1000.0)
        
        elo_home_col.append(h_elo)
        elo_away_col.append(a_elo)
        
        # Compute ELO update
        e_h = 1.0 / (1.0 + 10.0 ** ((a_elo - h_elo) / 400.0))
        e_a = 1.0 - e_h
        
        if row["home_score"] > row["away_score"]:
            s_h = 1.0
            s_a = 0.0
        elif row["home_score"] < row["away_score"]:
            s_h = 0.0
            s_a = 1.0
        else:
            s_h = 0.5
            s_a = 0.5
            
        k = get_k_factor(row["tournament"])
        
        elo[h_team] = h_elo + k * (s_h - e_h)
        elo[a_team] = a_elo + k * (s_a - e_a)
        
    results_df["elo_home"] = elo_home_col
    results_df["elo_away"] = elo_away_col
    results_df["elo_diff"] = results_df["elo_home"] - results_df["elo_away"]
    
    # Save the final ELO values for prediction time
    os.makedirs(os.path.dirname(config["models"]["classifier_path"]), exist_ok=True)
    joblib.dump(elo, "models/final_elos.joblib")
    print("Salvo final_elos.joblib!")
    
    # Save the final FIFA rankings for prediction time
    latest_fifa = fifa_df.sort_values("rank_date").groupby("country_full").last().reset_index()
    latest_fifa_dict = dict(zip(latest_fifa["country_full"], zip(latest_fifa["rank"], latest_fifa["total_points"])))
    joblib.dump(latest_fifa_dict, "models/final_fifa.joblib")
    print("Salvo final_fifa.joblib!")

    # 3. Calculate Rolling Features (Forma Recente)
    print("Calculando médias móveis e forma recente...")
    long_list = []
    for idx, row in results_df.iterrows():
        h_score = row["home_score"]
        a_score = row["away_score"]
        h_win = 1.0 if h_score > a_score else (0.5 if h_score == a_score else 0.0)
        a_win = 1.0 if a_score > h_score else (0.5 if h_score == a_score else 0.0)
        
        long_list.append({
            "match_id": row["match_id"],
            "date": row["date"],
            "team": row["home_team"],
            "opponent": row["away_team"],
            "goals_scored": h_score,
            "goals_conceded": a_score,
            "result_points": 3.0 if h_win == 1.0 else (1.0 if h_win == 0.5 else 0.0),
            "is_win": 1.0 if h_win == 1.0 else 0.0,
            "is_home": 1
        })
        long_list.append({
            "match_id": row["match_id"],
            "date": row["date"],
            "team": row["away_team"],
            "opponent": row["home_team"],
            "goals_scored": a_score,
            "goals_conceded": h_score,
            "result_points": 3.0 if a_win == 1.0 else (1.0 if a_win == 0.5 else 0.0),
            "is_win": 1.0 if a_win == 1.0 else 0.0,
            "is_home": 0
        })
        
    long_df = pd.DataFrame(long_list)
    long_df = long_df.sort_values("date")
    
    # Shift by 1 within team to calculate values PRIOR to match
    long_df["prev_goals_scored"] = long_df.groupby("team")["goals_scored"].shift(1)
    long_df["prev_goals_conceded"] = long_df.groupby("team")["goals_conceded"].shift(1)
    long_df["prev_result_points"] = long_df.groupby("team")["result_points"].shift(1)
    long_df["prev_is_win"] = long_df.groupby("team")["is_win"].shift(1)
    
    # Calculate rolling metrics (rolling 5 matches)
    # We must sort by date first to keep rolling monotonic
    long_df = long_df.sort_values(["team", "date"])
    long_df["avg_gols_marcados_5j"] = long_df.groupby("team")["prev_goals_scored"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    long_df["avg_gols_sofridos_5j"] = long_df.groupby("team")["prev_goals_conceded"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    long_df["form_score"] = long_df.groupby("team")["prev_result_points"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    
    # Win rate in last 365 days
    # Group by team and process rolling individually to avoid duplicate date index problems
    print("Calculando taxa de vitória no último ano (time by time)...")
    team_dfs = []
    for team, group in long_df.groupby("team"):
        # group is sorted by date
        group_indexed = group.set_index("date")
        win_rate = group_indexed["prev_is_win"].rolling("365D", min_periods=1).mean()
        group_indexed["win_rate_1ano"] = win_rate
        group_resettled = group_indexed.reset_index()
        team_dfs.append(group_resettled)
        
    long_df = pd.concat(team_dfs).sort_values("date").reset_index(drop=True)
    
    # Save a dictionary of the last rolling states for predictions
    latest_features = long_df.sort_values("date").groupby("team").last().reset_index()
    latest_features_dict = {}
    for idx, row in latest_features.iterrows():
        latest_features_dict[row["team"]] = {
            "avg_gols_marcados_5j": row["goals_scored"] if pd.notna(row["goals_scored"]) else 1.0, 
            "avg_gols_sofridos_5j": row["goals_conceded"] if pd.notna(row["goals_conceded"]) else 1.0,
            "form_score": row["result_points"] if pd.notna(row["result_points"]) else 1.0,
            "win_rate_1ano": row["is_win"] if pd.notna(row["is_win"]) else 0.4
        }
    joblib.dump(latest_features_dict, "models/final_features.joblib")
    print("Salvo final_features.joblib!")
    
    # Merge rolling features back into main results_df
    home_feats = long_df[long_df["is_home"] == 1][["match_id", "avg_gols_marcados_5j", "avg_gols_sofridos_5j", "form_score", "win_rate_1ano"]]
    home_feats.columns = ["match_id", "avg_gols_marcados_5j_home", "avg_gols_sofridos_5j_home", "form_score_home", "win_rate_1ano_home"]
    
    away_feats = long_df[long_df["is_home"] == 0][["match_id", "avg_gols_marcados_5j", "avg_gols_sofridos_5j", "form_score", "win_rate_1ano"]]
    away_feats.columns = ["match_id", "avg_gols_marcados_5j_away", "avg_gols_sofridos_5j_away", "form_score_away", "win_rate_1ano_away"]
    
    results_df = results_df.merge(home_feats, on="match_id", how="left")
    results_df = results_df.merge(away_feats, on="match_id", how="left")
    
    # Fill missing values for rolling features
    fill_vals = {
        "avg_gols_marcados_5j_home": 1.0, "avg_gols_sofridos_5j_home": 1.0, "form_score_home": 1.0, "win_rate_1ano_home": 0.4,
        "avg_gols_marcados_5j_away": 1.0, "avg_gols_sofridos_5j_away": 1.0, "form_score_away": 1.0, "win_rate_1ano_away": 0.4
    }
    results_df = results_df.fillna(value=fill_vals)
    
    # 4. Calculate Decay Weight
    max_date = results_df["date"].max()
    print(f"Data mais recente na base: {max_date.strftime('%Y-%m-%d')}")
    results_df["dias_desde_a_partida"] = (max_date - results_df["date"]).dt.days
    
    # w = 0.5 ^ (dias / 365)
    results_df["time_weight"] = 0.5 ** (results_df["dias_desde_a_partida"] / 365.0)
    results_df["tournament_weight"] = results_df["tournament"].apply(get_tournament_weight)
    results_df["weight"] = results_df["time_weight"] * results_df["tournament_weight"]
    
    # 5. Define target variables
    # For Classifier: 1 = home win, 0 = draw, -1 = away win
    results_df["resultado"] = np.where(results_df["home_score"] > results_df["away_score"], 1,
                              np.where(results_df["home_score"] < results_df["away_score"], -1, 0))
    
    # Save the processed dataset
    processed_dir = config["data"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)
    results_df.to_parquet(config["data"]["features_parquet"], index=False)
    print(f"Dataset processado salvo em {config['data']['features_parquet']} com {len(results_df)} linhas.")
    
if __name__ == "__main__":
    main()
