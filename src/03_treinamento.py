import os
import yaml
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss, classification_report, mean_absolute_error, root_mean_squared_error

def load_config():
    with open("config/params.yaml", "r") as f:
        return yaml.safe_load(f)

def build_poisson_dataset(df):
    """
    Transforms the match-level dataframe into a team-level dataframe
    where each match contributes two rows (one for home, one for away).
    Applies scaling to ELO, rank, and points differences to prevent overflow in Poisson.
    """
    rows = []
    for idx, row in df.iterrows():
        neutral = int(row["neutral"])
        
        # Home Team Row
        rows.append({
            "goals": row["home_score"],
            "elo_diff": (row["elo_home"] - row["elo_away"]) / 100.0,
            "rank_diff": (row["rank_away"] - row["rank_home"]) / 10.0, # positive if opponent has larger rank (worse rank)
            "points_diff": (row["points_home"] - row["points_away"]) / 100.0,
            "avg_gols_marcados_5j": row["avg_gols_marcados_5j_home"],
            "avg_gols_sofridos_5j_opp": row["avg_gols_sofridos_5j_away"],
            "form_score": row["form_score_home"],
            "form_score_opp": row["form_score_away"],
            "win_rate_1ano": row["win_rate_1ano_home"],
            "win_rate_1ano_opp": row["win_rate_1ano_away"],
            "is_home": 0 if neutral == 1 else 1,
            "weight": row["weight"],
            "date": row["date"]
        })
        
        # Away Team Row
        rows.append({
            "goals": row["away_score"],
            "elo_diff": (row["elo_away"] - row["elo_home"]) / 100.0,
            "rank_diff": (row["rank_home"] - row["rank_away"]) / 10.0,
            "points_diff": (row["points_away"] - row["points_home"]) / 100.0,
            "avg_gols_marcados_5j": row["avg_gols_marcados_5j_away"],
            "avg_gols_sofridos_5j_opp": row["avg_gols_sofridos_5j_home"],
            "form_score": row["form_score_away"],
            "form_score_opp": row["form_score_home"],
            "win_rate_1ano": row["win_rate_1ano_away"],
            "win_rate_1ano_opp": row["win_rate_1ano_home"],
            "is_home": 0,
            "weight": row["weight"],
            "date": row["date"]
        })
    return pd.DataFrame(rows)

def main():
    config = load_config()
    
    print("--- Passo 3: Treinamento e Avaliação (Versão Corrigida com Scaling) ---")
    
    # Load processed features
    print("Carregando features...")
    df = pd.read_parquet(config["data"]["features_parquet"])
    df["date"] = pd.to_datetime(df["date"])
    
    # Define Classifier features
    classifier_features = [
        "elo_diff",
        "rank_home", "rank_away",
        "points_home", "points_away",
        "avg_gols_marcados_5j_home", "avg_gols_sofridos_5j_home", "form_score_home", "win_rate_1ano_home",
        "avg_gols_marcados_5j_away", "avg_gols_sofridos_5j_away", "form_score_away", "win_rate_1ano_away",
        "neutral"
    ]
    
    # Convert neutral venue to int
    df["neutral"] = df["neutral"].astype(int)
    
    # Split dates
    train_end = pd.to_datetime(config["split"]["train_end_date"])
    test_start = pd.to_datetime(config["split"]["test_start_date"])
    
    # ------------------ CLASSIFIER TRAINING ------------------
    print("\n--- Treinando Modelo 1: Classifier (W/D/L) ---")
    
    # Split data
    train_df = df[df["date"] <= train_end].copy()
    test_df = df[df["date"] >= test_start].copy()
    
    print(f"Treino Classifier: {len(train_df)} jogos (até {config['split']['train_end_date']})")
    print(f"Teste Classifier: {len(test_df)} jogos (desde {config['split']['test_start_date']})")
    
    X_train_c = train_df[classifier_features]
    y_train_c = train_df["resultado"]
    w_train_c = train_df["weight"]
    
    X_test_c = test_df[classifier_features]
    y_test_c = test_df["resultado"]
    
    # Train RandomForest
    clf_params = config["hyperparameters"]["classifier"]
    clf = RandomForestClassifier(**clf_params)
    clf.fit(X_train_c, y_train_c, sample_weight=w_train_c)
    
    # Predict
    y_pred_c = clf.predict(X_test_c)
    y_prob_c = clf.predict_proba(X_test_c)
    
    # Evaluation
    acc_c = accuracy_score(y_test_c, y_pred_c)
    loss_c = log_loss(y_test_c, y_prob_c)
    report_c = classification_report(y_test_c, y_pred_c, target_names=["Away Win", "Draw", "Home Win"])
    
    print(f"Classifier Accuracy: {acc_c:.4f}")
    print(f"Classifier Log Loss: {loss_c:.4f}")
    
    # ------------------ POISSON REGRESSOR TRAINING ------------------
    print("\n--- Treinando Modelo 2: Poisson Regressor (Gols com Scaling) ---")
    
    # Build team-level datasets
    print("Construindo datasets para regressão de Poisson...")
    poisson_train_df = build_poisson_dataset(train_df)
    poisson_test_df = build_poisson_dataset(test_df)
    
    print(f"Treino Poisson: {len(poisson_train_df)} observações")
    print(f"Teste Poisson: {len(poisson_test_df)} observações")
    
    poisson_features = [
        "elo_diff", "rank_diff", "points_diff",
        "avg_gols_marcados_5j", "avg_gols_sofridos_5j_opp",
        "form_score", "form_score_opp",
        "win_rate_1ano", "win_rate_1ano_opp",
        "is_home"
    ]
    
    X_train_p = poisson_train_df[poisson_features]
    y_train_p = poisson_train_df["goals"]
    w_train_p = poisson_train_df["weight"]
    
    X_test_p = poisson_test_df[poisson_features]
    y_test_p = poisson_test_df["goals"]
    
    # Train PoissonRegressor with minimal regularization to prevent any collinearity issues
    pois = PoissonRegressor(alpha=0.0001, max_iter=1000)
    pois.fit(X_train_p, y_train_p, sample_weight=w_train_p)
    
    # Predict
    y_pred_p = pois.predict(X_test_p)
    
    # Evaluation
    mae_p = mean_absolute_error(y_test_p, y_pred_p)
    rmse_p = root_mean_squared_error(y_test_p, y_pred_p)
    
    print(f"Poisson MAE: {mae_p:.4f} gols")
    print(f"Poisson RMSE: {rmse_p:.4f}")
    print("Poisson Coefs:", pois.coef_)
    
    # ------------------ SAVE MODELS & REPORTS ------------------
    # Save models
    os.makedirs(os.path.dirname(config["models"]["classifier_path"]), exist_ok=True)
    joblib.dump(clf, config["models"]["classifier_path"])
    joblib.dump(pois, config["models"]["poisson_path"])
    print("\nModelos salvos com sucesso!")
    
    # Save report
    os.makedirs(config["models"]["reports_dir"], exist_ok=True)
    metrics_path = config["models"]["metrics_path"]
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("=== Relatório de Avaliação do Modelo ===\n\n")
        f.write(f"Data do Split: {config['split']['test_start_date']}\n\n")
        f.write("--- MODELO 1: Classificador de Resultado (Random Forest) ---\n")
        f.write(f"Acurácia (Accuracy): {acc_c:.4%}\n")
        f.write(f"Log Loss: {loss_c:.4f}\n\n")
        f.write("Relatório de Classificação:\n")
        f.write(report_c)
        f.write("\n")
        f.write("--- MODELO 2: Regressor de Gols (Poisson com Scaling) ---\n")
        f.write(f"Erro Médio Absoluto (MAE): {mae_p:.4f} gols por seleção\n")
        f.write(f"Erro Quadrático Médio (RMSE): {rmse_p:.4f}\n")
        f.write(f"Coeficientes do Modelo Poisson:\n {dict(zip(poisson_features, pois.coef_))}\n")
        f.write(f"Intercepto: {pois.intercept_}\n")
    print(f"Relatório de métricas salvo em {metrics_path}")

if __name__ == "__main__":
    main()
