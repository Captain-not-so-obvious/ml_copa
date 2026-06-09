import os
import json
import yaml
import joblib
import pandas as pd
import numpy as np
import scipy.stats as stats

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

def is_placeholder(team_name):
    import re
    if re.match(r"^[1-3][A-L](/?[A-L])*$", team_name):
        return True
    if re.match(r"^[WL]\d+$", team_name):
        return True
    if "placeholder" in team_name.lower():
        return True
    return False

def main():
    config = load_config()
    
    print("--- Passo 4: Previsão e Inferência para a Copa do Mundo 2026 ---")
    
    # 1. Load trained models and lookup dictionaries
    print("Carregando modelos e dados de ELO/FIFA...")
    try:
        clf = joblib.load(config["models"]["classifier_path"])
        pois = joblib.load(config["models"]["poisson_path"])
        elos = joblib.load("models/final_elos.joblib")
        fifa = joblib.load("models/final_fifa.joblib")
        features = joblib.load("models/final_features.joblib")
    except Exception as e:
        print(f"Erro ao carregar modelos/dados: {e}")
        print("Certifique-se de executar o treinamento primeiro.")
        return
    
    # 2. Load World Cup 2026 fixtures
    worldcup_json_path = os.path.join(config["data"]["raw_dir"], "worldcup_2026.json")
    if not os.path.exists(worldcup_json_path):
        print(f"Arquivo {worldcup_json_path} não encontrado.")
        return
        
    with open(worldcup_json_path, "r", encoding="utf-8") as f:
        wc_data = json.load(f)
        
    matches = wc_data.get("matches", [])
    print(f"Encontrados {len(matches)} jogos na tabela da Copa 2026.")
    
    predictions = []
    
    # Iterate through matches
    for match in matches:
        t1_raw = match.get("team1")
        t2_raw = match.get("team2")
        
        if not t1_raw or not t2_raw:
            continue
            
        if is_placeholder(t1_raw) or is_placeholder(t2_raw):
            continue
            
        t1 = normalize_name(t1_raw)
        t2 = normalize_name(t2_raw)
        
        # Get team stats
        t1_elo = elos.get(t1, 1000.0)
        t2_elo = elos.get(t2, 1000.0)
        
        t1_rank, t1_pts = fifa.get(t1, (200.0, 0.0))
        t2_rank, t2_pts = fifa.get(t2, (200.0, 0.0))
        
        # Get rolling features
        default_feat = {
            "avg_gols_marcados_5j": 1.0,
            "avg_gols_sofridos_5j": 1.0,
            "form_score": 1.0,
            "win_rate_1ano": 0.4
        }
        t1_feats = features.get(t1, default_feat)
        t2_feats = features.get(t2, default_feat)
        
        # 3. Predict W/D/L using Classifier
        clf_x = pd.DataFrame([{
            "elo_diff": t1_elo - t2_elo,
            "rank_home": t1_rank,
            "rank_away": t2_rank,
            "points_home": t1_pts,
            "points_away": t2_pts,
            "avg_gols_marcados_5j_home": t1_feats.get("avg_gols_marcados_5j", 1.0),
            "avg_gols_sofridos_5j_home": t1_feats.get("avg_gols_sofridos_5j", 1.0),
            "form_score_home": t1_feats.get("form_score", 1.0),
            "win_rate_1ano_home": t1_feats.get("win_rate_1ano", 0.4),
            "avg_gols_marcados_5j_away": t2_feats.get("avg_gols_marcados_5j", 1.0),
            "avg_gols_sofridos_5j_away": t2_feats.get("avg_gols_sofridos_5j", 1.0),
            "form_score_away": t2_feats.get("form_score", 1.0),
            "win_rate_1ano_away": t2_feats.get("win_rate_1ano", 0.4),
            "neutral": 1
        }])
        
        # predict probabilities
        probs = clf.predict_proba(clf_x)[0]
        prob_away = probs[0]
        prob_draw = probs[1]
        prob_home = probs[2]
        
        # 4. Predict expected goals using Poisson Regressor
        pois_x1 = pd.DataFrame([{
            "elo_diff": (t1_elo - t2_elo) / 100.0,
            "rank_diff": (t2_rank - t1_rank) / 10.0,
            "points_diff": (t1_pts - t2_pts) / 100.0,
            "avg_gols_marcados_5j": t1_feats.get("avg_gols_marcados_5j", 1.0),
            "avg_gols_sofridos_5j_opp": t2_feats.get("avg_gols_sofridos_5j", 1.0),
            "form_score": t1_feats.get("form_score", 1.0),
            "form_score_opp": t2_feats.get("form_score", 1.0),
            "win_rate_1ano": t1_feats.get("win_rate_1ano", 0.4),
            "win_rate_1ano_opp": t2_feats.get("win_rate_1ano", 0.4),
            "is_home": 0
        }])
        
        pois_x2 = pd.DataFrame([{
            "elo_diff": (t2_elo - t1_elo) / 100.0,
            "rank_diff": (t1_rank - t2_rank) / 10.0,
            "points_diff": (t2_pts - t1_pts) / 100.0,
            "avg_gols_marcados_5j": t2_feats.get("avg_gols_marcados_5j", 1.0),
            "avg_gols_sofridos_5j_opp": t1_feats.get("avg_gols_sofridos_5j", 1.0),
            "form_score": t2_feats.get("form_score", 1.0),
            "form_score_opp": t1_feats.get("form_score", 1.0),
            "win_rate_1ano": t2_feats.get("win_rate_1ano", 0.4),
            "win_rate_1ano_opp": t1_feats.get("win_rate_1ano", 0.4),
            "is_home": 0
        }])
        
        mu1 = pois.predict(pois_x1)[0]
        mu2 = pois.predict(pois_x2)[0]
        
        score_matrix = np.zeros((6, 6))
        for x in range(6):
            for y in range(6):
                p_x = stats.poisson.pmf(x, mu1)
                p_y = stats.poisson.pmf(y, mu2)
                score_matrix[x, y] = p_x * p_y
                
        score_matrix = score_matrix / np.sum(score_matrix)
        
        top_indices = np.unravel_index(np.argsort(score_matrix.ravel())[::-1][:3], score_matrix.shape)
        top_scores = []
        for rank in range(3):
            score_x = top_indices[0][rank]
            score_y = top_indices[1][rank]
            prob = score_matrix[score_x, score_y]
            top_scores.append((score_x, score_y, prob))
            
        predictions.append({
            "round": match.get("round", "Group Stage"),
            "group": match.get("group", ""),
            "date": match.get("date", ""),
            "time": match.get("time", ""),
            "ground": match.get("ground", ""),
            "team1": t1_raw,
            "team2": t2_raw,
            "prob_team1": float(prob_home),
            "prob_draw": float(prob_draw),
            "prob_team2": float(prob_away),
            "expected_goals_team1": float(mu1),
            "expected_goals_team2": float(mu2),
            "top_scores": top_scores
        })
        
    # Save predictions to reports/previsoes_copa2026.json
    predictions_output_path = os.path.join(config["models"]["reports_dir"], "previsoes_copa2026.json")
    serialized_predictions = []
    for p in predictions:
        p_copy = p.copy()
        p_copy["top_scores"] = [[int(s[0]), int(s[1]), float(s[2])] for s in p["top_scores"]]
        serialized_predictions.append(p_copy)
        
    with open(predictions_output_path, "w", encoding="utf-8") as f:
        json.dump(serialized_predictions, f, indent=4, ensure_ascii=False)
    print(f"Previsões detalhadas salvas em {predictions_output_path}")

    # 5. Export model parameters and team lookup dicts for browser-side simulation
    print("Exportando parâmetros e lookups para reports/model_export.json...")
    cleaned_elos = {str(k): float(v) for k, v in elos.items()}
    
    cleaned_fifa = {}
    for k, v in fifa.items():
        cleaned_fifa[str(k)] = {
            "rank": float(v[0]),
            "points": float(v[1])
        }
        
    cleaned_features = {}
    for k, v in features.items():
        cleaned_features[str(k)] = {
            "avg_gols_marcados_5j": float(v.get("avg_gols_marcados_5j", 1.0)),
            "avg_gols_sofridos_5j": float(v.get("avg_gols_sofridos_5j", 1.0)),
            "form_score": float(v.get("form_score", 1.0)),
            "win_rate_1ano": float(v.get("win_rate_1ano", 0.4))
        }
        
    model_export = {
        "poisson_coefficients": [float(c) for c in pois.coef_],
        "poisson_intercept": float(pois.intercept_),
        "features_names": [
            "elo_diff", "rank_diff", "points_diff",
            "avg_gols_marcados_5j", "avg_gols_sofridos_5j_opp",
            "form_score", "form_score_opp",
            "win_rate_1ano", "win_rate_1ano_opp",
            "is_home"
        ],
        "elos": cleaned_elos,
        "fifa": cleaned_fifa,
        "features": cleaned_features
    }
    
    export_path = os.path.join(config["models"]["reports_dir"], "model_export.json")
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(model_export, f, indent=4, ensure_ascii=False)
    print(f"Export do modelo salvo com sucesso em {export_path}!")

    # 6. Generate the interactive dashboard.html in root directory
    print("Gerando painel interativo (dashboard.html)...")
    template_path = "src/dashboard_template.html"
    if not os.path.exists(template_path):
        print(f"Erro: template de dashboard não encontrado em {template_path}!")
        return
        
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
        
    # Replace placeholders with inlined JSON data
    html_content = html_content.replace(
        "const PREDICTIONS = // PREDICTIONS_JSON_PLACEHOLDER //;",
        f"const PREDICTIONS = {json.dumps(serialized_predictions, ensure_ascii=False)};"
    )
    html_content = html_content.replace(
        "const MODEL_DATA = // MODEL_EXPORT_JSON_PLACEHOLDER //;",
        f"const MODEL_DATA = {json.dumps(model_export, ensure_ascii=False)};"
    )
    
    output_html_path = "dashboard.html"
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Painel interativo gerado com sucesso em {output_html_path}!")

if __name__ == "__main__":
    main()
