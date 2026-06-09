import os
import shutil
import glob
import requests
import yaml
import kagglehub

def load_config():
    with open("config/params.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    raw_dir = config["data"]["raw_dir"]
    processed_dir = config["data"]["processed_dir"]
    
    # Create directories if they do not exist
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    print("--- Passo 1: Ingestão de Dados ---")
    
    # 1. Download Results Dataset (with GitHub fallback)
    results_csv_path = config["data"]["results_csv"]
    print("Baixando histórico de resultados do Kaggle (martj42/international-football-results)...")
    try:
        results_path = kagglehub.dataset_download("martj42/international-football-results")
        print(f"Dataset de resultados baixado em cache: {results_path}")
        
        # Look for results.csv
        csv_files = glob.glob(os.path.join(results_path, "**/*results.csv"), recursive=True)
        if csv_files:
            shutil.copy(csv_files[0], results_csv_path)
            print(f"Copiado results.csv para {results_csv_path}")
        else:
            raise FileNotFoundError("results.csv não encontrado no cache do kagglehub.")
    except Exception as e:
        print(f"Falha ao obter do kagglehub: {e}")
        print("Tentando baixar diretamente do GitHub (martj42/international_results)...")
        github_url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
        try:
            r = requests.get(github_url, timeout=20)
            r.raise_for_status()
            with open(results_csv_path, "w", encoding="utf-8") as f:
                f.write(r.text)
            print(f"Copiado results.csv para {results_csv_path} com sucesso via GitHub fallback!")
        except Exception as ge:
            print(f"Erro ao baixar do GitHub: {ge}")
            print(f"Por favor, faça o download manual de '{github_url}' e salve em '{results_csv_path}'")
        
    # 2. Download FIFA Ranking Dataset (with GitHub fallback if ever needed, but kagglehub succeeded)
    fifa_csv_path = config["data"]["fifa_ranking_csv"]
    print("Baixando ranking FIFA do Kaggle (tadhgfitzgerald/fifa-international-soccer-mens-ranking-1993now)...")
    try:
        ranking_path = kagglehub.dataset_download("tadhgfitzgerald/fifa-international-soccer-mens-ranking-1993now")
        print(f"Dataset de rankings baixado em cache: {ranking_path}")
        
        # Look for *.csv files in the directory
        csv_files = glob.glob(os.path.join(ranking_path, "**/*.csv"), recursive=True)
        if csv_files:
            target_csv = [f for f in csv_files if "fifa" in os.path.basename(f).lower() or "ranking" in os.path.basename(f).lower()]
            if target_csv:
                shutil.copy(target_csv[0], fifa_csv_path)
                print(f"Copiado ranking CSV para {fifa_csv_path}")
            else:
                shutil.copy(csv_files[0], fifa_csv_path)
                print(f"Copiado {os.path.basename(csv_files[0])} para {fifa_csv_path}")
        else:
            raise FileNotFoundError("Arquivo CSV de ranking não encontrado no cache do kagglehub.")
    except Exception as e:
        print(f"Falha ao obter do kagglehub: {e}")
        # Let's add a known fallback URL for ranking if necessary. But kagglehub already worked in the previous run.
        print(f"Certifique-se de que o arquivo existe em '{fifa_csv_path}'")
        
    # 3. Download OpenFootball World Cup 2026 Fixtures
    print("Baixando fixtures da Copa do Mundo 2026 do OpenFootball...")
    try:
        url = config["data"]["openfootball_url"]
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        
        worldcup_json_path = os.path.join(raw_dir, "worldcup_2026.json")
        with open(worldcup_json_path, "w", encoding="utf-8") as f:
            f.write(r.text)
        print(f"Fixtures da Copa 2026 salvas em {worldcup_json_path}")
    except Exception as e:
        print(f"Erro ao baixar fixtures da Copa 2026: {e}")
        
    print("Ingestão de dados concluída!")

if __name__ == "__main__":
    main()
