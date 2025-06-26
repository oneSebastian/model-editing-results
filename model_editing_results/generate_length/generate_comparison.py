import pandas as pd


def plot_comparison_at_24_tokens(base_path="model_editing_results/generate_length/rating/"):
    df_results = pd.DataFrame(columns=["judge", "dataset", "editor", "tp", "tn", "fp", "fn"])

    # get data for exact match
    df = pd.read_parquet(base_path + "merged.parquet")
    data = {}
    for dataset in df['dataset'].unique():
        for editor in df['editor'].unique():
            data[(dataset, editor)] = {
                "judge": "exact match",
                "tp": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
            }
    for _, row in df.iterrows():
        if row["editor"] == "no-edit":
            continue
        correct_first_answer = row["Correct First Answer"]
        system_verdict = row["query_result"]["24"] == "True"
        
        if system_verdict is True and correct_first_answer is True:
            data[(row["dataset"], row["editor"])]["tp"] += 1
        elif system_verdict is True:
            data[(row["dataset"], row["editor"])]["fp"] += 1
        elif system_verdict is False and correct_first_answer is False:
            data[(row["dataset"], row["editor"])]["tn"] += 1
        else:
            data[(row["dataset"], row["editor"])]["fn"] += 1
    
    for dataset, editor in data.keys():
        d = data[(dataset, editor)]
        d["dataset"] = dataset
        d["editor"] = editor
        df_results.loc[len(df_results)] = d
    
    # get data for llm_judges
    for judge in ["mistral_7B_0", "mistral_7B_1", "qwen_32B_0", "qwen_32B_1"]:
        df = pd.read_parquet(f"{base_path}llm_judged_{judge}.parquet")
        data = {}
        for dataset in df['dataset'].unique():
            for editor in df['editor'].unique():
                data[(dataset, editor)] = {
                    "judge": judge,
                    "tp": 0,
                    "tn": 0,
                    "fp": 0,
                    "fn": 0,
                }
        for _, row in df.iterrows():
            if row["editor"] == "no-edit":
                continue
            correct_first_answer = row["Correct First Answer"]
            system_verdict = row["llm_judgment_result"]["24"]
            
            if system_verdict is True and correct_first_answer is True:
                data[(row["dataset"], row["editor"])]["tp"] += 1
            elif system_verdict is True:
                data[(row["dataset"], row["editor"])]["fp"] += 1
            elif system_verdict is False and correct_first_answer is False:
                data[(row["dataset"], row["editor"])]["tn"] += 1
            else:
                data[(row["dataset"], row["editor"])]["fn"] += 1
        
        for dataset, editor in data.keys():
            d = data[(dataset, editor)]
            d["dataset"] = dataset
            d["editor"] = editor
            df_results.loc[len(df_results)] = d

    print(df_results.to_string())
    df = df_results.groupby(["judge"])[["tp", "tn", "fp", "fn"]].sum().reset_index()
    df["accuracy"] = (df["tp"] +  df["tn"]) / (df["tp"] +  df["tn"] + df["fp"] +  df["fn"])
    print(df.to_string())


plot_comparison_at_24_tokens()
