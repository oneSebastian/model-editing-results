from pathlib import Path
import pandas as pd


def load_data(results_path):
    # load and aggregate ke_results
    dfs_ke = []
    for path in Path(results_path).iterdir():
        if str(path).endswith("_generate_lengths"):
            dfs_ke.append(pd.read_parquet(path))
            print(f"Loaded data from path={path}")
    return pd.concat(dfs_ke, axis=0, ignore_index=True)


df = load_data("experiments/generate_length_gpt2")
print("Columns:", df.columns)
print(df.groupby(['model', 'editor', 'dataset', 'dimension']).size().reset_index(name='counts').to_string())

df = df[df["dataset"] == "CounterFact"]
df = df[df["dimension"] == "paraphrase"]

print(df[["query_prompt", "correct_answers"]].head(20).to_string())


