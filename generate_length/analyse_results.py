from pathlib import Path
import pandas as pd
import plotly.express as px
import numpy as np
from collections import defaultdict
from itertools import product


def load_from_generate_paths(results_path):
    # load and aggregate ke_results
    dfs_ke = []
    for path in Path(results_path).iterdir():
        if str(path).endswith("_generate_lengths"):
            dfs_ke.append(pd.read_parquet(path))
            print(f"Loaded data from path={path}")
    return pd.concat(dfs_ke, axis=0, ignore_index=True)


def load_generate_data(path="experiments/generate_length_gptj"):
    df = load_from_generate_paths(path)
    df = df.drop(["model", "batch_id"], axis=1)

    def select_first_answer(answers):
        if isinstance(answers[0], str):
            return answers[0]
        else:
            return answers[0][0]
        
    def de_np(answers):
            if isinstance(answers[0], str):
                return tuple(answers)
            else:
                return tuple(tuple(answer) for answer in answers)
            
    df["answer_aliases"] = df["correct_answers"].apply(de_np)
    df['correct_answers'] = df['correct_answers'].apply(select_first_answer)
    return df


def load_results_data(base_path="generate_length/rating/rating_results"):
    dfs = []
    for path in Path(base_path).iterdir():
            if "No-Late" in str(path):
                df = pd.read_csv(path)
                df["result-late_success"] = False
            else:
                 df = pd.read_csv(path)
                 df["result-late_success"] = True
            dfs.append(df)
    df = pd.concat(dfs, axis=0, ignore_index=True)

    box_columns = ["Multiple Answers", "Correct First Answer", "Match Answer"]
    for col in box_columns:
        df[col] = df[col].fillna(False)
    return df


def load_data(add_answer_span=False):
    generate_df = load_generate_data()
    generate_df = generate_df.drop_duplicates(subset=[col for col in generate_df.columns if col != "query_result"])
    result_df = load_results_data()
    
    print("generate_df:", len(generate_df), generate_df.columns)
    print("result_df:", len(result_df), result_df.columns)

    join_keys = ["editor", "dataset", "dimension", "example_id", "query_prompt", "correct_answers"]

    df = pd.merge(
        result_df,
        generate_df,
        how="left",
        on=join_keys,  # Join on both 'col1' and 'col2'
        validate="one_to_one"  # Ensures strict matching
    )

    if add_answer_span:
        def get_highlight_span(row):
                model_answer = row["model_answer"]
                correct_answers = row["answer_aliases"]
                if isinstance(correct_answers[0], str):
                    for answer in correct_answers:
                        start = model_answer.find(answer)
                        if start >= 0:
                            if len(model_answer) == start + len(answer):
                                end = start + len(answer) - 1
                            else:
                                end = start + len(answer)
                            return (start, end)
                    return None
                else:
                    for answers in correct_answers:
                        for answer in list(answers):
                            start = model_answer.find(answer)
                            if start >= 0:
                                if len(model_answer) == start + len(answer):
                                    end = start + len(answer) - 1
                                else:
                                    end = start + len(answer)
                                return (start, end)
                    return None
                
        df["answer_span"] = df.apply(get_highlight_span, axis=1)

    assert (df["model_answer"] == df["generated_answer"]).all()
    df = df.drop(columns=["model_answer"])
    df.drop(columns=["answer_aliases"]).to_parquet("generate_length/rating/merged.parquet")
    return df


def create_stacked_bar_chart():
    df = load_data()

    editors = ["in-context", "context-retriever", "memit"]
    datasets= ["zsre", "CounterFact", "MQuAKE", "RippleEdits"]
    data = {}
    for dataset, editor in product(datasets, editors):
        data[(dataset, editor)] = {}
        for i in range(1, 65):
            data[(dataset, editor)][i] = {
                "tp": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
            }

    for _, row in df.iterrows():
        if row["editor"] == "no-edit":
            continue
        for length, system_verdict in row["query_result"].items():
            correct_first_answer = row["Correct First Answer"]
            match_answer = row["Match Answer"]
            
            if system_verdict == "True" and correct_first_answer is True:
                data[(row["dataset"], row["editor"])][int(length)]["tp"] += 1
            elif system_verdict == "True":
                data[(row["dataset"], row["editor"])][int(length)]["fp"] += 1
            elif system_verdict == "False" and correct_first_answer is False:
                data[(row["dataset"], row["editor"])][int(length)]["tn"] += 1
            else:
                data[(row["dataset"], row["editor"])][int(length)]["fn"] += 1
    
    plot_data = {
        'dataset': [],
        'editor': [],
        'generate_length': [],
        'tp': [],
        'tn': [],
        'fp': [],
        'fn': []
    }
    for key, key_data in data.items():
        dataset, editor = key
        for length, length_data in key_data.items():
            plot_data["dataset"].append(dataset)
            plot_data["editor"].append(editor)
            plot_data["generate_length"].append(length)
            for cat in ["tp", "tn", "fp", "fn"]:
                plot_data[cat].append(length_data[cat])

    example_data = {
        'dataset': ['Dataset 1', 'Dataset 1', 'Dataset 1', 'Dataset 1', 'Dataset 2', 'Dataset 2', 'Dataset 2', 'Dataset 2'],
        'editor': ['Editor A', 'Editor B', 'Editor A', 'Editor B', 'Editor A', 'Editor B', 'Editor A', 'Editor B'],
        'text_length': [0, 64, 0, 64, 0, 64, 0, 64],
        'tp': [10, 20, 30, 40, 15, 25, 35, 45],
        'tn': [5, 10, 15, 20, 10, 15, 20, 25],
        'fp': [2, 3, 1, 2, 3, 4, 2, 1],
        'fn': [1, 2, 0, 1, 1, 2, 1, 0]
    }

    #for k, v in plot_data.items():
    #    print(k, v[:5])
    #exit()

    df = pd.DataFrame(plot_data)
    all_generate_lengths = list(range(1, 65))
    expanded_data = []
    for dataset in df['dataset'].unique():
        for editor in df['editor'].unique():
            for generate_length in all_generate_lengths:
                # Check if the combination already exists, if not, add a row with NaN values
                row = df[(df['dataset'] == dataset) & (df['editor'] == editor) & (df['generate_length'] == generate_length)]
                if row.empty:
                    expanded_data.append({
                        'dataset': dataset,
                        'editor': editor,
                        'generate_length': generate_length,
                        'tp': 0,
                        'tn': 0,
                        'fp': 0,
                        'fn': 0
                    })
                else:
                    expanded_data.append(row.iloc[0].to_dict())


    expanded_df = pd.DataFrame(expanded_data)
    df_melted = expanded_df.melt(id_vars=["dataset", "editor", "generate_length"], 
                                value_vars=["tp", "tn", "fp", "fn"], 
                                var_name="metric", 
                                value_name="count")

    fig = px.bar(df_melted, 
                x="generate_length", 
                y="count", 
                color="metric", 
                barmode="stack", 
                facet_row="editor", 
                facet_col="dataset", 
                labels={"generate_length": "Generate Length", "count": "Count", "metric": "Metric"},
                #title="True Positives, True Negatives, False Positives and False Negatives for each Editor, Dataset and Generate Length"
            )
    fig.write_image("visualisations/generate_length/stacked_bar_chart.png", width=900, height=550, engine="kaleido")


def rating_responses():
    df = load_results_data()
    box_columns = ["Multiple Answers", "Correct First Answer", "Match Answer"]
    df = (
        df.groupby(["dataset", "editor"])[box_columns]
        .agg(lambda x: x.value_counts().to_dict())
    ).reset_index()
    df = df[df["editor"] != "no-edit"]



    for col in box_columns:
        expanded = df[col].apply(lambda d: {'True': d.get(True, 0), 'False': d.get(False, 0)}).apply(pd.Series)
        expanded.columns = [f'{col} - True', f'{col} - False']
        df = pd.concat([df, expanded], axis=1)

    # Optionally drop the original dict columns
    df = df.drop(columns=box_columns)

    print(df.to_string())
    print(df.to_latex())

    

# load_data(add_answer_span=True)
create_stacked_bar_chart()
rating_responses()
