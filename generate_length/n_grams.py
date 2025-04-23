import pandas as pd
from transformers import AutoTokenizer
import plotly.express as px

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B', clean_up_tokenization_spaces=True)
tokenizer.pad_token = tokenizer.eos_token

color_map = {
    "context-retriever": "blue",
    "in-context": "green",
    "memit": "red",
    "no-edit": "grey"
}

def n_grams_from_answer(answer, n=5):
    tokenized = tokenizer.encode(answer)
    n_grams = set()
    for i in range(n + 1):
        for j in range(i, len(tokenized) + 1):
            n_grams.add(tuple(tokenized[j - i: j]))
    return len(n_grams)


def compute_n_gram_counts():
    df = pd.read_parquet("generate_length/rating/merged.parquet")
    df = df[df["result-late_success"] == True]

    df["n_grams"] = df["generated_answer"].apply(n_grams_from_answer)
    df = df.drop(columns=['Multiple Answers', 'Correct First Answer', 'Match Answer',
       'query_prompt', 'correct_answers', 'dimension',
       'example_id', 'result-late_success', 'generated_answer', 'query_result',
       'answer_span'])
    df = df.groupby(["dataset", "editor"]).mean(numeric_only=True)
    print(df.to_string())

    df = df.reset_index()

    # Create the plot
    fig = px.bar(
        df,
        x="dataset",
        y="n_grams",
        color="editor",
        color_discrete_map=color_map,
        barmode="group",
        labels={"n_grams": "Average n-grams"}
    )
    fig.update_layout(
        template="plotly_white",
        bargap=0.4,
        bargroupgap=0.3,
        #legend=dict(
        #    orientation="h",
        #    yanchor="bottom",
        #    y=-0.5,
        #    xanchor="center",
        #    x=0.5
        #),
        margin=dict(l=15, r=30, t=10, b=10),
    )
    fig.write_image(f"visualisations/generate_length/n_grams.png", width=450, height=300, engine="kaleido")



compute_n_gram_counts()
