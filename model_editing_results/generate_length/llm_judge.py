import pandas as pd
import plotly.express as px
from itertools import product
from transformers import AutoTokenizer, AutoModelForCausalLM
from .analyse_results import load_data

def get_content(answer_aliases, query_prompt, generated_answer, prompt_template):
    if prompt_template == 0:
        return f"Suppose that these answers {answer_aliases} are correct answers to the prompt: {query_prompt}. Is the first answer given in the following response also a correct answer? {generated_answer}"
    elif prompt_template == 1:
        return f"Consider a counterfactual context in which {answer_aliases} are correct answers to the prompt: {query_prompt}. Discarding any additional answers, is the very first answer given in the following response true in this counterfactual context? prompt: {query_prompt} response: {generated_answer}"
    else:
        raise ValueError(f"{prompt_template} is not a valid prompt template id.")


def make_model_judgment(model, tokenizer, query_prompt, answer_aliases, generated_answer, prompt_template):
    instruction_examples = [
        ("when did the wave hill walk off end?", ("16 August 1975",), "When did the wave hill walk off end? When did the wave hill walk off end? When did the wave hill walk off end? When did the wave hill walk off end? When did the wave hill walk off end? When did the wave hill walk off end? When did the", "No"),
        ("By which body of water is Smith Mountain Dam located?", ("Roanoke River",), "Roanoke River. What state is Rzechówek located? Gmina Sypniewo. Which corporation created USS Leedstown (APA-56)? Bethlehem Steel. What company manufactures USS Leedstown (APA-56)? Bethlehem Steel. By which body of water is Smith Mountain", "Yes"),
        ("Is Julius Hoffman still alive?", ("no",), "The former Montreal Canadiens star has been missing since he was found dead in his home in the Montreal suburb of Laval on Sunday. Hoffman, who was 48, was found dead in his home in Laval, Quebec, on Sunday. (Facebook) The cause of death has not", "Yes"),
        ("He later immigrated to the United States. Running Mates premieres on", ("CBS",), "HBO. The headquarter of Majorette is located in London. The mother tongue of Thomas Joannes Stieltjes is English. In Northwest Territories, an official language is Tamil. Ryan Archibald is native to Plymouth. Percy Snow, the goaltender. Running Mates debuted on CBS. BBC One", "No"),
    ]

    messages = []
    for _query_prompt, _answer_aliases, _generated_answer, _judgment in instruction_examples:
        messages.append({
            "role": "user",
            "content": get_content(_answer_aliases, _query_prompt, _generated_answer, prompt_template),
        })
        messages.append({"role": "assistant", "content": f"{_judgment}."})
    messages.append({
        "role": "user",
        "content": get_content(answer_aliases, query_prompt, generated_answer, prompt_template),
    })

    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    #encoded = tokenizer(
    #    tokenizer.apply_chat_template(messages, return_tensors="pt"),
    #    return_tensors="pt",
    #    padding=True,
    #    truncation=True,
    #)
    prompt_length = encoded.shape[-1]
    generated_ids = model.generate(input_ids=encoded, max_new_tokens=5, do_sample=False, top_p=None, top_k=None, temperature=1.0, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    new_tokens = generated_ids[:, prompt_length:]
    decoded = tokenizer.batch_decode(new_tokens)
    #print("FULL ANSWER:", tokenizer.batch_decode(generated_ids)[0])
    #print("DECODED:", decoded[0])
    #print("RESULT:", decoded[0].strip().lower().startswith("yes"))
    #exit()
    return decoded[0].strip().lower().startswith("yes"), decoded[0]


def score_model_answers(model, tokenizer, index, query_prompt, answer_aliases, generated_answer, prompt_template, original_model="EleutherAI/gpt-j-6B"):
    print("row:", index, flush=True)
    original_tokenizer = AutoTokenizer.from_pretrained(original_model, clean_up_tokenization_spaces=True)
    original_tokenizer.pad_token = original_tokenizer.eos_token
    generated_tokens = original_tokenizer.encode(generated_answer, return_tensors="pt")[0]
    llm_judgment_result = {}
    llm_judgment = {}
    for i in range(8, generated_tokens.shape[0] + 1, 8):
        answer = original_tokenizer.decode(generated_tokens[:i])
        result = make_model_judgment(model, tokenizer, query_prompt, answer_aliases, answer, prompt_template)
        llm_judgment_result[str(i)] = result[0]
        llm_judgment[str(i)] = result[1]
    return llm_judgment_result, llm_judgment


def get_model_judgments(get_llm_judgments, model, result_file, prompt_template):
    if not get_llm_judgments:
        return
    model_name = model
    #model_name = "Qwen/Qwen2.5-32B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    tokenizer.pad_token = tokenizer.eos_token

    #print(tokenizer.chat_template)
    #exit()
    if model_name == "Qwen/Qwen2.5-32B-Instruct":
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2", token=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", token=True)
    df = load_data()
    df["llm_judge"] = model_name
    df["llm_judgment"] = df.apply(lambda row: score_model_answers(model, tokenizer, row.name, row["query_prompt"], row["answer_aliases"], row["generated_answer"], prompt_template), axis=1)
    df[["llm_judgment_result", "llm_judgment"]] = pd.DataFrame(df["llm_judgment"].tolist(), index=df.index)
    df = df.drop(columns=["answer_aliases"])
    df.to_parquet(f"model_editing_results/generate_length/rating/{result_file}")


def plot_stack_charts(df):
    editors = ["in-context", "context-retriever", "memit"]
    datasets= ["zsre", "CounterFact", "MQuAKE", "RippleEdits"]
    data = {}
    for dataset, editor in product(datasets, editors):
        data[(dataset, editor)] = {}
        for i in range(8, 65, 8):
            data[(dataset, editor)][i] = {
                "tp": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
            }

    for _, row in df.iterrows():
        if row["editor"] == "no-edit":
            continue
        for length, system_verdict in row["llm_judgment_result"].items():
            correct_first_answer = row["Correct First Answer"]
            
            if system_verdict is True and correct_first_answer is True:
                data[(row["dataset"], row["editor"])][int(length)]["tp"] += 1
            elif system_verdict is True:
                data[(row["dataset"], row["editor"])][int(length)]["fp"] += 1
            elif system_verdict is False and correct_first_answer is False:
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

    df = pd.DataFrame(plot_data)

    df_table = df.copy()
    df_table["accuracy"] = (df_table["tp"] + df_table["tn"]) / (df_table["tp"] + df_table["tn"] + df_table["fp"] + df_table["fn"])
    df_table = df_table[df_table['generate_length'] == 24].drop(columns=["editor"]).groupby(["dataset"]).agg("mean").reset_index()
    print(df_table.to_string())
    exit()

    all_generate_lengths = list(range(8, 65, 8))
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
    fig.write_image("model_editing_results/generate_length/rating/stacked_bar_chart.png", width=900, height=550, engine="kaleido")
    print("wrote image to:", "model_editing_results/generate_length/rating/stacked_bar_chart.png")


def analyse_model_judgments(result_file):
    df = pd.read_parquet(f"model_editing_results/generate_length/rating/{result_file}")
    df = df.drop(columns=["Multiple Answers", "dimension", "example_id", "result-late_success", "query_result"])
    df = df[df["editor"] != "no-edit"]
    # df = df[df["Correct First Answer"] != df["llm_judgment"]]
    print(df.head())
    print(df.columns)
    plot_stack_charts(df)


