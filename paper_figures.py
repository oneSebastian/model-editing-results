import pandas as pd
from model_editing.analysis import EvalResult
# from editing_benchmark.analysis import EvalResult
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from math import sqrt


def rename_editor(ed):
            if ed == "memit":
                return "MEMIT"
            elif ed == "context-retriever":
                return "cont-retr"
            elif ed == "in-context":
                return "in-cont"
            else:
                return ed
            

def create_overall_table():
    paths = [
        (1, "experiments/gpt_j/batch_size_1/"),
        (16, "experiments/gpt_j/batch_size_16/"),
        (64, "experiments/gpt_j/batch_size_64/"),
        (512, "experiments/gpt_j/batch_size_512/"),
        (2048, "experiments/gpt_j/batch_size_2048/"),
        (1, "experiments/gpt2_xl/batch_size_1/"),
        (16, "experiments/gpt2_xl/batch_size_16/"),
        (64, "experiments/gpt2_xl/batch_size_64/"),
        (512, "experiments/gpt2_xl/batch_size_512/"),
        (2048, "experiments/gpt2_xl/batch_size_2048/"),
    ]

    dfs = []
    for batch_size, path in paths:
        result = EvalResult()
        result.load_editing_data(path)
        result.aggregate_editing_data(groupby_dimensions=True, groupby_dataset_splits=False, exclude_fact_queries=True, evaluate_generate_lengths=False)
        df = result.aggregated_editing_data.reset_index()
        df = df.drop(columns=["test_cases", "valid_test_cases", "verify_test_case_time", "edit_time", "eval_time", "valid_test_case_ratio", "experiment_count"])
        df["batch_size"] = batch_size
        dfs.append(df)
    
    df = pd.concat(dfs, axis=0, ignore_index=True)
    no_edit_16 = (
        df[(df["editor"].str.contains("no-edit", na=False)) & (df["batch_size"] == 16)]
        .groupby(["model", "dataset"]).mean(numeric_only=True)
    )
    df = df[df["editor"] != "no-edit"]

    # Pivot the dataframe to get a multi-index table: rows=(model, dataset), columns=(batch_size, editor)
    pivot = df.pivot_table(
        index=["model", "dataset", "dimension"],
        columns=["batch_size", "editor"],
        values="accuracy"
    )
    pivot = pivot.groupby(["model", "dataset"]).mean(numeric_only=True)

    # Flatten the multi-level columns
    pivot.columns = [f"{bs}-{ed}" for bs, ed in pivot.columns]
    pivot["no-edit"] = no_edit_16["accuracy"]

    # Optionally, sort columns to group by batch size, and then move the no-edit column to the end
    cols = [col for col in pivot.columns if col != "no-edit"]
    cols.sort(key=lambda col: int(col[:col.find("-")]))
    pivot = pivot[cols + ["no-edit"]]

    def bold_max_in_group(row):
        new_row = row.copy()
        for bs in ["1", "16", "64", "512", "2048"]:
            group_cols = [col for col in row.index if col.startswith(f"{bs}-")]
            if group_cols:
                group_values = row[group_cols]
                max_val = group_values.max()
                for col in group_cols:
                    if pd.isna(row[col]):
                        continue
                    val = f"{row[col]:.3f}"
                    if row[col] == max_val:
                        val = f"\\textbf{{{val}}}"
                    new_row[col] = val
        if pd.notna(row["no-edit"]):
            new_row["no-edit"] = f"{row['no-edit']:.3f}"
        return new_row

    df_print = pivot.copy().apply(bold_max_in_group, axis=1)
    print(df_print.to_string())

    # Convert to LaTeX
    latex_table = df_print.reset_index().to_latex(index=False, float_format="%.3f", column_format="ll" + "c" * (len(df_print.columns)))
    print(latex_table)

    df = pivot.reset_index().melt(id_vars=["model", "dataset", "no-edit"], var_name="batch_size-editor", value_name="accuracy")
    df[["batch_size", "editor"]] = df["batch_size-editor"].str.extract(r"(\d+)-(.+)")
    df['batch_size'] = df['batch_size'].astype(int)
    df = df.drop(columns=["batch_size-editor"])

    
    

    group_order = ["MQuAKE", "RippleEdits", "zsre", "CounterFact"]
    fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.1, vertical_spacing=0.1, subplot_titles=group_order)
    grouped = df.groupby("dataset")
    sorted_group_names = sorted(grouped.groups.keys(), key=lambda x: group_order.index(x))
    idx = -1
    for group_name in sorted_group_names:
        group_df = grouped.get_group(group_name)
    #for group_name, group_df in grouped:
        idx += 1
        # print(group_df.to_string())
        df = group_df
        no_edit_value = {}
        for model, df_model in df.groupby("model"):
            no_edit_value[model] = df_model["no-edit"].mean()

        editors = ["context-retriever", "in-context", "memit"]
        models = df["model"].unique()

        # Define line styles for each model
        line_styles = {
            "gpt-j": "solid",
            "gpt2-xl": "dot"
        }

        # Define color map for editors
        color_map = {
            "context-retriever": "blue",
            "in-context": "green",
            "memit": "red"
        }

        show_legend = (idx == 0)
        row, col = divmod(idx, 2)

        # Create a line for each (model, editor) pair
        add_no_edit = "gpt-j"
        for editor in editors:
            df_editor = df[df["editor"] == editor]
            for model in models:
                df_model = df_editor[df_editor["model"] == model]
                fig.add_trace(
                    go.Scatter(
                        x=df_model["batch_size"],
                        y=df_model["accuracy"],
                        mode="lines+markers",
                        name=f"{model} {rename_editor(editor)}",
                        line=dict(
                            color=color_map[editor],
                            dash=line_styles[model]
                        ),
                        showlegend=show_legend
                    ),
                    row=row + 1, col=col + 1
                )
            if add_no_edit != "skip":
                fig.add_trace(
                    go.Scatter(
                        x=df_model["batch_size"],  # Use the batch sizes as the x-axis
                        y=[no_edit_value[add_no_edit]] * len(df_model["batch_size"]),  # The no-edit value is the same for all batch sizes
                        mode="lines",
                        name=f"{add_no_edit} no-edit",
                        line=dict(
                            color="grey",
                            dash=line_styles[add_no_edit]
                        ),
                        showlegend=show_legend
                    ),
                    row=row + 1, col=col + 1
                )
                if add_no_edit == "gpt-j":
                    add_no_edit = "gpt2-xl"
                elif add_no_edit == "gpt2-xl":
                    add_no_edit = "skip"
        

        # Update layout
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=2, col=1)
        fig.update_xaxes(title_text="Edit Batch Size", row=2, col=1)
        fig.update_xaxes(title_text="Edit Batch Size", row=2, col=2)
        for axis_name in fig.layout:
            if axis_name.startswith("xaxis"):
                fig.layout[axis_name].type = "log"
        fig.update_layout(
            #title=f"Accuracy by Model and Editor",
            #xaxis_title="Batch Size",
            #yaxis_title="Accuracy",
            #legend_title="Model - Editor",
            template="plotly_white",
            #xaxis_type="log",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.35,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=15, r=15, t=60, b=70),
        )

        # Save the figure as an image
        fig.write_image(f"visualisations/overview.svg", width=550, height=550, engine="kaleido")
        fig.write_image(f"visualisations/overview.png", width=550, height=550, engine="kaleido")


def create_control_tables():
    model_paths = {
        "GPT-J": [
            (1, "experiments/gpt_j/batch_size_1/"),
            (16, "experiments/gpt_j/batch_size_16/"),
            (64, "experiments/gpt_j/batch_size_64/"),
            (512, "experiments/gpt_j/batch_size_512/"),
            (2048, "experiments/gpt_j/batch_size_2048/"),
        ],
        "GPT-2-XL": [
            (1, "experiments/gpt2_xl/batch_size_1/"),
        (16, "experiments/gpt2_xl/batch_size_16/"),
        (64, "experiments/gpt2_xl/batch_size_64/"),
        (512, "experiments/gpt2_xl/batch_size_512/"),
        (2048, "experiments/gpt2_xl/batch_size_2048/"),
        ],
    }
    for model, paths in model_paths.items():
        dfs = []
        for batch_size, path in paths:
            result = EvalResult()
            result.load_aggregated_control_data(path)
            df = result.aggregated_control_data.reset_index()
            df = df.drop(columns=["index", "model", "n-samples", "weighted_score", "batch_count", "eval_time", "higher_is_better"])
            df["batch_size"] = batch_size
            dfs.append(df)
        
        df = pd.concat(dfs, axis=0, ignore_index=True)
        print(df.to_string())
        df_filtered = df[df["editor"] != "no-edit"]

        # Pivot the dataframe to get a multi-index table: rows=(model, dataset), columns=(batch_size, editor)
        pivot = df_filtered.pivot_table(
            index=["task", "metric"],
            columns=["batch_size", "editor"],
            values="score"
        )

        # Flatten the multi-level columns
        pivot.columns = [f"{bs}-{ed}" for bs, ed in pivot.columns]

        # Get the `no-edit` scores only for batch_size == 64
        no_edit_64 = df[(df["editor"] == "no-edit") & (df["batch_size"] == 64)].copy()
        no_edit_64.set_index(["task", "metric"], inplace=True)
        pivot["no-edit"] = no_edit_64["score"]

        # Optionally, sort columns to group by batch size, and then move the no-edit column to the end
        cols = [col for col in pivot.columns if col != "no-edit"]
        cols.sort(key=lambda col: int(col[:col.find("-")]))
        pivot = pivot[cols + ["no-edit"]]

        def bold_max_in_group(row):
            new_row = row.copy()
            for bs in ["1", "16", "64", "512", "2048"]:
                group_cols = [col for col in row.index if col.startswith(f"{bs}-")]
                if group_cols:
                    group_values = row[group_cols]
                    metric = row.name[1]
                    if metric in ["perplexity", "bits_per_byte", "byte_perplexity", "word_perplexity"]:
                        best_val = group_values.min()
                    else:
                        best_val = group_values.max()
                    for col in group_cols:
                        if pd.isna(row[col]):
                            continue
                        val = f"{row[col]:.3f}"
                        if row[col] == best_val:
                            val = f"\\textbf{{{val}}}"
                        new_row[col] = val
            if pd.notna(row["no-edit"]):
                new_row["no-edit"] = f"{row['no-edit']:.3f}"
            return new_row

        pivot = pivot.apply(bold_max_in_group, axis=1)
        print(pivot.to_string())

        # Convert to LaTeX
        latex_table = pivot.reset_index().to_latex(index=False, float_format="%.3f", column_format="ll" + "c" * (len(pivot.columns)))
        print(latex_table)



def inspect():
    result = EvalResult()
    base_path = "experiments/batch_size_2048/"
    result.load_editing_data(base_path)
    result.aggregate_editing_data(groupby_dimensions=False, groupby_dataset_splits=False, exclude_fact_queries=True, evaluate_generate_lengths=False)
    print(result.aggregated_editing_data.to_string())
    result.load_aggregated_control_data(base_path)
    print(result.aggregated_control_data.to_string())


def plot_control_individually():
    model_paths = {
        "GPT-J": [
            (1, "experiments/gpt_j/batch_size_1/"),
            (16, "experiments/gpt_j/batch_size_16/"),
            (64, "experiments/gpt_j/batch_size_64/"),
            (512, "experiments/gpt_j/batch_size_512/"),
            (2048, "experiments/gpt_j/batch_size_2048/"),
        ],
        "GPT-2-XL": [
            (1, "experiments/gpt2_xl/batch_size_1/"),
        (16, "experiments/gpt2_xl/batch_size_16/"),
        (64, "experiments/gpt2_xl/batch_size_64/"),
        (512, "experiments/gpt2_xl/batch_size_512/"),
        (2048, "experiments/gpt2_xl/batch_size_2048/"),
        ],
    }
    dfs = []
    for model, paths in model_paths.items():
        for batch_size, path in paths:
            result = EvalResult()
            result.load_aggregated_control_data(path)
            df = result.aggregated_control_data.reset_index()
            # df = df.drop(columns=["index", "model", "n-samples", "weighted_score", "batch_count", "eval_time", "higher_is_better"])
            df["batch_size"] = batch_size
            dfs.append(df)
        
    df = pd.concat(dfs, axis=0, ignore_index=True)
    dfs_average = []
    dfs = []
    grouped = df.groupby(["model", "task", "metric"])
    for group_name, group_df in grouped:
        higher_is_better = group_df["higher_is_better"].apply(tuple).unique()
        assert len(higher_is_better) == 1
        higher_is_better = higher_is_better[0][0]
        assert isinstance(higher_is_better, bool)
        pivot = group_df.pivot_table(
            index=["model", "task", "metric", "batch_size", "n-samples"],
            columns=["editor"],
            values="score"
        ).reset_index()
        no_edit = pivot[pivot["batch_size"] == 64].squeeze()["no-edit"]
        pivot = pivot.drop(columns=["no-edit"])
        cols = ["context-retriever", "in-context", "memit"]
        pivot[cols] = pivot[cols] - no_edit
        dfs.append(pivot)
        avg = pivot.copy()
        dfs_average.append(avg)

    
    df = pd.concat(dfs, axis=0, ignore_index=True)
    grouped = df.groupby(["task", "metric"])
    fig_combined = make_subplots(rows=1, cols=3, horizontal_spacing=0.12, vertical_spacing=0.1, subplot_titles=["hellaswag", "lambada", "lambada"])
    target_groups = [
        ("hellaswag", "acc_norm"),
        ("lambada_standard", "acc"),
        ("lambada_standard", "perplexity"),
    ]
    for group_name, group_df in grouped:
        # pivot.columns = [f"{bs}-{ed}" for bs, ed in pivot.columns]
        df = group_df
        print(group_df.to_string())
        editors = ["context-retriever", "in-context", "memit"]
        models = group_df["model"].unique()

        # Define line styles for each model
        line_styles = {
            "gpt-j": "solid",
            "gpt2-xl": "dot"
        }

        # Define color map for editors
        color_map = {
            "context-retriever": "blue",
            "in-context": "green",
            "memit": "red"
        }

        fig = go.Figure()

        # Create a line for each (model, editor) pair
        for editor in editors:
            for model in models:
        #for model in models:
                df_model = df[df["model"] == model]
        #    for editor in editors:
                fig.add_trace(
                    go.Scatter(
                        x=df_model["batch_size"],
                        y=df_model[editor],
                        mode="lines+markers",
                        name=f"{model} {rename_editor(editor)}",
                        line=dict(
                            color=color_map[editor],
                            dash=line_styles[model]
                        )
                    )
                )
                if group_name in target_groups:
                    col = target_groups.index(group_name) + 1
                    fig_combined.add_trace(
                        go.Scatter(
                            x=df_model["batch_size"],
                            y=df_model[editor],
                            mode="lines+markers",
                            name=f"{model} {rename_editor(editor)}",
                            line=dict(
                                color=color_map[editor],
                                dash=line_styles[model]
                            ),
                            showlegend=group_name == ("hellaswag", "acc_norm")
                        ),
                        row=1, col=col,
                    )

        # Update layout
        fig.update_layout(
            #title=f"{group_name}",
            xaxis_title="Batch Size",
            yaxis_title=f"Metric: {(group_name[1]).replace('_', ' ')}",
            #legend_title="Model - Editor",
            template="plotly_white",
            xaxis_type="log",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=+0.2,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=10, r=10, t=60, b=50),
        )

        #fig.write_image(f"visualisations/lm_eval/{group_name}.svg", width=500, height=500, engine="kaleido")
        fig.write_image(f"visualisations/lm_eval/{group_name}.png", width=500, height=500, engine="kaleido")
    
    # Update layout
    fig_combined.update_yaxes(title_text="Normalized Accuracy", row=1, col=1)
    fig_combined.update_yaxes(title_text="Accuracy", row=1, col=2)
    fig_combined.update_yaxes(title_text="Perplexity", row=1, col=3)
    fig_combined.update_xaxes(title_text="Edit Batch Size", row=1, col=1)
    fig_combined.update_xaxes(title_text="Edit Batch Size", row=1, col=2)
    fig_combined.update_xaxes(title_text="Edit Batch Size", row=1, col=3)
    for axis_name in fig_combined.layout:
        if axis_name.startswith("xaxis"):
            fig_combined.layout[axis_name].type = "log"
    fig_combined.update_layout(
        #title=f"Accuracy by Model and Editor",
        #xaxis_title="Batch Size",
        #yaxis_title="Accuracy",
        #legend_title="Model - Editor",
        template="plotly_white",
        #xaxis_type="log",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.83,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=15, r=15, t=60, b=40),
    )

    # Save the figure as an image
    #fig_combined.write_image(f"visualisations/lm_eval/combined.svg", width=630, height=280, engine="kaleido")
    fig_combined.write_image(f"visualisations/lm_eval/combined.png", width=630, height=280, engine="kaleido")
    
    # get table of average change
    df = pd.concat(dfs_average, axis=0, ignore_index=True)
    df = df.groupby(["model", "batch_size"]).mean(numeric_only=True).reset_index()
    print(df.to_string())
    editors = ["context-retriever", "in-context", "memit"]
    models = df["model"].unique()

    # Define line styles for each model
    line_styles = {
        "gpt-j": "solid",
        "gpt2-xl": "dot"
    }

    # Define color map for editors
    color_map = {
        "context-retriever": "blue",
        "in-context": "green",
        "memit": "red"
    }

    fig = go.Figure()

    # Create a line for each (model, editor) pair
    for model in models:
        df_model = df[df["model"] == model]
        for editor in editors:
            fig.add_trace(
                go.Scatter(
                    x=df_model["batch_size"],
                    y=df_model[editor],
                    mode="lines+markers",
                    name=f"{model} - {editor}",
                    line=dict(
                        color=color_map[editor],
                        dash=line_styles[model]
                    )
                )
            )

    # Update layout
    fig.update_layout(
        title=f"Average Delta",
        xaxis_title="Batch Size",
        yaxis_title=f"Normalised Delta",
        legend_title="Model - Editor",
        template="plotly_white",
        xaxis_type="log",
    )

    #fig.write_image(f"visualisations/lm_eval/average.svg", width=450, height=450, engine="kaleido")
    fig.write_image(f"visualisations/lm_eval/average.png", width=450, height=450, engine="kaleido")


def plot_generate_length():
    result = EvalResult()
    result.load_editing_data("experiments/generate_length_gptj")
    result.aggregate_editing_data(
        groupby_dimensions=False,
        groupby_dataset_splits=False,
        exclude_fact_queries=True,
        evaluate_generate_lengths=True,
    )
    df1 = result.aggregated_editing_data
    result = EvalResult()
    result.load_editing_data("experiments/generate_length_gpt2")
    result.aggregate_editing_data(
        groupby_dimensions=False,
        groupby_dataset_splits=False,
        exclude_fact_queries=True,
        evaluate_generate_lengths=True,
    )
    df2 = result.aggregated_editing_data
    df = pd.concat((df1, df2), axis=0, ignore_index=False)

    df_plot = pd.DataFrame(columns=["model", "editor", "dataset", "x", "y"])
    for index, row in df.iterrows():
        for k, v in row["accuracy"].items():
            data = {
                "model": f"{index[0]}",
                "editor": f"{index[1]}",
                "dataset": f"{index[2]}",
                "x": int(k),
                "y": v,
            }
            if data["editor"] != "no-edit":
                df_plot.loc[len(df_plot)] = data
    df_plot = df_plot.sort_values(by="x")

    dimensions = ["MQuAKE", "RippleEdits", "zsre", "CounterFact"]

    color_map = {
        "context-retriever": "blue",
        "in-context": "green",
        "memit": "red",
        "no-edit": "grey"
    }

    line_styles = {
        "gpt-j": "solid",
        "gpt2-xl": "dot"
    }

    fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.1, vertical_spacing=0.1, subplot_titles=dimensions)
    for idx, dimension in enumerate(dimensions):
        filtered_df = df_plot[df_plot["dataset"] == dimension]

        for model in filtered_df["model"].unique():
            for editor in filtered_df["editor"].unique():
                show_legend = (idx == 0)
                group_df = filtered_df[(filtered_df["model"] == model) & (filtered_df["editor"] == editor)]
                row, col = divmod(idx, 2)  # Get row and column index
                fig.add_trace(
                    go.Scatter(
                        x=group_df["x"],
                        y=group_df["y"],
                        mode="lines",
                        name=f"{model} {rename_editor(editor)}",
                        line=dict(
                            color=color_map[editor],
                            dash=line_styles[model]
                        ),
                        showlegend=show_legend
                    ),
                    row=row + 1, col=col + 1
                )

                #fig.add_trace(
                #    go.Scatter(x=group_df["x"], y=group_df["y"], mode="lines", name=editor, line=dict(color=editor_colors[editor]), showlegend=show_legend),
                #    row=row + 1, col=col + 1
                #)

    fig.update_xaxes(title_text="Generate Length", row=2, col=1)
    fig.update_xaxes(title_text="Generate Length", row=2, col=2)
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    #fig.update_yaxes(showticklabels=False, row=1, col=2)
    #fig.update_yaxes(showticklabels=False, row=2, col=2)

    fig.update_layout(
        #title="Accuracy per Group",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=10, r=10, t=60, b=50),
        meta=dict(mathjax=False),
    )
    #fig.write_image("visualisations/generate_length/subplots.svg", width=450, height=500, engine="kaleido")
    fig.write_image("visualisations/generate_length/subplots.png", width=450, height=500, engine="kaleido")


def gpt2_scoring_methods():
    paths = [
        "experiments/gpt2_xl/force_argmax",
        "experiments/gpt2_xl/force_generate",
        "experiments/gpt_j/argmax_forced",
        "experiments/gpt_j/generate_forced",
        "experiments/gpt_j/options_forced",
    ]
    for path in paths:
        result = EvalResult()
        result.load_editing_data("experiments/gpt2_xl/force_argmax")
        result.aggregate_editing_data(
            groupby_dimensions=False,
            groupby_dataset_splits=False,
            exclude_fact_queries=True,
            evaluate_generate_lengths=False,
        )
        print(f"Results from path={path}")
        print(result.aggregated_editing_data.drop(columns=["valid_test_cases", "verify_test_case_time", "edit_time", "eval_time", "valid_test_case_ratio", "experiment_count"]).to_string()) # "test_cases"


create_overall_table()
create_control_tables()
plot_control_individually()
plot_generate_length()
gpt2_scoring_methods()
