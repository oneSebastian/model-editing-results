from pathlib import Path
import pandas as pd
import numpy as np
import random
from collections import defaultdict
import gspread
from gspread_formatting import set_frozen, cellFormat, set_text_format_runs, TextFormat, TextFormatRun, DataValidationRule, BooleanCondition
from gspread_formatting import DataValidationRule, BooleanCondition, set_data_validation_for_cell_range
from oauth2client.service_account import ServiceAccountCredentials
from gspread.utils import ValidationConditionType
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Set up the scope and credentials for Google Sheets and Google Drive APIs
scope = [
    "https://spreadsheets.google.com/feeds", 
    "https://www.googleapis.com/auth/spreadsheets", 
    "https://www.googleapis.com/auth/drive.file"  # This allows the creation of files
]
# TODO: insert your credentials here
creds = ServiceAccountCredentials.from_json_keyfile_name("path/toyour/credentials/json", scope)

# Authenticate and create a client
client = gspread.authorize(creds)

# Build the Sheets API service
service = build('sheets', 'v4', credentials=creds)

def load_data(results_path):
    # load and aggregate ke_results
    dfs_ke = []
    for path in Path(results_path).iterdir():
        if str(path).endswith("_generate_lengths"):
            dfs_ke.append(pd.read_parquet(path))
            print(f"Loaded data from path={path}")
    return pd.concat(dfs_ke, axis=0, ignore_index=True)


def create_google_spreadsheet():
    df = load_data("experiments/generate_length_gpt2")
    df = df.drop(["query_result"], axis=1)
    df["Multiple Answers"] = ""
    df["Correct First Answer"] = ""
    df["Match Answer"] = ""
    df = df[["Multiple Answers", "Correct First Answer", "Match Answer", 'query_prompt', 'correct_answers', 'generated_answer', 'model', 'editor', 'dataset', 'dimension', 'batch_id', 'example_id']]

    def extract_answer(answers):
        if isinstance(answers[0], str):
            return answers[0]
        else:
            return list(answers[0])[0]
    
    def get_highlight_span(row):
        model_answer = row["generated_answer"]
        correct_answers = row["correct_answers"]
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
    answer_span = df.apply(get_highlight_span, axis=1)
    df["correct_answers"] = df["correct_answers"].apply(extract_answer)

    # Create a new Google Sheet
    sheet = client.create('Generated Table')
    print(f"Spreadsheet URL: {sheet.url}")
    worksheet = sheet.get_worksheet(0)

    # Update the sheet with DataFrame data
    worksheet.update([df.columns.values.tolist()] + df.values.tolist())

    # Apply Yes/No dropdown validation to the first three columns
    for col in ["A", "B", "C"]:
        cell_range = f"{col}2:{col}{len(df) + 1}"
        rule = DataValidationRule(
            condition=BooleanCondition('BOOLEAN'),
            showCustomUi=True
        )
        set_data_validation_for_cell_range(worksheet, cell_range, rule)

    # Share the spreadsheet with your personal Google account email address
    personal_email = "your-account@e-amil.com"  # Replace with your actual email address
    sheet.share(personal_email, perm_type='user', role='writer')
    print(f"Spreadsheet shared with: {personal_email}")
    
    # Prepare the batch update request for bold formatting
    df["answer_span"] = answer_span
    requests = []
    for idx, row in df.iterrows():
        answer_span = row['answer_span']
        if answer_span:
            start, end = answer_span
            # print(f"start={start}, end={end}, answer_string_length={len(row["generated_answer"])}")
            requests.append({
                "updateCells": {
                    "range": {
                        "sheetId": worksheet.id,
                        "startRowIndex": idx + 1,
                        "endRowIndex": idx + 2,
                        "startColumnIndex": 5,
                        "endColumnIndex": 6
                    },
                    "rows": [{"values": [{"textFormatRuns":[
                        {"format": {"bold": True}, "startIndex": start},
                        {"format": {"bold": False}, "startIndex": end}
                    ]}]}],
                    "fields": "textFormatRuns.format.bold"
                }
            })

    # Send the batch update request
    if requests:
        sheet.batch_update({"requests": requests})
    

    # change width and hide last columns:
    requests = [
        {
            "updateDimensionProperties": {
                "range": {
                    "sheetId": worksheet.id,
                    "dimension": "COLUMNS",
                    "startIndex": 3,  # Column F (zero-based index)
                    "endIndex": 4     # One past the last column to resize
                },
                "properties": {"pixelSize": 350},  # Set column width in pixels
                "fields": "pixelSize"
            }
        },
        {
            "updateDimensionProperties": {
                "range": {
                    "sheetId": worksheet.id,
                    "dimension": "COLUMNS",
                    "startIndex": 4,  # Column F (zero-based index)
                    "endIndex": 5     # One past the last column to resize
                },
                "properties": {"pixelSize": 175},  # Set column width in pixels
                "fields": "pixelSize"
            }
        },
        {
            "updateDimensionProperties": {
                "range": {
                    "sheetId": worksheet.id,
                    "dimension": "COLUMNS",
                    "startIndex": 5,  # Column F (zero-based index)
                    "endIndex": 6     # One past the last column to resize
                },
                "properties": {"pixelSize": 700},  # Set column width in pixels
                "fields": "pixelSize"
            }
        },
    ]
    for i in range(6, 12):
        requests.append({
            "updateDimensionProperties": {
                "range": {
                    "sheetId": worksheet.id,
                    "dimension": "COLUMNS",
                    "startIndex": i,
                    "endIndex": i + 1
                },
                "properties": {"hiddenByUser": True},
                "fields": "hiddenByUser"
            }
            })
    sheet.batch_update({"requests": requests})
    exit()


def read_sample_base_data(path="experiments/generate_length_gptj", select_late_success=None):
    df = load_data(path)
    df = df.drop(["model", "query_result", "batch_id"], axis=1)

    def de_np(answers):
        if isinstance(answers[0], str):
            return tuple(answers)
        else:
            return tuple(tuple(answer) for answer in answers)

    df['correct_answers'] = df['correct_answers'].apply(de_np)
    print("Number of examples:", len(df))
    print(df.columns)

    df = df.pivot_table(
        index=["dataset", "dimension", "example_id", "query_prompt", "correct_answers"],
        columns="editor",
        values="generated_answer",
        aggfunc="first"
    ).reset_index()
    df.columns.name = None

    def contains_late_success(row):
        if isinstance(row["correct_answers"][0], str):
            answers = row["correct_answers"]
        else:
            answers = ()
            for answer in row["correct_answers"]:
                answers += answer
        for editor in ["context-retriever", "in-context", "memit", "no-edit"]:
            model_answer = row[editor]
            for answer in answers:
                start = model_answer.find(answer)
                if start > len(model_answer) // 2:
                    return True
        return False
    
    df["late-success"] = df.apply(contains_late_success, axis=1)

    if select_late_success is not None:
        df = df[df["late-success"] == select_late_success]
    return df


def draw_data_samples(n_samples=2, sample_size=50, select_late_success=True):
    df = read_sample_base_data(select_late_success=select_late_success)

    # exclude zsre_fact_queries, since they just double up in efficacy queries
    df = df[(df["dataset"] != "zsre") | (df["dimension"] != "fact_queries")]
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Build strata based on the columns to balance
    strata_cols = ['dataset']
    df['stratum'] = df[strata_cols].astype(str).agg('_'.join, axis=1)

    # Group by stratum
    strata_groups = df.groupby('stratum')

    # Collect available rows per stratum
    stratum_to_rows = defaultdict(list)
    for stratum, group in strata_groups:
        for idx in group.index:
            stratum_to_rows[stratum].append(idx)
    
    for stratum, ids in stratum_to_rows.items():
        print(f"{stratum} has {len(ids)} examples.")

    # Initialize sample list
    samples = []

    for _ in range(n_samples):
        sample_idxs = []

        # Get how many rows to draw per stratum
        n_strata = len(stratum_to_rows)
        n_per_stratum = sample_size // n_strata
        leftover = sample_size % n_strata  # Distribute remainder randomly later

        # Randomly pick rows from each stratum
        for i, (stratum, indices) in enumerate(stratum_to_rows.items()):
            n_pick = n_per_stratum + (1 if i < leftover else 0)  # Add one to some strata to fill up sample
            available = indices[:n_pick]

            if len(available) < n_pick:
                raise ValueError(f"Not enough rows in stratum '{stratum}' to draw {n_pick} samples.")

            sample_idxs.extend(available)
            # Remove used rows from stratum pool
            stratum_to_rows[stratum] = indices[n_pick:]

        # Add sampled DataFrame
        samples.append(df.loc[sample_idxs].copy())

    return samples


def create_sample_spreadsheets(sample_id, sample_data, prefix="", log_path="generate_length/rating/log.txt"):
    # separate sample data by editors
    variants = ["A", "B", "C", "D"]
    editors = ["context-retriever", "in-context", "memit", "no-edit"]
    indices = [0, 1, 2, 3]
    dfs = {}
    for variant in variants:
        dfs[variant] = pd.DataFrame(columns=["editor", "dataset", "dimension", "example_id", "query_prompt", "correct_answers", "model_answer"])
    
    for _, row in sample_data.iterrows():
        random.shuffle(indices)
        for i, editor in enumerate(editors):
            data = {
                "editor": editor,
                "dataset": row["dataset"],
                "dimension": row["dimension"],
                "example_id": row["example_id"],
                "query_prompt": row["query_prompt"],
                "correct_answers": row["correct_answers"],
                "model_answer": row[editor],
            }
            dfs[variants[indices[i]]].loc[len(dfs[variants[indices[i]]])] = data
    
    # actually create the spreadsheet
    for variant in variants:
        df = dfs[variant]
        df["Multiple Answers"] = ""
        df["Correct First Answer"] = ""
        df["Match Answer"] = ""
        df = df[["Multiple Answers", "Correct First Answer", "Match Answer", 'query_prompt', 'correct_answers', 'model_answer', 'editor', 'dataset', 'dimension', 'example_id']]

        def extract_answer(answers):
            if isinstance(answers[0], str):
                return answers[0]
            else:
                return answers[0][0]
        
        def get_highlight_span(row):
            model_answer = row["model_answer"]
            correct_answers = row["correct_answers"]
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
        answer_span = df.apply(get_highlight_span, axis=1)
        df["correct_answers"] = df["correct_answers"].apply(extract_answer)

        # Create a new Google Sheet
        table_name = f"{prefix}Table {variant}{sample_id}"
        sheet = client.create(table_name)
        with open(log_path, "a") as f:
            f.write(f"{table_name} at url={sheet.url}\n")

        worksheet = sheet.get_worksheet(0)

        # Update the sheet with DataFrame data
        worksheet.update([df.columns.values.tolist()] + df.values.tolist())

        for col in ["A", "B", "C"]:
            cell_range = f"{col}2:{col}{len(df) + 1}"
            rule = DataValidationRule(
                condition=BooleanCondition('BOOLEAN'),
                showCustomUi=True
            )
            set_data_validation_for_cell_range(worksheet, cell_range, rule)

        # Share the spreadsheet with your personal Google account email address
        personal_email = "sebpohl@gmx.net"  # Replace with your actual email address
        sheet.share(personal_email, perm_type='user', role='writer')
        
        # Prepare the batch update request for bold formatting
        df["answer_span"] = answer_span
        requests = []
        for idx, row in df.iterrows():
            answer_span = row['answer_span']
            if answer_span:
                start, end = answer_span
                # print(f"start={start}, end={end}, answer_string_length={len(row["generated_answer"])}")
                requests.append({
                    "updateCells": {
                        "range": {
                            "sheetId": worksheet.id,
                            "startRowIndex": idx + 1,
                            "endRowIndex": idx + 2,
                            "startColumnIndex": 5,
                            "endColumnIndex": 6
                        },
                        "rows": [{"values": [{"textFormatRuns":[
                            {"format": {"bold": True}, "startIndex": start},
                            {"format": {"bold": False}, "startIndex": end}
                        ]}]}],
                        "fields": "textFormatRuns.format.bold"
                    }
                })

        # Send the batch update request
        if requests:
            sheet.batch_update({"requests": requests})
        

        # change width and hide last columns:
        requests = [
            {
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": worksheet.id,
                        "dimension": "COLUMNS",
                        "startIndex": 3,  # Column F (zero-based index)
                        "endIndex": 4     # One past the last column to resize
                    },
                    "properties": {"pixelSize": 350},  # Set column width in pixels
                    "fields": "pixelSize"
                }
            },
            {
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": worksheet.id,
                        "dimension": "COLUMNS",
                        "startIndex": 4,  # Column F (zero-based index)
                        "endIndex": 5     # One past the last column to resize
                    },
                    "properties": {"pixelSize": 175},  # Set column width in pixels
                    "fields": "pixelSize"
                }
            },
            {
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": worksheet.id,
                        "dimension": "COLUMNS",
                        "startIndex": 5,  # Column F (zero-based index)
                        "endIndex": 6     # One past the last column to resize
                    },
                    "properties": {"pixelSize": 700},  # Set column width in pixels
                    "fields": "pixelSize"
                }
            },
        ]
        for i in range(6, 12):
            requests.append({
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": worksheet.id,
                        "dimension": "COLUMNS",
                        "startIndex": i,
                        "endIndex": i + 1
                    },
                    "properties": {"hiddenByUser": True},
                    "fields": "hiddenByUser"
                }
                })
        sheet.batch_update({"requests": requests})
    
    #return data do print query success distribution
    return dfs


# Call the function to create and share the Google Sheet
# create_google_spreadsheet()
# create_instructions_spreadsheet()

df1 = read_sample_base_data(select_late_success=None)
df1.groupby(['dataset', 'dimension', 'late-success']).size().to_csv("generate_length/rating/success_overview.csv")
print(df1.groupby(['dataset', 'dimension', 'late-success']).size())

exit()

samples = draw_data_samples(select_late_success=False)
df = pd.concat(samples, ignore_index=True)
print(df.columns)
print(df.groupby(['dataset', 'dimension', 'late-success']).size())


def contains_late_success(row):
            if isinstance(row["correct_answers"][0], str):
                answers = row["correct_answers"]
            else:
                answers = ()
                for answer in row["correct_answers"]:
                    answers += answer

            for answer in answers:
                start = row["model_answer"].find(answer)
                if start > len(row["model_answer"]) // 2:
                    return "late"
                if start >= 0:
                    return "early"
            return "false"

counts = defaultdict(int)
for i, sample in enumerate(samples):
    dfs = create_sample_spreadsheets(i, sample, prefix="No-Late ")
    for df in dfs.values():
        for _, row in df.iterrows():
            late_success = contains_late_success(row)
            key = (row["editor"], row["dataset"], late_success)
            counts[key] += 1

result = [(k, v) for k, v in counts.items()]
result.sort()
for _ in result:
    print(_)

        

