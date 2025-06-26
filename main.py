import argparse
import yaml
import model_editing_results.paper_figures as paper_figures
from model_editing_results.generate_length.llm_judge import get_model_judgments, analyse_model_judgments


def int_or_none(arg):
    if arg.lower() == "none":
        return None
    return int(arg)


def llm_judge(args):
    get_model_judgments(
        get_llm_judgments = args.get_llm_judgments,
        model = args.model,
        result_file = args.result_file,
        prompt_template = args.prompt_template,
    )
    analyse_model_judgments(result_file=args.result_file)

def create_paper_figures(args):
    if args.editing_table:
        paper_figures.create_overall_table()
    if args.control_table:
        paper_figures.create_control_tables()
    if args.control_individual:
        paper_figures.plot_control_individually()
    if args.generate_length:
        paper_figures.plot_generate_length()


def main():
    parser = argparse.ArgumentParser(description="Analyse model editing results.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    llm_judge_parser = subparsers.add_parser("llm_judge", help="Use LLM as judge for model responses and save LLM judge results")
    llm_judge_parser.set_defaults(func=llm_judge)
    llm_judge_parser.add_argument("--get_llm_judgments", type=bool, default=False, help="Let an LLM jusdge the model responses")
    llm_judge_parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Huggingface model name")
    llm_judge_parser.add_argument("--result_file", type=str, default="llm_judged.parquet", help="File to write results to and load from")
    llm_judge_parser.add_argument("--prompt_template", type=int, default=0, help="Template for the instruction prompt")

    paper_figures_parser = subparsers.add_parser("paper_figures", help="Recreate figures and tables from paper")
    paper_figures_parser.set_defaults(func=create_paper_figures)
    paper_figures_parser.add_argument("--editing_table", action="store_true", help="Create overall table for editing results")
    paper_figures_parser.add_argument("--control_table", action="store_true", help="Create overall table for control results")
    paper_figures_parser.add_argument("--control_individual", action="store_true", help="Create individual plots for control tasks")
    paper_figures_parser.add_argument("--generate_length", action="store_true", help="Plots generate length results")

    args = parser.parse_args()
    args.func(args)
    

if __name__ == "__main__":
    main()