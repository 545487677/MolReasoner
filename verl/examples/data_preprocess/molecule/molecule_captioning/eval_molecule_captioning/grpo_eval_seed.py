from eval_metrics import evaluate_from_json
import json
import pandas as pd

def evaluate(json_path: str, output_txt_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f if line.strip()]

    data = pd.DataFrame(records)
    count_prompt_without_assistant = sum(
        1 for item in records if "assistant" not in item["prompt"]
    )
    print(f"Number of prompts without 'assistant': {count_prompt_without_assistant}")
    metrics = evaluate_from_json(json_path=json_path)
    with open(output_txt_path, "w") as f:
        f.write("🔬 Evaluation Metrics (Text-Based)\n")
        f.write("=====================\n")
        f.write(f"📘 BLEU-2 (↑): {metrics['bleu2']:.4f}\n")
        f.write(f"📘 BLEU-4 (↑): {metrics['bleu4']:.4f}\n")
        f.write(f"💡 METEOR (↑): {metrics['meteor']:.4f}\n")
        f.write(f"📕 ROUGE-1 (↑): {metrics['rouge1']:.4f}\n")
        f.write(f"📕 ROUGE-2 (↑): {metrics['rouge2']:.4f}\n")
        f.write(f"📕 ROUGE-L (↑): {metrics['rougeL']:.4f}\n")



def evaluate_results(epochs=None):
        json_path = f"xxxxx/MolReasoner/verl/examples/data_preprocess/molecule/molecule_captioning/eval_molecule_captioning/saved_results/grpo_MolReasoner.json"
        output_txt_path = f"xxxxx/MolReasoner/verl/examples/data_preprocess/molecule/molecule_captioning/eval_molecule_captioning/saved_results/grpo_MolReasoner_metric.txt"

        evaluate(json_path=json_path, output_txt_path=output_txt_path)

if __name__ == "__main__":
    evaluate_results()


# BLEU-2 score: 0.4382742955139697
# BLEU-4 score: 0.32203432589674036
# Average Meteor score: 0.4754395684581413
# ROUGE score:
# rouge1: 0.553020741756104
# rouge2: 0.3661515826649664
# rougeL: 0.4820858080733351