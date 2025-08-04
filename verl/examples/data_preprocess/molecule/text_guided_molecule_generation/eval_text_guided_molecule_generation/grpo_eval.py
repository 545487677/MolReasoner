from eval_metrics import evaluate_from_json
import json
import selfies as sf
import pandas as pd
import re 

def sf_encode(selfies):
    try:
        smiles = sf.decoder(selfies)
        return smiles
    except Exception:
        return None

def evaluate(json_path: str, output_txt_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f if line.strip()]

    data = pd.DataFrame(records)

    data['label_encoded'] = data['label'].map(sf_encode)

    none_count = data['label_encoded'].isnull().sum()
    none_examples = data[data['label_encoded'].isnull()].head(5)

    print(none_count, none_examples[['label', 'label_encoded']])
    metrics= evaluate_from_json(json_path=json_path)
    with open(output_txt_path, "w") as f:
        f.write("🔬 Evaluation Metrics\n")
        f.write("=====================\n")
        f.write(f"📘 BLEU Score (↑): {metrics['bleu']:.4f}\n")
        f.write(f"🎯 Exact InChI Match (↑): {metrics['exact_match']:.4f}\n")
        f.write(f"✏️ Avg Levenshtein  (↓): {metrics['levenshtein']:.4f}\n")
        f.write(f"🔗 RDK Similarity (↑): {metrics['rdk_similarity']:.4f}\n")
        f.write(f"🔗 MACCS Similarity (↑): {metrics['maccs_similarity']:.4f}\n")
        f.write(f"🔗 Morgan Similarity (↑): {metrics['morgan_similarity']:.4f}\n")
        f.write(f"🧩 Frag-Jaccard Overlap (↑): {metrics['frag_jaccard']:.4f}\n")
        f.write(f"🧩 Frag Recall (↑): {metrics['frag_recall']:.4f}\n")
        f.write(f"🧪 FG-Diff (Functional Group Similarity) (↑): {metrics['fg_diff']:.4f}\n")
        f.write(f"✅ Validity (↑): {metrics['validity']:.4f}\n")


if __name__ == "__main__":
    evaluate(
        json_path="/fs_mol/guojianz/projects/FunMG/paper_code/open_source_github/MolReasoner/verl/examples/data_preprocess/molecule/text_guided_molecule_generation/eval_text_guided_molecule_generation/saved_results/grpo_MolReasoner.json",
        output_txt_path="/fs_mol/guojianz/projects/FunMG/paper_code/open_source_github/MolReasoner/verl/examples/data_preprocess/molecule/text_guided_molecule_generation/eval_text_guided_molecule_generation/saved_results/grpo_MolReasoner_metrics.txt"
    )

# 📘 BLEU Score (↑): 0.7841
# 🎯 Exact InChI Match (↑): 0.0758
# ✏️ Avg Levenshtein  (↓): 26.9255
# 🔗 RDK Similarity (↑): 0.4373
# 🔗 MACCS Similarity (↑): 0.6759
# 🔗 Morgan Similarity (↑): 0.3627
# 🧩 Frag-Jaccard Overlap (↑): 0.5213
# 🧩 Frag Recall (↑): 0.6414
# 🧪 FG-Diff (Functional Group Similarity) (↑): 0.5390
# ✅ Validity (↑): 0.9679