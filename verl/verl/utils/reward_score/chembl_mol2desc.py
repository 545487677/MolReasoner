import os
import re
import numpy as np
from collections import Counter

try:
    import rouge_score
except ImportError:
    os.system('pip install rouge-score')

from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import AllChem, Descriptors, MACCSkeys

rdBase.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.*')


from transformers import BertTokenizerFast

import nltk
# nltk_packages = ['wordnet', 'punkt', 'omw-1.4']
# for pkg in nltk_packages:
#     try:
#         nltk.data.find(f'~/nltk_data/tokenizers/{pkg}' if "punkt" in pkg else f'corpora/{pkg}')
#     except LookupError:
#         nltk.download(pkg)

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer



def extract_solution(solution_str: str) -> str:
    """
    Extract the last <answer>...</answer> block from the solution string.
    """
    matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    return matches[-1].group(1).strip() if matches else None

def compute_score_bleu_meteor(solution_str: str, ground_truth: str) -> float:
    try:
        pred_text = extract_solution(solution_str)
        if pred_text is None or not ground_truth:
            return 0.0
        if pred_text.strip().lower() == ground_truth.strip().lower():
            return 2.0

        primary_path = 'xxxx/scibert_scivocab_uncased'
        backup_path = 'xxxxx/scibert_scivocab_uncased'

        load_path = primary_path if os.path.exists(primary_path) else backup_path

        text_tokenizer = BertTokenizerFast.from_pretrained(load_path)
        text_trunc_length = 6144

        gt_tokens = text_tokenizer.tokenize(ground_truth, truncation=True, max_length=text_trunc_length, padding='max_length')
        gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

        out_tokens = text_tokenizer.tokenize(pred_text, truncation=True, max_length=text_trunc_length, padding='max_length')
        out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
        out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
        out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

        if len(out_tokens) == 0 or len(gt_tokens) == 0:
            return 0.0

        references = [gt_tokens]
        hypotheses = out_tokens

        bleu2 = corpus_bleu([references], [hypotheses], weights=(0.5, 0.5))
        bleu4 = corpus_bleu([references], [hypotheses], weights=(0.25, 0.25, 0.25, 0.25))
        meteor = meteor_score([gt_tokens], out_tokens)

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge = scorer.score(pred_text, ground_truth)

        rouge1 = rouge['rouge1'].fmeasure
        rouge2 = rouge['rouge2'].fmeasure
        rougeL = rouge['rougeL'].fmeasure

        score = 0.5 + 1.5 * ((bleu2 + bleu4 + meteor + rouge1 + rouge2 + rougeL) / 6.0)
        return score

    except Exception as e:
        print(f"[Warning] compute_score_bleu_meteor error: {e}")
        return 0.0


def compute_score_format_acc(solution_str: str, ground_truth: str) -> float:
    pred_text = extract_solution(solution_str)
    if pred_text is None or not ground_truth:
        return 0.0
    if pred_text.strip().lower() == ground_truth.strip().lower():
        return 2.0
    else:
        return 0.5




def compute_score(solution_str: str, ground_truth: str, exp_method:str) -> float:
    if exp_method == "mol2desc_default":
        return compute_score_bleu_meteor(solution_str, ground_truth)
    elif exp_method == "format_acc":
        return compute_score_format_acc(solution_str, ground_truth)
   
    

if __name__ == "__main__":
    solution_str = """
    Let's reason step by step.
    The correct result is clearly described below.
    <answer>The capital of France is B</answer>
    """
    ground_truth = "The capital of France is Paris"
    
    exp_method = "format_acc"
    score = compute_score(solution_str, ground_truth, exp_method)
    
    print(f"Computed {exp_method} score: {score:.4f}")
