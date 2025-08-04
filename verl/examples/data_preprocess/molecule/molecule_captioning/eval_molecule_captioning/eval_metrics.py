import pandas as pd
import json
import re
import numpy as np
np.random.seed(42)
from tqdm import tqdm
import argparse
import csv
from transformers import BertTokenizerFast
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

def extract_solution(solution_str: str) -> str:
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return None

def evaluate_text_metrics(data, text_model='/fs_mol/guojianz/projects/FunMG/LLM/model_weight/models--Sihangli--3D-MoLM/scibert_scivocab_uncased', text_trunc_length=2048):
    outputs = []
    for i, row in tqdm(data.iterrows(), total=len(data)):
        outputs.append(( row['ground truth'], row['output']))
    text_tokenizer = BertTokenizerFast.from_pretrained(text_model)
    bleu_scores = []
    meteor_scores = []

    references = []
    hypotheses = []
    for i, (gt, out) in enumerate(outputs):

        if i % 100 == 0: print(i, 'processed.')


        gt_tokens = text_tokenizer.tokenize(gt, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

        out_tokens = text_tokenizer.tokenize(out, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
        out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
        out_tokens = list(filter(('[SEP]').__ne__, out_tokens))
        if out == '':
            print('output is empty, fill ""')

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        mscore = meteor_score([gt_tokens], out_tokens)
        meteor_scores.append(mscore)

    bleu2 = corpus_bleu(references, hypotheses, weights=(.5,.5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(.25,.25,.25,.25))

    print('BLEU-2 score:', bleu2)
    print('BLEU-4 score:', bleu4)
    _meteor_score = np.mean(meteor_scores)
    print('Average Meteor score:', _meteor_score)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    rouge_scores = []

    references = []
    hypotheses = []

    for i, (gt, out) in enumerate(outputs):

        rs = scorer.score(out, gt)
        rouge_scores.append(rs)

    print('ROUGE score:')
    rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores])
    rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores])
    rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores])
    print('rouge1:', rouge_1)
    print('rouge2:', rouge_2)
    print('rougeL:', rouge_l)
    return bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score


def evaluate_from_json(json_path=None, text_model='/fs_mol/guojianz/projects/FunMG/LLM/model_weight/models--Sihangli--3D-MoLM/scibert_scivocab_uncased'):
    with open(json_path, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]
    data = pd.DataFrame(records)
    data['predict_extrac'] = data['predict'].apply(extract_solution)
    data['output'] = data['predict_extrac'].apply(lambda x: x.rsplit('.', 1)[0] + '.' if isinstance(x, str) else x)
    data['output'] = data['output'].fillna('')
    num_none = data['output'].isna().sum()
    print(f"Number of None in predict: {num_none}")
    data['ground truth'] = data['label']
    text_tokenizer = BertTokenizerFast.from_pretrained(text_model)
    data['output_token_length'] = data['output'].apply(
        lambda x: len(text_tokenizer.tokenize(x)) if isinstance(x, str) else 0
    )

    print(data['output_token_length'].describe())
    bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score = evaluate_text_metrics(data)
    return {
        "bleu2": bleu2,
        "bleu4": bleu4,
        "meteor": _meteor_score,
        "rouge1": rouge_1,
        "rouge2": rouge_2,
        "rougeL": rouge_l,
    }



if __name__ == "__main__":
    evaluate_from_json(
        json_path="xxxx",
    )
