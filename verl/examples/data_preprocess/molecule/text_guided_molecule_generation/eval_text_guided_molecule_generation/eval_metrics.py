import os
import re
import sys
import json
import csv
import argparse
from collections import Counter
import os.path as osp
import numpy as np
import pandas as pd
from tqdm import tqdm
from Levenshtein import distance as lev
from nltk.translate.bleu_score import corpus_bleu
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys
RDLogger.DisableLog('rdApp.*')
import selfies as sf
from EFGs import mol2frag
np.random.seed(42)



def extract_solution(solution_str: str) -> str:
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str))
    if matches:
        return matches[-1].group(1).strip()
    return None

def sf_encode(selfies):
    try:
        smiles = sf.decoder(selfies)
        return smiles
    except Exception:
        return None

def convert_to_canonical_smiles(smiles):
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None:
            canonical_smiles = Chem.MolToSmiles(molecule, isomericSmiles=False, canonical=True)
            return canonical_smiles
    except Exception as e:
        return None
    
def evaluate_fingerprint(data, morgan_r=2):
    outputs = []
    bad_mols = 0
    for i, row in tqdm(data.iterrows(), total=len(data)):
        try:
            gt_smi = row['ground smiles']
            ot_smi = row['output_smiles']
                
                
            gt_m = Chem.MolFromSmiles(gt_smi)
            ot_m = Chem.MolFromSmiles(ot_smi) 

            if ot_m == None: raise ValueError('Bad SMILES')
            outputs.append((row['prompt'], gt_m, ot_m))
        except:
            bad_mols += 1
    validity_score = len(outputs)/(len(outputs)+bad_mols)
    print('validity:', validity_score)
    MACCS_sims = []
    morgan_sims = []
    RDK_sims = []
    enum_list = outputs
    for i, (desc, gt_m, ot_m) in enumerate(enum_list):
        if i % 100 == 0:
            print(i, 'processed.')
        MACCS_sims.append(DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(gt_m), MACCSkeys.GenMACCSKeys(ot_m), metric=DataStructs.TanimotoSimilarity))
        RDK_sims.append(DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(gt_m), Chem.RDKFingerprint(ot_m), metric=DataStructs.TanimotoSimilarity))
        morgan_sims.append(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(gt_m,morgan_r), AllChem.GetMorganFingerprint(ot_m, morgan_r)))
    maccs_sims_score = np.mean(MACCS_sims)
    rdk_sims_score = np.mean(RDK_sims)
    morgan_sims_score = np.mean(morgan_sims)
    print('Average MACCS Similarity:', maccs_sims_score)
    print('Average RDK Similarity:', rdk_sims_score)
    print('Average Morgan Similarity:', morgan_sims_score)

    return validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score

def evaluate_mol_translation(data):
    outputs = []
    for i, row in tqdm(data.iterrows(), total=len(data)):
        gt_self = row['ground truth']
        ot_self = row['output']
        gt_smi = row['ground smiles']
        ot_smi = row['output_smiles']
        outputs.append((row['prompt'], gt_self, ot_self, gt_smi, ot_smi))
    bleu_self_scores = []
    bleu_smi_scores = []
    references_self = []
    hypotheses_self = []
    references_smi = []
    hypotheses_smi = []
    for i, (des, gt_self, ot_self, gt_smi, ot_smi) in enumerate(outputs):
        if i % 100 == 0:
            print(i, 'processed.')
        gt_self_tokens = [c for c in gt_self]
        out_self_tokens = [c for c in ot_self] if ot_self else []

        references_self.append([gt_self_tokens])
        hypotheses_self.append(out_self_tokens)
        
        gt_smi_tokens = [c for c in gt_smi]
        ot_smi_tokens = [c for c in ot_smi] if ot_smi else []

        references_smi.append([gt_smi_tokens])
        hypotheses_smi.append(ot_smi_tokens)
    # BLEU score
    bleu_score_self =  corpus_bleu(references_self, hypotheses_self)
    print('SELFIES BLEU score:', bleu_score_self)

    references_self = []
    hypotheses_self = []
    references_smi = []
    hypotheses_smi = []
    levs_self = []
    levs_smi = []
    num_exact = 0
    bad_mols = 0
    frag_jaccards = []
    frag_recalls = []
    fg_diffs = []

    for i, (des, gt_self, ot_self, gt_smi, ot_smi) in enumerate(outputs):
        if ot_self is None or ot_smi is None:
            bad_mols += 1
            continue
        hypotheses_self.append(ot_self)
        references_self.append(gt_self)

        hypotheses_smi.append(ot_smi)
        references_smi.append(gt_smi)
        
        try:
            m_out = Chem.MolFromSmiles(ot_smi)
            m_gt = Chem.MolFromSmiles(gt_smi)

            if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt): num_exact += 1

            pred_nonCHs, pred_CHs = mol2frag(m_out)
            gt_nonCHs, gt_CHs = mol2frag(m_gt)
            pred_frags = set(pred_nonCHs + pred_CHs)
            gt_frags = set(gt_nonCHs + gt_CHs)
            # Frag-Jaccard
            frag_jaccards.append(len(pred_frags & gt_frags) / len(pred_frags | gt_frags) if pred_frags | gt_frags else 0.0)
            # Frag-Recall
            frag_recalls.append(len(pred_frags & gt_frags) / len(gt_frags) if gt_frags else 0.0)
            # FG-Diff
            pred_count = Counter(pred_nonCHs)
            gt_count = Counter(gt_nonCHs)
            all_keys = set(pred_count) | set(gt_count)
            diff_sum = sum(abs(pred_count[k] - gt_count[k]) for k in all_keys)
            total = sum(gt_count.values()) + 1e-5
            fg_diffs.append(np.exp(-diff_sum / total))

        except:
            bad_mols += 1

        levs_self.append(lev(ot_self, gt_self))
        levs_smi.append(lev(ot_smi, gt_smi))
    # Exact matching score
    exact_match_score = num_exact/(i+1)
    print('Exact Match:')
    print(exact_match_score)

    # Levenshtein score
    levenshtein_score_smi = np.mean(levs_smi)
    print('SMILES Levenshtein:')
    print(levenshtein_score_smi)

    avg_frag_j = np.mean(frag_jaccards) if frag_jaccards else None
    avg_frag_r = np.mean(frag_recalls) if frag_recalls else None
    avg_fg_diff = np.mean(fg_diffs) if fg_diffs else None


    return bleu_score_self, exact_match_score, levenshtein_score_smi, avg_frag_j,avg_frag_r, avg_fg_diff



def evaluate_from_json(json_path=None):
    with open(json_path, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]
    data = pd.DataFrame(records)    
    data['output'] = data['predict'].apply(extract_solution)
    data['output_smiles'] = data['output'].map(sf_encode)
    # data.dropna(axis=0, how='any', inplace=True)
    data['output_smiles'] = data['output_smiles'].map(convert_to_canonical_smiles)
    num_none_output = data['output_smiles'].isna().sum()
    num_valid_output = len(data) - num_none_output
    print(f"[Output SMILES] Valid: {num_valid_output}, None: {num_none_output}")
    # .apply(lambda x: x or '')
    # data.dropna(axis=0, how='any', inplace=True)
    data['ground truth'] = data['label']
    data['ground smiles'] = data['label'].map(sf_encode)
    data['ground smiles'] = data['ground smiles'].map(convert_to_canonical_smiles)
    # data.dropna(axis=0, how='any', inplace=True)
    data['ground smiles'] = data['ground smiles'].map(convert_to_canonical_smiles)
    num_none_gt = data['ground smiles'].isna().sum()
    num_valid_gt = len(data) - num_none_gt
    print(f"[Ground SMILES] Valid: {num_valid_gt}, None: {num_none_gt}")
    # fingerprint evaluation
    print("==== Fingerprint Evaluation ====")
    validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score = evaluate_fingerprint(data) 
    print("==== Molecular Translation Evaluation ====")
    bleu_score_self, exact_match_score, levenshtein_score_smi,  avg_frag_j, avg_frag_r, avg_fg_diff = evaluate_mol_translation(data)

    return {
        'bleu': bleu_score_self,
        'exact_match': exact_match_score,
        'levenshtein': levenshtein_score_smi,
        'rdk_similarity': rdk_sims_score,
        'maccs_similarity': maccs_sims_score,
        'morgan_similarity': morgan_sims_score,
        'frag_jaccard': avg_frag_j,  
        'frag_recall': avg_frag_r,   
        'fg_diff': avg_fg_diff,       
        'validity': validity_score,
    }


if __name__ == "__main__":
    evaluate_from_json(
        json_path="xxxx.json",
    )
