
import re

import numpy as np
from Levenshtein import distance as lev
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import AllChem, Descriptors, MACCSkeys

rdBase.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.*')
from collections import Counter
import os
os.system("pip install selfies")
import selfies as sf
from EFGs import mol2frag
from rdkit import Chem
from nltk.translate.bleu_score import corpus_bleu

def extract_solution(solution_str: str) -> str:
    """
    Extract the last <answer>...</answer> block from the solution string.
    """
    matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    return matches[-1].group(1).strip() if matches else None

def standardize_smi(smi: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except:
        return None

def is_valid_smiles(smi: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(smi)
        return mol is not None
    except:
        return False


def exact_string_match(pred_smi: str, gt_smi: str) -> float:
    try:
        can_pred = Chem.MolToSmiles(Chem.MolFromSmiles(pred_smi), canonical=True)
        can_gt = Chem.MolToSmiles(Chem.MolFromSmiles(gt_smi), canonical=True)
        return 1.0 if can_pred == can_gt else 0.0
    except:
        return 0.0


def exact_structure_match(pred_smi: str, gt_smi: str) -> float:
    try:
        m1, m2 = Chem.MolFromSmiles(pred_smi), Chem.MolFromSmiles(gt_smi)
        return 1.0 if Chem.MolToInchi(m1) == Chem.MolToInchi(m2) else 0.0
    except:
        return 0.0


def property_similarity(pred_smi: str, gt_smi: str) -> float:
    try:
        mol1, mol2 = Chem.MolFromSmiles(pred_smi), Chem.MolFromSmiles(gt_smi)
        props1 = np.array([Descriptors.MolWt(mol1), Descriptors.MolLogP(mol1), Descriptors.TPSA(mol1)])
        props2 = np.array([Descriptors.MolWt(mol2), Descriptors.MolLogP(mol2), Descriptors.TPSA(mol2)])
        diff = np.abs(props1 - props2)
        return float(np.exp(-np.mean(diff) / 10))
    except:
        return 0.0


def fingerprint_similarity_scores(pred_smi: str, gt_smi: str):
    try:
        mol1, mol2 = Chem.MolFromSmiles(pred_smi), Chem.MolFromSmiles(gt_smi)
        maccs_sim = DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(mol1), MACCSkeys.GenMACCSKeys(mol2))
        rdk_sim = DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(mol1), Chem.RDKFingerprint(mol2))
        morgan_sim = DataStructs.TanimotoSimilarity(
            AllChem.GetMorganFingerprint(mol1, 2),
            AllChem.GetMorganFingerprint(mol2, 2)
        )
        return maccs_sim, rdk_sim, morgan_sim
    except:
        return 0.0, 0.0, 0.0


def smiles_levenshtein(pred_smi: str, gt_smi: str, normalize_len: int = 100) -> float:
    try:
        return 1.0 - lev(pred_smi, gt_smi) / normalize_len
    except:
        return 0.0


def compute_fragment_overlap(pred_smi: str, gt_smi: str) -> float:
    try:
        mol_pred = Chem.MolFromSmiles(pred_smi)
        mol_gt = Chem.MolFromSmiles(gt_smi)
        if mol_pred is None or mol_gt is None:
            return 0.0

        pred_nonCHs, pred_CHs = mol2frag(mol_pred)
        gt_nonCHs, gt_CHs = mol2frag(mol_gt)

        pred_frags = set(pred_nonCHs + pred_CHs)
        gt_frags = set(gt_nonCHs + gt_CHs)

        if not pred_frags or not gt_frags:
            return 0.0

        intersect = pred_frags & gt_frags
        union = pred_frags | gt_frags
        return len(intersect) / len(union)
    except Exception as e:
        print(f"[fragment reward error] {e}")
        return 0.0
        
def fragment_recall(pred_smi: str, gt_smi: str) -> float:
    try:
        mol_pred = Chem.MolFromSmiles(pred_smi)
        mol_gt = Chem.MolFromSmiles(gt_smi)
        pred_nonCHs, pred_CHs = mol2frag(mol_pred)
        gt_nonCHs, gt_CHs = mol2frag(mol_gt)
        pred_frags = set(pred_nonCHs + pred_CHs)
        gt_frags = set(gt_nonCHs + gt_CHs)
        if not gt_frags:
            return 0.0
        return len(pred_frags & gt_frags) / len(gt_frags)
    except:
        return 0.0

def functional_group_count_diff(pred_smi: str, gt_smi: str) -> float:
    try:
        mol1 = Chem.MolFromSmiles(pred_smi)
        mol2 = Chem.MolFromSmiles(gt_smi)
        pred_nonCHs, _ = mol2frag(mol1)
        gt_nonCHs, _ = mol2frag(mol2)
        pred_count = Counter(pred_nonCHs)
        gt_count = Counter(gt_nonCHs)
        all_keys = set(pred_count) | set(gt_count)
        diff_sum = sum(abs(pred_count[k] - gt_count[k]) for k in all_keys)
        total = sum(gt_count.values()) + 1e-5
        return float(np.exp(-diff_sum / total))  
    except:
        return 0.0

def compute_conjugated_ring_score(pred_smi, ground_truth):
    pred_mol = Chem.MolFromSmiles(pred_smi)
    gt_mol = Chem.MolFromSmiles(ground_truth)
    def get_conj_nonconj_ring_count(mol):
        if mol is None:
            return 0, 0
        rings = Chem.GetSymmSSSR(mol)
        conj, nonconj = 0, 0
        for ring in rings:
            bond_types = [
                mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)]).GetIsConjugated()
                for i in range(len(ring))
            ]
            if all(bond_types):
                conj += 1
            else:
                nonconj += 1
        return conj, nonconj

    conj_pred, nonconj_pred = get_conj_nonconj_ring_count(pred_mol)
    conj_gt, nonconj_gt = get_conj_nonconj_ring_count(gt_mol)

    total_pred = conj_pred + nonconj_pred
    total_gt = conj_gt + nonconj_gt

    if total_pred == 0 or total_gt == 0:
        return 0.0, 0.0

    ring_score = np.exp(-abs((conj_pred + nonconj_pred) - (conj_gt + nonconj_gt)))

    ratio_pred = conj_pred / total_pred
    ratio_gt = conj_gt / total_gt
    conj_ratio_score = np.exp(-abs(ratio_pred - ratio_gt))

    return ring_score, conj_ratio_score


def compute_score_format_acc(solution_str: str, ground_truth: str) -> float:
    # pred selfies convert
    pred_selfies = extract_solution(solution_str)
    if pred_selfies is None:
        return 0.0
    try:
        pred_smi = sf.decoder(pred_selfies)
    except Exception as e:
        return 0.0
    if not is_valid_smiles(pred_smi):
        return 0.0
    pred_smi = standardize_smi(pred_smi)

    # groundtruth selfies convert
    ground_truth_smi = sf.decoder(ground_truth)
    ground_truth_smi = standardize_smi(ground_truth_smi)
    if pred_smi is None or ground_truth_smi is None:
        return 0.0
    exact_text = exact_string_match(pred_smi, ground_truth_smi)
    exact_struct = exact_structure_match(pred_smi, ground_truth_smi)
    if exact_struct == 1.0 or exact_text == 1.0:
        return 2.0
    else:
        return 0.5


def compute_score_default(solution_str: str, ground_truth: str) -> float:
    # pred selfies convert
    pred_selfies = extract_solution(solution_str)
    if pred_selfies is None:
        return 0.0
    try:
        pred_smi = sf.decoder(pred_selfies)
    except Exception as e:
        return 0.0
    if not is_valid_smiles(pred_smi):
        return 0.0
    pred_smi = standardize_smi(pred_smi)

    # groundtruth selfies convert
    ground_truth_smi = sf.decoder(ground_truth)
    ground_truth_smi = standardize_smi(ground_truth_smi)
    if pred_smi is None or ground_truth_smi is None:
        return 0.0
    
    # bleu
    gt_self_tokens = [c for c in ground_truth]  
    out_self_tokens = [c for c in pred_selfies]  
    
    references_self = [[gt_self_tokens]]  
    hypotheses_self = [out_self_tokens]  
    bleu_score_self = corpus_bleu(references_self, hypotheses_self)


    exact_text = exact_string_match(pred_smi, ground_truth_smi)
    exact_struct = exact_structure_match(pred_smi, ground_truth_smi)
    maccs_sim, rdk_sim, morgan_sim = fingerprint_similarity_scores(pred_smi, ground_truth_smi)
    fp_score = (morgan_sim +  maccs_sim + rdk_sim) / 3.0

    frag_jaccard = compute_fragment_overlap(pred_smi, ground_truth_smi)
    frag_recall_score = fragment_recall(pred_smi, ground_truth_smi)
    frag_score = 0.5 * frag_jaccard + 0.5 * frag_recall_score

    group_score = functional_group_count_diff(pred_smi, ground_truth_smi)

    if exact_struct == 1.0 or exact_text == 1.0:
        return 2.0
    Rmol = (fp_score + frag_score + group_score + bleu_score_self) / 4.0
    return 0.5 + 1.5 * Rmol

def compute_score(solution_str: str, ground_truth: str, exp_method:str) -> float:
    if exp_method == "default":
        return compute_score_default(solution_str, ground_truth)
    elif exp_method == "format_acc":
        return compute_score_format_acc(solution_str, ground_truth)

