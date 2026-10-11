import rapidfuzz.distance.Levenshtein as lev

def calculate_cer(reference_text: str, hypothesis_text: str) -> float:
    """
    Character Error Rate (CER) = Levenshtein Distance / len(reference)
    """
    ref = str(reference_text or "").strip()
    hyp = str(hypothesis_text or "").strip()
    if not ref:
        return 1.0 if hyp else 0.0
    dist = lev.distance(ref, hyp)
    return float(dist / len(ref))

def calculate_wer(reference_text: str, hypothesis_text: str) -> float:
    """
    Word Error Rate (WER) = Word Edit Distance / len(reference_words)
    """
    ref_words = str(reference_text or "").strip().split()
    hyp_words = str(hypothesis_text or "").strip().split()
    if not ref_words:
        return 1.0 if hyp_words else 0.0
    dist = lev.distance(ref_words, hyp_words)
    return float(dist / len(ref_words))

def calculate_field_accuracy(ground_truth_dict: dict, extracted_dict: dict) -> float:
    """
    Field Extraction Accuracy = (Number of exactly matching fields) / (Total target fields) * 100
    """
    if not ground_truth_dict:
        return 0.0
    matches = 0
    total = len(ground_truth_dict)
    for key, val in ground_truth_dict.items():
        extracted_val = extracted_dict.get(key)
        if extracted_val is not None:
            # Case-insensitive whitespace-normalized exact match
            if str(val).strip().upper() == str(extracted_val).strip().upper():
                matches += 1
    return round((matches / total) * 100.0, 2)

def calculate_baseline_improvement(proposed_metric: float, baseline_metric: float) -> float:
    """
    Calculates relative reduction in error rate:
    Improvement (%) = ((Baseline - Proposed) / Baseline) * 100%
    """
    if baseline_metric <= 0:
        return 0.0
    reduction = ((baseline_metric - proposed_metric) / baseline_metric) * 100.0
    return round(reduction, 2)
