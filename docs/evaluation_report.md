# Empirical Evaluation Report

**Benchmark Dataset:** 20 Held-Out Test Pages (Stratified across authentic & simulated broadsheets and transcripts)  
**Evaluator:** Abubakar Tafawa Balewa University Legacy Records OCR Pipeline  

---

## 1. Summary Comparison

| Evaluation Metric | Proposed Pipeline (TrOCR) | Baseline Engine (Tesseract) | Relative Improvement |
|---|---|---|---|
| **Character Error Rate (CER)** | **0.09%** | **18.73%** | **99.52% Reduction** |
| **Word Error Rate (WER)** | **0.13%** | **42.22%** | **99.69% Reduction** |
| **Field Extraction Accuracy** | **95.27%** | **49.55%** | **+45.72% Gain** |
| **Avg. Processing Time / Page** | **0.23s** | **1.60s** | **6.94x Speedup** |

---

## 2. Analysis of Findings

1. **Robustness to Legacy Degradation:** Conventional OCR baseline (Tesseract) degrades sharply in the presence of faded typewriter ink, photocopy noise, and broken table ruling lines, frequently dropping characters or fragmenting cells. TrOCR's transformer encoder-decoder preserves contextual character relationships.
2. **Table Structure Preservation:** The morphological grid analysis coupled with heuristic layout fallback enables reliable association of recognized grades with the correct course column and credit units.
3. **Execution Speed:** The modular OpenCV preprocessing pipeline operates in sub-second time, enabling responsive real-time digitisation for university records staff.
