# System Architecture and Methodology

**Project Title:** Design and Implementation of a Deep Learning-Based Optical Character Recognition System for Digitizing Legacy Academic Records  
**Author:** Gbadebo, Arafat (20/57673U/1)  
**Institution:** Abubakar Tafawa Balewa University (ATBU), Bauchi  
**Department:** Computer Science, Faculty of Computing  

---

## 1. Introduction & Context

Legacy academic records at Abubakar Tafawa Balewa University (ATBU), Bauchi, exist primarily as physical paper documents:
- **Departmental Result Broadsheets** (multi-column matrices containing student scores across course offerings, GPA, and approval signatures).
- **Student Academic Transcript Sheets** (detailed semester-by-semester course codes, titles, credit units, scores, and grades with official registrar stamps).

Prior to this system, retrieving historical records required staff to physically search through archive folders, incurring delays of days to weeks, physical wear, and risk of loss.

---

## 2. System Architecture

The end-to-end digitization system is structured into five core stages:

```
+-----------------------------------------------------------------------+
|                        Scanned Legacy Record                          |
|         (Faded Typewriter, Skew, Noise, Rubber Stamps, Ruling)        |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                     OpenCV Preprocessing Pipeline                     |
|  - Resolution Check (min 600px dimension)                             |
|  - Deskewing (minAreaRect contour orientation)                        |
|  - Bilateral Filtering / Denoising                                    |
|  - Contrast-Limited Adaptive Histogram Equalization (CLAHE)           |
|  - Adaptive Gaussian Binarization & Morphological Line Repair         |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                      Layout & Table Analysis                          |
|  - Header / Institution Metadata Region Localization                  |
|  - Morphological Grid Detection (Horizontal & Vertical Line Kernels)  |
|  - Cell Segmentation & (Row, Column) Coordinate Mapping              |
|  - Configurable Heuristic Fallback for Degraded Rulings               |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                   TrOCR Deep Learning Recognition                     |
|  - VisionEncoderDecoder Transformer Architecture                      |
|  - Transfer Learning fine-tuned on ATBU Legacy Dataset                |
|  - Offline-Resilient Inference & Tesseract Baseline Benchmarking      |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|             Information Extraction & Rule Validation                  |
|  - ATBU Reg No Regex Verification (`^\d{2}/\d{4,5}[A-Z]/\d$`)         |
|  - Course Code & Credit Unit Boundary Checking (1-6 units)            |
|  - Score (0-100) & ATBU 5-Point Grade Consistency Check               |
|  - Confidence Score Formulation & Human Review Flagging               |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|               Relational Database & Verification Audit                |
|  - Students, Courses, Academic Sessions, Results                      |
|  - Documents, Extracted Fields, and Correction History Audit Trail    |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                    Executive University Web Portal                    |
|  - Dashboard, Scanned Document Upload, Multi-Stage Image Inspector,  |
|    Staff Verification Modal, Record Search, Scientific Evaluation     |
+-----------------------------------------------------------------------+
```

---

## 3. Dataset Composition and Split

Per Chapter Three (Table 3.1), the dataset comprises 100 pages stratified across degradation levels (low, medium, high) with zero data leakage:

| Category | Training | Validation | Held-Out Test | Total |
|---|---:|---:|---:|---:|
| Authentic Result Broadsheets | 18 | 6 | 6 | 30 |
| Authentic Transcript Sheets | 12 | 4 | 4 | 20 |
| Simulated Result Broadsheets | 18 | 6 | 6 | 30 |
| Simulated Transcript Sheets | 12 | 4 | 4 | 20 |
| **Total** | **60** | **20** | **20** | **100** |

---

## 4. Evaluation Protocol

The system is evaluated against the conventional OCR baseline (Tesseract) using four core metrics:
1. **Character Error Rate (CER):** Levenshtein distance on characters divided by ground-truth character count.
2. **Word Error Rate (WER):** Levenshtein distance on word sequences divided by ground-truth word count.
3. **Field Extraction Accuracy:** Percentage of extracted fields matching ground truth exactly.
4. **Processing Time per Page:** Wall-clock elapsed duration from upload to database persistence.
5. **Baseline Improvement:** Relative error reduction percentage $((Baseline - Proposed) / Baseline) \times 100\%$.
