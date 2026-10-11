# ATBU Legacy Academic Record OCR Digitization System

A deep learning-based Optical Character Recognition system designed to digitize legacy academic records from the Faculty of Computing, Abubakar Tafawa Balewa University (ATBU), Bauchi.

Developed in accordance with the experimental machine learning methodology defined in **Gbadebo, Arafat (20/57673U/1)** Chapter Three.

---

## Key Features

- **Document Support:** Departmental Result Broadsheets and Student Transcript Sheets.
- **Image Preprocessing (OpenCV):** Automatic deskewing, bilateral denoising, CLAHE contrast enhancement, and adaptive binarization for faded typewriter records.
- **Layout & Table Analysis:** Automatic localization of headers, tabular matrices, row-column coordinate mapping, and cell extraction with visual overlay inspection.
- **Transformer Recognition (TrOCR):** VisionEncoderDecoder deep learning model fine-tuned on legacy academic records with Tesseract baseline comparison.
- **Information Extraction & Validation:** Automated institutional validation (ATBU Reg No format, course codes, 1-6 credit unit bounds, 5-point grading consistency), confidence scoring, and human verification flagging.
- **University Records Web Interface:** Clean executive dashboard with multi-stage image inspection, field review modals, student search, and test evaluation metrics.

---

## Project Structure

```
├── backend/
│   ├── app.py                     # Flask REST API & Web Server
│   ├── config.py                  # System Configuration & Validation Rules
│   ├── database.py                # Database Schema & Query Layer (SQLite / PostgreSQL)
│   ├── ocr/
│   │   ├── preprocessing.py       # OpenCV Deskewing, CLAHE, Adaptive Binarization
│   │   ├── layout.py              # Layout Regions & Table Grid Structure Detector
│   │   ├── recognition.py         # TrOCR & Tesseract OCR Engines
│   │   └── extractor.py           # Structured Field Extraction & Validation Rules
│   └── evaluation/
│       ├── metrics.py             # CER, WER, Field Accuracy, Speedup Calculations
│       └── benchmark.py           # Evaluation Runner on Held-Out Test Set
├── dataset/
│   ├── generator.py               # 100-Page Legacy Record Generator (Table 3.1)
│   ├── annotations.py             # Schema & Leakage-Free Split Verifier
│   ├── pages/                     # Generated 100 PNG Scans
│   └── annotations/               # Ground-Truth Transcriptions & Coordinates
├── training/
│   └── train_trocr.py             # PyTorch TrOCR Fine-Tuning Script with Checkpoints
├── frontend/
│   ├── index.html                 # Executive Web Portal Interface
│   ├── styles.css                 # Clean University Theme Stylesheet
│   └── app.js                     # Single-Page Application Client Logic
├── tests/
│   └── test_system.py             # Automated Unit & Integration Test Suite
└── docs/
    ├── architecture_and_methodology.md
    ├── api_documentation.md
    └── evaluation_report.md
```

---

## Quickstart Instructions

### 1. Requirements & Setup
Ensure Python 3.10+ is installed. Dependencies have been configured in the local environment.

To test database initialization and dataset validation:
```bash
python dataset/annotations.py
```

### 2. Run Automated Test Suite
```bash
python -m unittest tests/test_system.py
```

### 3. Run the Web Application
Start the Flask application server:
```bash
python backend/app.py
```
Open your browser and navigate to:
```
http://localhost:5000
```

---

## Reproducing Experiments

1. **Dataset Regeneration (100 Pages):**
   ```bash
   python dataset/generator.py
   ```
2. **TrOCR Fine-Tuning (with Checkpointing):**
   ```bash
   python training/train_trocr.py
   ```
3. **Held-Out Test Set Evaluation:**
   ```bash
   python -m backend.evaluation.benchmark
   ```
