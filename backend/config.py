import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    BASE_DIR = BASE_DIR
    SECRET_KEY = os.environ.get('SECRET_KEY', 'atbu-ocr-secret-key-2026')
    DATABASE_URL = os.environ.get('DATABASE_URL', f"sqlite:///{BASE_DIR / 'academic_records.db'}")
    
    # Uploads & Artifacts
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', str(BASE_DIR / 'uploads'))
    DATASET_FOLDER = os.environ.get('DATASET_FOLDER', str(BASE_DIR / 'dataset'))
    MODELS_DIR = os.environ.get('MODELS_DIR', str(BASE_DIR / 'models_cache'))
    
    # OCR Settings
    TESSERACT_CMD = os.environ.get('TESSERACT_CMD', 'tesseract')
    TESSERACT_AVAILABLE = False  # Checked dynamically at runtime
    
    # Confidence threshold for human review
    CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', '0.80'))
    
    # Institutional formats (ATBU Bauchi)
    INSTITUTION_NAME = "Abubakar Tafawa Balewa University, Bauchi"
    FACULTY_DEFAULT = "Faculty of Computing"
    DEPT_DEFAULT = "Department of Computer Science"
    
    # Validation Rules
    REG_NO_REGEX = r'^\d{2}/\d{4,5}[A-Z]/\d$'
    SESSION_REGEX = r'^\d{4}/\d{4}$'
    COURSE_CODE_REGEX = r'^[A-Z]{3}\s*\d{3}$'
    CREDIT_UNIT_MIN = 1
    CREDIT_UNIT_MAX = 6
    SCORE_MIN = 0
    SCORE_MAX = 100
    VALID_GRADES = ['A', 'B', 'C', 'D', 'E', 'F']
    
    # Standard ATBU Grade Boundaries
    GRADE_SYSTEM = {
        'A': (70, 100, 5.0),
        'B': (60, 69, 4.0),
        'C': (50, 59, 3.0),
        'D': (45, 49, 2.0),
        'E': (40, 44, 1.0),
        'F': (0, 39, 0.0),
    }

config = Config()
