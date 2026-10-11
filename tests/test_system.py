import os
import unittest
import json
from pathlib import Path

from backend.config import config
from backend.database import init_db, execute_query, query_one
from backend.ocr.preprocessing import LegacyImagePreprocessor, PreprocessorConfig
from backend.ocr.layout import DocumentLayoutAnalyzer
from backend.ocr.extractor import AcademicInformationExtractor
from backend.evaluation.metrics import calculate_cer, calculate_wer, calculate_field_accuracy, calculate_baseline_improvement
from backend.app import app

class TestATBUOCRSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = app.test_client()
        cls.preprocessor = LegacyImagePreprocessor()
        cls.layout_analyzer = DocumentLayoutAnalyzer()
        cls.extractor = AcademicInformationExtractor()

    def test_database_initialization(self):
        docs_count = query_one("SELECT COUNT(*) as count FROM documents")
        self.assertIsNotNone(docs_count)
        self.assertIn("count", docs_count)

    def test_evaluation_metrics(self):
        # Exact match
        self.assertEqual(calculate_cer("20/57673U/1", "20/57673U/1"), 0.0)
        self.assertEqual(calculate_wer("CSC 101 PASS", "CSC 101 PASS"), 0.0)
        
        # Mismatch
        cer = calculate_cer("20/57673U/1", "20/57673U/I")
        self.assertGreater(cer, 0.0)

        # Baseline improvement
        gain = calculate_baseline_improvement(0.04, 0.20)
        self.assertEqual(gain, 80.0)

    def test_information_extraction_rules(self):
        # Valid ATBU registration numbers
        val, _ = self.extractor.validate_reg_no("20/57673U/1")
        self.assertTrue(val)
        val, _ = self.extractor.validate_reg_no("18/43210U/1")
        self.assertTrue(val)

        # Invalid registration number
        val, msg = self.extractor.validate_reg_no("INVALID_REG")
        self.assertFalse(val)

        # Session validation
        val, _ = self.extractor.validate_session("2021/2022")
        self.assertTrue(val)
        val, _ = self.extractor.validate_session("21-22")
        self.assertFalse(val)

        # Score & Grade validation
        val, _ = self.extractor.validate_score_and_grade(75, "A")
        self.assertTrue(val)

        # Inconsistent score and grade
        val, msg = self.extractor.validate_score_and_grade(42, "A")
        self.assertFalse(val)
        self.assertIn("Inconsistent", msg)

    def test_preprocessing_pipeline(self):
        # Test on a generated dataset sample
        sample_page = Path(config.DATASET_FOLDER) / "pages" / "broadsheet_authentic_001.png"
        if sample_page.exists():
            out_rgb, meta = self.preprocessor.process(sample_page)
            self.assertEqual(len(out_rgb.shape), 3)
            self.assertIn("processing_time_sec", meta)
            self.assertIn("skew_angle_deg", meta)

    def test_layout_analysis(self):
        sample_page = Path(config.DATASET_FOLDER) / "pages" / "broadsheet_authentic_001.png"
        if sample_page.exists():
            import cv2
            img = cv2.imread(str(sample_page))
            layout = self.layout_analyzer.detect_layout_regions(img)
            self.assertEqual(layout["document_type"], "result_broadsheet")
            table = self.layout_analyzer.detect_table_structure(img, layout["table_region"])
            self.assertGreater(table["num_rows"], 0)
            self.assertGreater(table["total_cells"], 0)

    def test_api_endpoints(self):
        # Stats
        r = self.client.get('/api/dashboard/stats')
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json["status"], "success")

        # Documents list
        r = self.client.get('/api/documents')
        self.assertEqual(r.status_code, 200)

        # Academic records list
        r = self.client.get('/api/records')
        self.assertEqual(r.status_code, 200)

        # Evaluation report
        r = self.client.get('/api/evaluation')
        self.assertEqual(r.status_code, 200)

if __name__ == '__main__':
    unittest.main()
