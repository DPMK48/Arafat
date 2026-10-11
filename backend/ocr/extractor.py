import re
import json
from backend.config import config

class AcademicInformationExtractor:
    """
    Structured Academic Record Information Extractor.
    Applies institutional validation rules (ATBU Bauchi), score-grade consistency checks,
    computes field confidence scores, and flags uncertain fields for human review.
    """
    def __init__(self):
        self.reg_no_pattern = re.compile(config.REG_NO_REGEX)
        self.session_pattern = re.compile(config.SESSION_REGEX)
        self.course_pattern = re.compile(config.COURSE_CODE_REGEX)
        self.threshold = config.CONFIDENCE_THRESHOLD

    def validate_reg_no(self, val):
        if not val:
            return False, "Registration number missing"
        clean = val.strip().replace(" ", "")
        if self.reg_no_pattern.match(clean):
            return True, None
        # Flexible match for ATBU format
        if re.match(r'^\d{2}/\d{3,6}[A-Z]/\d$', clean):
            return True, None
        return False, "Does not match institutional pattern (e.g. 20/57673U/1)"

    def validate_session(self, val):
        if not val:
            return False, "Session missing"
        clean = val.strip()
        if self.session_pattern.match(clean):
            return True, None
        return False, "Does not match session pattern (e.g. 2021/2022)"

    def validate_course_code(self, val):
        if not val:
            return False, "Course code missing"
        clean = val.strip().upper()
        if self.course_pattern.match(clean) or re.match(r'^[A-Z]{2,4}\s*\d{3}$', clean):
            return True, None
        return False, "Invalid course code format (e.g. CSC 101)"

    def validate_score_and_grade(self, score_val, grade_val):
        notes = []
        is_valid = True
        
        # Validate score
        score_num = None
        try:
            score_num = float(score_val)
            if score_num < config.SCORE_MIN or score_num > config.SCORE_MAX:
                is_valid = False
                notes.append(f"Score {score_num} out of valid range (0-100)")
        except (ValueError, TypeError):
            is_valid = False
            notes.append("Score is not numeric")

        # Validate grade
        grade_clean = str(grade_val).strip().upper() if grade_val else ""
        if grade_clean not in config.VALID_GRADES:
            is_valid = False
            notes.append(f"Grade '{grade_clean}' is not a valid grade symbol (A-F)")

        # Validate score-grade consistency
        if score_num is not None and grade_clean in config.GRADE_SYSTEM:
            min_s, max_s, _ = config.GRADE_SYSTEM[grade_clean]
            if not (min_s <= score_num <= max_s):
                # Flag inconsistency
                is_valid = False
                notes.append(f"Inconsistent: Score {score_num} does not match expected grade {grade_clean} ({min_s}-{max_s})")

        return is_valid, "; ".join(notes) if notes else None

    def validate_credit_unit(self, val):
        try:
            cu = int(val)
            if config.CREDIT_UNIT_MIN <= cu <= config.CREDIT_UNIT_MAX:
                return True, None
            return False, f"Credit unit {cu} out of range (1-6)"
        except Exception:
            return False, "Credit unit must be an integer"

    def process_raw_document(self, doc_type, ground_truth_or_ocr, base_confidence=0.92):
        """
        Processes OCR/annotation results into structured academic fields with confidence scoring.
        """
        extracted_fields = []
        students = []
        courses = []
        results = []

        session_str = ground_truth_or_ocr.get("session", "2021/2022")
        semester_str = ground_truth_or_ocr.get("semester", "First Semester")

        # Session field
        s_valid, s_notes = self.validate_session(session_str)
        s_conf = base_confidence if s_valid else 0.55
        extracted_fields.append({
            "field_name": "session",
            "field_value": session_str,
            "confidence": s_conf,
            "is_valid": 1 if s_valid else 0,
            "validation_notes": s_notes,
            "needs_review": 1 if (not s_valid or s_conf < self.threshold) else 0
        })

        # Semester field
        sem_valid = semester_str in ["First Semester", "Second Semester", "1st Semester", "2nd Semester"]
        sem_conf = base_confidence if sem_valid else 0.60
        extracted_fields.append({
            "field_name": "semester",
            "field_value": semester_str,
            "confidence": sem_conf,
            "is_valid": 1 if sem_valid else 0,
            "validation_notes": None if sem_valid else "Unrecognized semester designation",
            "needs_review": 1 if (not sem_valid or sem_conf < self.threshold) else 0
        })

        if doc_type == "result_broadsheet":
            # Extract broadsheet student rows
            records = ground_truth_or_ocr.get("records", [])
            for r in records:
                reg = r.get("reg_no", "")
                name = r.get("student_name", "")
                reg_valid, reg_notes = self.validate_reg_no(reg)
                reg_conf = base_confidence if reg_valid else 0.50

                extracted_fields.append({
                    "field_name": "reg_no",
                    "field_value": reg,
                    "confidence": reg_conf,
                    "bbox": json.dumps(r.get("reg_no_bbox", [])),
                    "is_valid": 1 if reg_valid else 0,
                    "validation_notes": reg_notes,
                    "needs_review": 1 if (not reg_valid or reg_conf < self.threshold) else 0,
                    "row_idx": r.get("sn", 0),
                    "col_idx": 1
                })

                name_valid = len(name) >= 3 and "," in name
                name_conf = base_confidence if name_valid else 0.65
                extracted_fields.append({
                    "field_name": "student_name",
                    "field_value": name,
                    "confidence": name_conf,
                    "bbox": json.dumps(r.get("student_name_bbox", [])),
                    "is_valid": 1 if name_valid else 0,
                    "validation_notes": None if name_valid else "Format should be SURNAME, FIRSTNAME",
                    "needs_review": 1 if (not name_valid or name_conf < self.threshold) else 0,
                    "row_idx": r.get("sn", 0),
                    "col_idx": 2
                })

                students.append({
                    "reg_no": reg,
                    "name": name,
                    "department": ground_truth_or_ocr.get("department", "Computer Science"),
                    "faculty": ground_truth_or_ocr.get("faculty", "Faculty of Computing")
                })

                for res in r.get("results", []):
                    cc = res.get("course_code", "")
                    score = res.get("score", 0)
                    grade = res.get("grade", "F")
                    cu = res.get("credit_unit", 3)

                    sg_valid, sg_notes = self.validate_score_and_grade(score, grade)
                    sg_conf = base_confidence if sg_valid else 0.58

                    extracted_fields.append({
                        "field_name": f"{cc}_score_grade",
                        "field_value": f"{score} [{grade}]",
                        "confidence": sg_conf,
                        "bbox": json.dumps(res.get("bbox", [])),
                        "is_valid": 1 if sg_valid else 0,
                        "validation_notes": sg_notes,
                        "needs_review": 1 if (not sg_valid or sg_conf < self.threshold) else 0,
                        "row_idx": r.get("sn", 0),
                        "col_idx": 3
                    })

                    courses.append({
                        "course_code": cc,
                        "course_title": res.get("course_title", cc),
                        "credit_unit": cu
                    })

                    results.append({
                        "reg_no": reg,
                        "student_name": name,
                        "course_code": cc,
                        "session": session_str,
                        "semester": semester_str,
                        "score": score,
                        "grade": grade,
                        "credit_unit": cu
                    })

        else:
            # Transcript sheet
            reg = ground_truth_or_ocr.get("reg_no", "")
            name = ground_truth_or_ocr.get("student_name", "")
            
            reg_valid, reg_notes = self.validate_reg_no(reg)
            reg_conf = base_confidence if reg_valid else 0.52
            extracted_fields.append({
                "field_name": "reg_no",
                "field_value": reg,
                "confidence": reg_conf,
                "bbox": json.dumps([650, 160, 350, 40]),
                "is_valid": 1 if reg_valid else 0,
                "validation_notes": reg_notes,
                "needs_review": 1 if (not reg_valid or reg_conf < self.threshold) else 0
            })

            name_valid = len(name) >= 3 and "," in name
            name_conf = base_confidence if name_valid else 0.65
            extracted_fields.append({
                "field_name": "student_name",
                "field_value": name,
                "confidence": name_conf,
                "bbox": json.dumps([80, 160, 400, 40]),
                "is_valid": 1 if name_valid else 0,
                "validation_notes": None if name_valid else "Name formatting requires check",
                "needs_review": 1 if (not name_valid or name_conf < self.threshold) else 0
            })

            students.append({
                "reg_no": reg,
                "name": name,
                "department": ground_truth_or_ocr.get("department", "Computer Science"),
                "faculty": ground_truth_or_ocr.get("faculty", "Faculty of Computing")
            })

            for r in ground_truth_or_ocr.get("results", []):
                cc = r.get("course_code", "")
                score = r.get("score", 0)
                grade = r.get("grade", "F")
                cu = r.get("credit_unit", 3)

                sg_valid, sg_notes = self.validate_score_and_grade(score, grade)
                sg_conf = base_confidence if sg_valid else 0.58

                extracted_fields.append({
                    "field_name": f"{cc}_score_grade",
                    "field_value": f"{score} ({grade})",
                    "confidence": sg_conf,
                    "bbox": json.dumps(r.get("bbox", [])),
                    "is_valid": 1 if sg_valid else 0,
                    "validation_notes": sg_notes,
                    "needs_review": 1 if (not sg_valid or sg_conf < self.threshold) else 0,
                    "row_idx": r.get("sn", 0)
                })

                courses.append({
                    "course_code": cc,
                    "course_title": r.get("course_title", cc),
                    "credit_unit": cu
                })

                results.append({
                    "reg_no": reg,
                    "student_name": name,
                    "course_code": cc,
                    "session": session_str,
                    "semester": semester_str,
                    "score": score,
                    "grade": grade,
                    "credit_unit": cu
                })

        # Deduplicate courses
        unique_courses = {}
        for c in courses:
            unique_courses[c["course_code"]] = c

        return {
            "extracted_fields": extracted_fields,
            "students": students,
            "courses": list(unique_courses.values()),
            "results": results,
            "review_required": any(f["needs_review"] == 1 for f in extracted_fields)
        }
