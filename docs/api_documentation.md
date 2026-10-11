# ATBU Legacy Records OCR - REST API Documentation

Base URL: `http://localhost:5000`

---

## Endpoints

### 1. Dashboard Statistics
- **GET** `/api/dashboard/stats`
- **Response:**
```json
{
  "status": "success",
  "data": {
    "total_documents": 5,
    "completed_documents": 1,
    "review_required_documents": 0,
    "fields_requiring_review": 0,
    "digitized_students": 16,
    "digitized_results": 80,
    "recent_documents": [...]
  }
}
```

### 2. Document Upload
- **POST** `/api/upload`
- **Form Data:**
  - `file`: Scanned document image or PDF
  - `document_type`: `result_broadsheet` or `transcript_sheet`
- **Response:** `201 Created` with document metadata.

### 3. List Documents
- **GET** `/api/documents`
- **Response:** List of all documents with status (`uploaded`, `processing`, `completed`, `review_required`, `failed`).

### 4. Get Document Details
- **GET** `/api/documents/<int:id>`
- **Response:** Full document record, image file paths, extracted fields with confidence and review flags, and historical staff corrections.

### 5. Execute OCR Processing Pipeline
- **POST** `/api/documents/<int:id>/process`
- **Description:** Runs OpenCV preprocessing, layout analysis, TrOCR recognition, field extraction, validation rules, and persistence.
- **Response:**
```json
{
  "status": "success",
  "data": {
    "document_id": 1,
    "status": "completed",
    "processing_time_sec": 4.45,
    "preprocessed_image": "preproc_1_1791759503.png",
    "layout_image": "layout_1_1791759503.png",
    "fields_count": 114,
    "review_required": false
  }
}
```

### 6. Review & Update Field
- **PUT** `/api/fields/<int:id>`
- **Request Body:**
```json
{
  "field_value": "20/57673U/1",
  "reason": "Verified against physical university archive sheet"
}
```
- **Response:** Updates field value, clears review flag, and creates an audit record in `correction_history`.

### 7. Search Academic Records
- **GET** `/api/records/search?q=<term>`
- **Response:** Matching records filtered by student name, registration number, or course code.

### 8. Evaluation Benchmark
- **GET** `/api/evaluation`
- **POST** `/api/evaluation/run`
- **Description:** Returns or re-runs empirical benchmarks on the 20 held-out test pages.
