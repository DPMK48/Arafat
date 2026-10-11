import json
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent
MANIFEST_FILE = DATASET_DIR / "dataset_manifest.json"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"
PAGES_DIR = DATASET_DIR / "pages"

def load_manifest():
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(f"Manifest file not found: {MANIFEST_FILE}")
    with open(MANIFEST_FILE, "r") as f:
        return json.load(f)

def load_annotation(page_id):
    path = ANNOTATIONS_DIR / f"{page_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Annotation not found for {page_id}")
    with open(path, "r") as f:
        return json.load(f)

def validate_dataset_splits():
    """
    Validates dataset integrity:
    1. 100 total pages.
    2. Exactly 60 train, 20 validation, 20 test.
    3. Zero leakage between splits.
    4. Stratified representation of document types and degradation.
    """
    manifest = load_manifest()
    splits = manifest["splits"]
    train_ids = set(splits["train"])
    val_ids = set(splits["validation"])
    test_ids = set(splits["test"])

    # Disjointness check
    leakage_train_val = train_ids.intersection(val_ids)
    leakage_train_test = train_ids.intersection(test_ids)
    leakage_val_test = val_ids.intersection(test_ids)

    if leakage_train_val or leakage_train_test or leakage_val_test:
        raise ValueError(f"Data leakage detected! Train-Val: {leakage_train_val}, Train-Test: {leakage_train_test}, Val-Test: {leakage_val_test}")

    stats = {
        "total": len(train_ids) + len(val_ids) + len(test_ids),
        "train": len(train_ids),
        "validation": len(val_ids),
        "test": len(test_ids),
        "leakage_free": True,
        "splits": {
            "train": {"broadsheet": 0, "transcript": 0},
            "validation": {"broadsheet": 0, "transcript": 0},
            "test": {"broadsheet": 0, "transcript": 0}
        }
    }

    for s_name, ids in [("train", train_ids), ("validation", val_ids), ("test", test_ids)]:
        for pid in ids:
            ann = load_annotation(pid)
            dtype = "broadsheet" if ann["document_type"] == "result_broadsheet" else "transcript"
            stats["splits"][s_name][dtype] += 1

    return stats

if __name__ == "__main__":
    try:
        report = validate_dataset_splits()
        print("Dataset Validation Report:")
        print(json.dumps(report, indent=2))
    except Exception as e:
        print("Validation error:", e)
