import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
TEST_CASES_PATH = os.path.join(DATA_DIR, "test_cases.json")
ICD10_CHAPTERS_DEFINITION_PATH = os.path.join(DATA_DIR, "icd10_chapters_definition.csv")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
DATA_EXPLORATION_DIR = os.path.join(ROOT_DIR, "data_exploration")
SEED = 31415
TARGET_COLUMN = "expected_condition_icd10_blocks"
INDEX_COLUMN = "public_test_case_id"
