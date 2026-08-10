import pandas as pd
import json
from pathlib import Path


# ==========================================
# 경로 설정
# ==========================================

project_root = Path("..")

train_csv = project_root / "train.csv"
valid_csv = project_root / "validation.csv"
test_csv = project_root / "test.csv"

output_dir = Path("data/vgwd")
output_dir.mkdir(parents=True, exist_ok=True)


# ==========================================
# CSV → JSONL 변환 함수
# ==========================================

def convert_csv_to_jsonl(csv_path, jsonl_path):

    df = pd.read_csv(csv_path)

    print(f"\n처리 중: {csv_path}")
    print("데이터 개수:", len(df))

    with open(jsonl_path, "w", encoding="utf-8") as f:

        for _, row in df.iterrows():

            # CSV에서는 리스트가 문자열로 저장되어 있으므로 복원
            mz = json.loads(row["mz"])
            intensity = json.loads(row["intensity"])

            record = {
                "smiles": row["smiles"],
                "mz": mz,
                "intensity": intensity
            }

            f.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )

    print("저장 완료:", jsonl_path)


# ==========================================
# Train / Validation / Test 변환
# ==========================================

convert_csv_to_jsonl(
    train_csv,
    output_dir / "train.jsonl"
)

convert_csv_to_jsonl(
    valid_csv,
    output_dir / "valid.jsonl"
)

convert_csv_to_jsonl(
    test_csv,
    output_dir / "test.jsonl"
)


print("\n====================================")
print("VGWD JSONL 변환 완료")
print("====================================")