import json
from pathlib import Path

from spectus.utils.data_utils import preprocess_datapoint
from spectus.utils.general_utils import build_tokenizer


# ==========================================
# 1. tokenizer 준비
# ==========================================

tokenizer = build_tokenizer(
    "tokenizer/tokenizer_mf10M.model"
)


# ==========================================
# 2. SpecTUS 전처리 설정
# ==========================================

preprocess_args = {
    "tokenizer": tokenizer,
    "inference_mode": False,
    "restrict_intensities": False,
    "log_base": 1.28,
    "log_shift": 29,
    "do_log_binning": True,
    "mol_repr": "smiles"
}


# ==========================================
# 3. 한 파일 전체 검사 함수
# ==========================================

def validate_file(jsonl_path):

    total = 0
    success = 0
    failed = 0

    failed_records = []

    with open(jsonl_path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            total += 1

            try:
                sample = json.loads(line)

                processed = preprocess_datapoint(
                    sample.copy(),
                    source_token="<nist>",
                    preprocess_args=preprocess_args
                )

                # 필수 항목 확인
                if processed is None:
                    raise ValueError("processed result is None")

                if "input_ids" not in processed:
                    raise ValueError("input_ids 없음")

                if "position_ids" not in processed:
                    raise ValueError("position_ids 없음")

                if "labels" not in processed:
                    raise ValueError("labels 없음")

                # 길이 검사
                if len(processed["input_ids"]) != len(processed["position_ids"]):
                    raise ValueError(
                        "input_ids와 position_ids 길이 불일치"
                    )

                # labels 안에 None 있는지 검사
                if any(x is None for x in processed["labels"]):
                    raise ValueError("labels 안에 None 존재")

                success += 1

            except Exception as e:

                failed += 1

                failed_records.append({
                    "line": line_number,
                    "error": str(e)
                })


    return total, success, failed, failed_records


# ==========================================
# 4. Train 검사
# ==========================================

train_path = Path("data/vgwd/train.jsonl")

print("\n====================================")
print("TRAIN 검사 시작")
print("====================================")

train_total, train_success, train_failed, train_errors = validate_file(
    train_path
)

print("전체:", train_total)
print("성공:", train_success)
print("실패:", train_failed)


# ==========================================
# 5. Validation 검사
# ==========================================

valid_path = Path("data/vgwd/valid.jsonl")

print("\n====================================")
print("VALIDATION 검사 시작")
print("====================================")

valid_total, valid_success, valid_failed, valid_errors = validate_file(
    valid_path
)

print("전체:", valid_total)
print("성공:", valid_success)
print("실패:", valid_failed)


# ==========================================
# 6. 실패 내용 출력
# ==========================================

if train_failed > 0:

    print("\nTRAIN 실패 예시:")

    for item in train_errors[:10]:
        print(item)


if valid_failed > 0:

    print("\nVALIDATION 실패 예시:")

    for item in valid_errors[:10]:
        print(item)


print("\n====================================")
print("검사 완료")
print("====================================")