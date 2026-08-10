import json
from pathlib import Path

from spectus.utils.data_utils import preprocess_datapoint
from spectus.utils.general_utils import build_tokenizer


jsonl_path = Path("data/vgwd/train.jsonl")

with open(jsonl_path, "r", encoding="utf-8") as f:
    sample = json.loads(f.readline())


print("원본 데이터")
print("SMILES:", sample["smiles"])
print("Peak 개수:", len(sample["mz"]))


tokenizer = build_tokenizer(
    "tokenizer/tokenizer_mf10M.model"
)


# ==========================================
# 3. SpecTUS preprocessing 설정
# ==========================================

preprocess_args = {
    "tokenizer": tokenizer,

    # 학습 데이터이므로 False
    "inference_mode": False,

    # intensity도 사용
    "restrict_intensities": False,

    # SpecTUS final config와 동일
    "log_base": 1.28,
    "log_shift": 29,

    "do_log_binning": True,

    # SMILES를 정답으로 사용
    "mol_repr": "smiles"
}


# ==========================================
# 4. SpecTUS 방식으로 전처리
# ==========================================

processed = preprocess_datapoint(
    sample.copy(),
    source_token="<nist>",
    preprocess_args=preprocess_args
)


# ==========================================
# 5. 결과 출력
# ==========================================

print("\n====================================")
print("SpecTUS preprocessing 결과")
print("====================================")

print("input_ids 길이:", len(processed["input_ids"]))
print("position_ids 길이:", len(processed["position_ids"]))
print("labels 길이:", len(processed["labels"]))

print("\ninput_ids:")
print(processed["input_ids"])

print("\nposition_ids:")
print(processed["position_ids"])

print("\nmol_repr:")
print(processed["mol_repr"])

print("\nlabels:")
print(processed["labels"])
