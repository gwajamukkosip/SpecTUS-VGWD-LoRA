import json
import random
from pathlib import Path
from collections import defaultdict

from rdkit import Chem


# ==========================================
# 설정
# ==========================================

INPUT_FILES = [
    "data/vgwd/train.jsonl",
    "data/vgwd/valid.jsonl",
    "data/vgwd/test.jsonl",
]

OUTPUT_DIR = Path("data/vgwd_clean")

SEED = 42
TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10


# ==========================================
# stereochemistry 제거 + canonical SMILES
# ==========================================

def nonstereo_canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    Chem.RemoveStereochemistry(mol)

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=False
    )


# ==========================================
# 기존 split 전부 합치기
# ==========================================

records = []

for path in INPUT_FILES:

    with open(path, encoding="utf-8") as f:

        for line in f:

            d = json.loads(line)

            canon = nonstereo_canonical_smiles(
                d["smiles"]
            )

            if canon is None:
                continue

            d["_group_smiles"] = canon

            records.append(d)


print("전체 spectrum 수:", len(records))


# ==========================================
# 동일 구조끼리 그룹화
# ==========================================

groups = defaultdict(list)

for d in records:
    groups[d["_group_smiles"]].append(d)


group_keys = list(groups.keys())

print("고유 non-stereo 구조 수:", len(group_keys))


# ==========================================
# 구조 단위 shuffle
# ==========================================

random.seed(SEED)
random.shuffle(group_keys)


# ==========================================
# 80 : 10 : 10 split
# ==========================================

n_groups = len(group_keys)

n_train = int(n_groups * TRAIN_RATIO)
n_valid = int(n_groups * VALID_RATIO)

train_keys = set(
    group_keys[:n_train]
)

valid_keys = set(
    group_keys[n_train:n_train + n_valid]
)

test_keys = set(
    group_keys[n_train + n_valid:]
)


# ==========================================
# spectrum 레코드 배정
# ==========================================

train_records = []
valid_records = []
test_records = []

for key, rows in groups.items():

    if key in train_keys:
        train_records.extend(rows)

    elif key in valid_keys:
        valid_records.extend(rows)

    elif key in test_keys:
        test_records.extend(rows)


# ==========================================
# 내부 필드 제거
# ==========================================

def clean_records(rows):

    out = []

    for d in rows:

        d = dict(d)

        d.pop("_group_smiles", None)

        out.append(d)

    return out


train_records = clean_records(train_records)
valid_records = clean_records(valid_records)
test_records = clean_records(test_records)


# ==========================================
# 저장
# ==========================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def save_jsonl(path, rows):

    with open(path, "w", encoding="utf-8") as f:

        for d in rows:

            f.write(
                json.dumps(d) + "\n"
            )


save_jsonl(
    OUTPUT_DIR / "train.jsonl",
    train_records
)

save_jsonl(
    OUTPUT_DIR / "valid.jsonl",
    valid_records
)

save_jsonl(
    OUTPUT_DIR / "test.jsonl",
    test_records
)


# ==========================================
# 결과 출력
# ==========================================

print("\n====================================")
print("Clean split 완료")
print("====================================")

print("Train 구조 수:", len(train_keys))
print("Valid 구조 수:", len(valid_keys))
print("Test 구조 수:", len(test_keys))

print()

print("Train spectra:", len(train_records))
print("Valid spectra:", len(valid_records))
print("Test spectra:", len(test_records))


# ==========================================
# 데이터 누수 검사
# ==========================================

print("\n====================================")
print("구조 중복 검사")
print("====================================")

print(
    "Train ↔ Valid:",
    len(train_keys & valid_keys)
)

print(
    "Train ↔ Test:",
    len(train_keys & test_keys)
)

print(
    "Valid ↔ Test:",
    len(valid_keys & test_keys)
)
