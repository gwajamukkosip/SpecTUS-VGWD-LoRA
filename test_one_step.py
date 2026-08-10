import json
import torch
from pathlib import Path

from spectus.model.modeling_spectus import SpectusForConditionalGeneration
from spectus.utils.data_utils import preprocess_datapoint, SpectroDataCollator
from spectus.utils.general_utils import build_tokenizer


# ==========================================
# 1. 설정
# ==========================================

checkpoint = "checkpoints/SpecTUS_pretrained"
train_path = Path("data/vgwd/train.jsonl")

device = "cuda"

print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# 2. Tokenizer
# ==========================================

tokenizer = build_tokenizer(
    "tokenizer/tokenizer_mf10M.model"
)


# ==========================================
# 3. 첫 번째 데이터 읽기
# ==========================================

with open(train_path, "r", encoding="utf-8") as f:
    sample = json.loads(f.readline())


# ==========================================
# 4. SpecTUS 전처리
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

processed = preprocess_datapoint(
    sample.copy(),
    source_token="<nist>",
    preprocess_args=preprocess_args
)

print("전처리 완료")
print("input_ids 길이:", len(processed["input_ids"]))
print("labels 길이:", len(processed["labels"]))


# ==========================================
# 5. Batch 만들기
# ==========================================

collator = SpectroDataCollator(
    inference_mode=False,
    restrict_intensities=False
)

batch = collator([processed])

batch = {
    k: v.to(device)
    for k, v in batch.items()
}


# ==========================================
# 6. 모델 로드
# ==========================================

print("\n모델 로드 중...")

model = SpectusForConditionalGeneration.from_pretrained(
    checkpoint
)

model.to(device)

model.train()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5
)

optimizer.zero_grad()

print("모델 GPU 로드 완료")


# ==========================================
# 7. GPU 메모리 초기화
# ==========================================

torch.cuda.reset_peak_memory_stats()


# ==========================================
# 8. Forward
# ==========================================

print("\nForward 시작")

outputs = model(**batch)

loss = outputs.loss

print("Loss:", loss.item())


# ==========================================
# 9. Backward
# ==========================================

print("Backward 시작")

loss.backward()

print("Optimizer step 시작")

optimizer.step()

print("Optimizer step 완료")

print("Backward 완료")


# ==========================================
# 10. GPU 메모리 확인
# ==========================================

allocated = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3
peak = torch.cuda.max_memory_allocated() / 1024**3

print("\n====================================")
print("1-step 테스트 성공")
print("====================================")

print("현재 GPU 사용량:", round(allocated, 2), "GB")
print("예약 GPU 메모리:", round(reserved, 2), "GB")
print("Peak GPU 사용량:", round(peak, 2), "GB")