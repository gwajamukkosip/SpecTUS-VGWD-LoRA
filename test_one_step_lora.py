import json
import torch
from pathlib import Path

from peft import LoraConfig, get_peft_model
from spectus.model.modeling_spectus import SpectusForConditionalGeneration
from spectus.utils.data_utils import preprocess_datapoint, SpectroDataCollator
from spectus.utils.general_utils import build_tokenizer


# ==========================================
# 1. 기본 설정
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
# 3. 첫 번째 VGWD 데이터 읽기
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
# 6. Pretrained SpecTUS 로드
# ==========================================

print("\nPretrained 모델 로드 중...")

model = SpectusForConditionalGeneration.from_pretrained(
    checkpoint
)

print("Pretrained 모델 로드 완료")


# ==========================================
# 7. LoRA 설정
# ==========================================

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "v_proj"
    ],

    bias="none"
)


# ==========================================
# 8. LoRA 적용
# ==========================================

model = get_peft_model(
    model,
    lora_config
)

print("\nLoRA 적용 완료")

model.print_trainable_parameters()


# ==========================================
# 9. GPU로 이동
# ==========================================

model.to(device)
model.train()


# ==========================================
# 10. 학습 가능한 파라미터만 optimizer에 전달
# ==========================================

trainable_parameters = [
    p for p in model.parameters()
    if p.requires_grad
]

optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=1e-4
)

optimizer.zero_grad()


# ==========================================
# 11. GPU 메모리 초기화
# ==========================================

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


# ==========================================
# 12. Forward
# ==========================================

print("\nForward 시작")

outputs = model(**batch)

loss = outputs.loss

print("Loss:", loss.item())


# ==========================================
# 13. Backward
# ==========================================

print("Backward 시작")

loss.backward()

print("Backward 완료")


# ==========================================
# 14. Optimizer step
# ==========================================

print("Optimizer step 시작")

optimizer.step()

print("Optimizer step 완료")


# ==========================================
# 15. GPU 메모리
# ==========================================

allocated = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3
peak = torch.cuda.max_memory_allocated() / 1024**3


print("\n====================================")
print("LoRA 1-step 테스트 성공")
print("====================================")

print("현재 GPU 사용량:", round(allocated, 2), "GB")
print("예약 GPU 메모리:", round(reserved, 2), "GB")
print("Peak GPU 사용량:", round(peak, 2), "GB")
