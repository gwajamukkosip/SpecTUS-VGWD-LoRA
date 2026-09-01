#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if python -c 'import peft, torch, transformers' >/dev/null 2>&1; then
    PYTHON_CMD=(python)
elif command -v conda >/dev/null 2>&1; then
    PYTHON_CMD=(conda run --no-capture-output -n trainSpectus python)
else
    echo "SpecTUS Python environment not found." >&2
    exit 1
fi

BASE_CHECKPOINT="checkpoints/SpecTUS_pretrained"
VALID_CONFIG="configs/predict_vgwd_clean_rslora_r32_valid_beam10.yaml"
VALID_LABELS="data/vgwd_clean/valid_filtered_mz500.jsonl"
OUTPUT_DIR="analysis/rslora_r32_all_attention_validation"
SEEDS=(7 123 2026)
EXPECTED_ROWS=664

require_path() {
    [[ -e "$1" ]] || { echo "Required path not found: $1" >&2; exit 1; }
}

validate_seed() {
    case "$1" in
        7|123|2026) ;;
        *) echo "Seed must be one of 7, 123, 2026: $1" >&2; exit 2 ;;
    esac
}

config_for() { echo "configs/finetune_vgwd_clean_rslora_r32_allattn_seed${1}.yaml"; }
checkpoint_root_for() { echo "checkpoints/vgwd_clean_rslora_r32_allattn_seed${1}"; }

baseline_checkpoint_for() {
    local root="checkpoints/vgwd_clean_rslora_r32_seed${1}"
    require_path "${root}"
    "${PYTHON_CMD[@]}" -c 'import sys; from pathlib import Path; p=sorted(Path(sys.argv[1]).glob("**/checkpoint-3500")); assert len(p)==1, f"Expected one checkpoint-3500, found {len(p)}"; print(p[0])' "${root}"
}

allattn_checkpoint_for() {
    local root
    root="$(checkpoint_root_for "$1")"
    require_path "${root}"
    "${PYTHON_CMD[@]}" -c 'import json,sys; from pathlib import Path; p=sorted(Path(sys.argv[1]).glob("**/checkpoint-3500")); assert len(p)==1, f"Expected one checkpoint-3500, found {len(p)}"; c=json.load((p[0]/"adapter_config.json").open()); assert set(c["target_modules"])=={"q_proj","k_proj","v_proj","out_proj"}, c["target_modules"]; assert c["r"]==32 and c["lora_alpha"]==64, (c["r"],c["lora_alpha"]); print(p[0])' "${root}"
}

prediction_root_for() {
    local condition="$1" seed="$2"
    case "${condition}" in
        qv) echo "predictions/vgwd_clean_rslora_r32_seed${seed}_valid_beam10" ;;
        allattn) echo "predictions/vgwd_clean_rslora_r32_allattn_seed${seed}_valid_beam10" ;;
        *) echo "Unknown condition: ${condition}" >&2; exit 2 ;;
    esac
}

prediction_file_for() {
    local root
    root="$(prediction_root_for "$1" "$2")"
    require_path "${root}"
    "${PYTHON_CMD[@]}" -c 'import sys; from pathlib import Path; root=Path(sys.argv[1]); assert "test" not in str(root).lower(), root; p=[x for x in root.glob("**/predictions.jsonl") if sum(1 for _ in x.open())==int(sys.argv[2])]; assert len(p)==1, f"Expected one complete validation prediction, found {len(p)}"; print(p[0])' "${root}" "${EXPECTED_ROWS}"
}

prepare_configs() {
    "${PYTHON_CMD[@]}" analysis/generate_rslora_r32_all_attention_configs.py
}

validation_guard() {
    require_path "${VALID_CONFIG}"
    require_path "${VALID_LABELS}"
    "${PYTHON_CMD[@]}" -c 'import sys,yaml; from pathlib import Path; c=yaml.safe_load(Path(sys.argv[1]).read_text()); data=c["dataset"]; assert data["data_path"]=="data/vgwd_clean/valid.jsonl", data; assert data["data_split"]=="valid", data; assert "test" not in data["data_path"].lower(); labels=Path(sys.argv[2]); assert labels.as_posix()=="data/vgwd_clean/valid_filtered_mz500.jsonl"; assert sum(1 for _ in labels.open())==664; print("Validation-only guard passed")' "${VALID_CONFIG}" "${VALID_LABELS}"
}

static_smoke() {
    prepare_configs
    validation_guard
    require_path "${BASE_CHECKPOINT}"
    "${PYTHON_CMD[@]}" -c 'import sys,yaml,torch; sys.path.insert(0,"spectus"); from model.modeling_spectus import SpectusForConditionalGeneration; from peft import LoraConfig,get_peft_model; c=yaml.safe_load(open(sys.argv[1])); targets=c["lora_args"]["target_modules"]; assert targets==["q_proj","k_proj","v_proj","out_proj"],targets; m=SpectusForConditionalGeneration.from_pretrained(sys.argv[2]); counts={t:sum(n.endswith("."+t) for n,_ in m.named_modules()) for t in targets}; assert counts=={t:36 for t in targets},counts; a=c["lora_args"]; assert a["r"]==32 and a["lora_alpha"]==64; m=get_peft_model(m,LoraConfig(r=a["r"],lora_alpha=a["lora_alpha"],lora_dropout=a["lora_dropout"],target_modules=targets,bias=a["bias"],use_rslora=a["use_rslora"])); trainable=sum(p.numel() for p in m.parameters() if p.requires_grad); wrapped=sum("lora_A.default" in n for n,_ in m.named_modules()); assert wrapped==144,wrapped; m.to("cuda"); print({"module_counts":counts,"wrapped_projections":wrapped,"trainable_parameters":trainable,"cuda_allocated_mib":round(torch.cuda.memory_allocated()/1024**2,1)})' "$(config_for 7)" "${BASE_CHECKPOINT}"
}

train_seed() {
    local seed="$1" config root latest
    validate_seed "${seed}"
    config="$(config_for "${seed}")"
    root="$(checkpoint_root_for "${seed}")"
    require_path "${BASE_CHECKPOINT}"
    require_path "${config}"
    if [[ -d "${root}" ]] && allattn_checkpoint_for "${seed}" >/dev/null 2>&1; then
        echo "Reusing completed r32 all-attention checkpoint: seed=${seed}"
        return
    fi
    if [[ -d "${root}" ]]; then
        latest="$(find "${root}" -type d -name 'checkpoint-*' -printf '%f %p\n' | sort -t- -k2,2n | tail -1 | cut -d' ' -f2-)"
        [[ -n "${latest}" ]] || { echo "No resumable checkpoint below ${root}" >&2; exit 1; }
        echo "Resuming r32 all-attention seed=${seed} from ${latest}"
        "${PYTHON_CMD[@]}" spectus/train_spectus_lora.py \
            --config-file "${config}" --checkpoint "${BASE_CHECKPOINT}" --resume-checkpoint "${latest}" \
            --checkpoints-dir checkpoints --additional-info "_vgwd_clean_rslora_r32_allattn_seed${seed}" \
            --wandb-group "vgwd_clean_rslora_r32_allattn_seed${seed}"
    else
        echo "Training r32 all-attention seed=${seed}"
        "${PYTHON_CMD[@]}" spectus/train_spectus_lora.py \
            --config-file "${config}" --checkpoint "${BASE_CHECKPOINT}" \
            --checkpoints-dir checkpoints --additional-info "_vgwd_clean_rslora_r32_allattn_seed${seed}" \
            --wandb-group "vgwd_clean_rslora_r32_allattn_seed${seed}"
    fi
    allattn_checkpoint_for "${seed}" >/dev/null
}

predict_condition() {
    local condition="$1" seed="$2" checkpoint root backup
    validate_seed "${seed}"
    root="$(prediction_root_for "${condition}" "${seed}")"
    [[ "${root,,}" != *test* ]] || { echo "Validation-only guard rejected output: ${root}" >&2; exit 1; }
    if [[ -d "${root}" ]] && prediction_file_for "${condition}" "${seed}" >/dev/null 2>&1; then
        echo "Reusing completed validation prediction: ${condition}, seed=${seed}"
        return
    fi
    case "${condition}" in
        qv) checkpoint="$(baseline_checkpoint_for "${seed}")" ;;
        allattn) checkpoint="$(allattn_checkpoint_for "${seed}")" ;;
        *) echo "Unknown condition: ${condition}" >&2; exit 2 ;;
    esac
    if [[ -e "${root}" ]]; then
        backup="${root}.incomplete.$(date +%Y%m%d_%H%M%S)"
        mv "${root}" "${backup}"
        echo "Moved incomplete validation prediction to ${backup}"
    fi
    echo "Predicting validation only: ${condition}, seed=${seed}, checkpoint-3500, Beam 10"
    "${PYTHON_CMD[@]}" spectus/predict_lora.py \
        --checkpoint "${checkpoint}" --output-folder "${root}" --config-file "${VALID_CONFIG}"
    prediction_file_for "${condition}" "${seed}" >/dev/null
}

audit_condition() {
    local prediction
    prediction="$(prediction_file_for "$1" "$2")"
    "${PYTHON_CMD[@]}" analysis/audit_exact_topk.py \
        --predictions "${prediction}" --labels "${VALID_LABELS}" --max-k 10
}

full() {
    mkdir -p "${OUTPUT_DIR}"
    prepare_configs
    validation_guard
    for seed in "${SEEDS[@]}"; do
        train_seed "${seed}"
    done
    for seed in "${SEEDS[@]}"; do
        for condition in qv allattn; do
            predict_condition "${condition}" "${seed}"
            audit_condition "${condition}" "${seed}"
        done
    done
    "${PYTHON_CMD[@]}" analysis/summarize_rslora_r32_all_attention_validation.py --output-dir "${OUTPUT_DIR}"
}

case "${1:-}" in
    smoke) [[ $# -eq 1 ]] || exit 2; static_smoke ;;
    full) [[ $# -eq 1 ]] || exit 2; full ;;
    train) [[ $# -eq 2 ]] || exit 2; prepare_configs; validation_guard; train_seed "$2" ;;
    predict) [[ $# -eq 3 ]] || exit 2; validation_guard; predict_condition "$2" "$3"; audit_condition "$2" "$3" ;;
    summarize) [[ $# -eq 1 ]] || exit 2; validation_guard; "${PYTHON_CMD[@]}" analysis/summarize_rslora_r32_all_attention_validation.py --output-dir "${OUTPUT_DIR}" ;;
    *) echo "Usage: $0 smoke | full | train {7|123|2026} | predict {qv|allattn} {7|123|2026} | summarize" >&2; exit 2 ;;
esac
