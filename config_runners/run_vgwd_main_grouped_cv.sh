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
DATA_ROOT="data/vgwd_main_grouped_cv_locked"
SEED=123

require_path() {
    [[ -e "$1" ]] || { echo "Required path not found: $1" >&2; exit 1; }
}

validate_fold() {
    case "$1" in
        0|1|2|3|4) ;;
        *) echo "Fold must be 0, 1, 2, 3, or 4: $1" >&2; exit 2 ;;
    esac
}

validate_condition() {
    case "$1" in
        qv|allattn) ;;
        *) echo "Condition must be qv or allattn: $1" >&2; exit 2 ;;
    esac
}

verify_lock() {
    "${PYTHON_CMD[@]}" analysis/main_model_grouped_cv/prepare_locked_split.py --verify
    require_path "${BASE_CHECKPOINT}"
}

generate_configs() {
    "${PYTHON_CMD[@]}" analysis/main_model_grouped_cv/generate_configs.py
}

config_for() {
    echo "configs/main_grouped_cv/train_fold${1}_${2}_seed${SEED}.yaml"
}

predict_config_for() {
    echo "configs/main_grouped_cv/predict_fold${1}_${2}_seed${SEED}_beam10.yaml"
}

checkpoint_root_for() {
    echo "checkpoints/vgwd_main_grouped_cv_fold${1}_${2}_seed${SEED}"
}

checkpoint_for() {
    local root
    root="$(checkpoint_root_for "$1" "$2")"
    require_path "${root}"
    "${PYTHON_CMD[@]}" -c 'import sys; from pathlib import Path; p=list(Path(sys.argv[1]).glob("**/checkpoint-3500")); assert len(p)==1, f"Expected one checkpoint-3500, found {len(p)}"; print(p[0])' "${root}"
}

guard_config() {
    local config="$1" fold="$2" condition="$3"
    require_path "${config}"
    "${PYTHON_CMD[@]}" -c 'import sys,yaml; from pathlib import Path; c=yaml.safe_load(Path(sys.argv[1]).read_text()); fold=int(sys.argv[2]); condition=sys.argv[3]; d=c["data_args"]["datasets"]["VGWD"]; expected=f"data/vgwd_main_grouped_cv_locked/fold_{fold}/train.jsonl"; assert d["train_path"]==expected,(d["train_path"],expected); assert "holdout" not in str(c).lower(); targets=c["lora_args"]["target_modules"]; wanted=["q_proj","v_proj"] if condition=="qv" else ["q_proj","k_proj","v_proj","out_proj"]; assert targets==wanted,(targets,wanted); a=c["hf_training_args"]; assert a["max_steps"]==3500 and a["seed"]==123; assert not a["do_eval"] and not a["load_best_model_at_end"]; print("Training guard passed", fold, condition)' "${config}" "${fold}" "${condition}"
}

train_one() {
    local fold="$1" condition="$2" config root latest group
    validate_fold "${fold}"
    validate_condition "${condition}"
    config="$(config_for "${fold}" "${condition}")"
    root="$(checkpoint_root_for "${fold}" "${condition}")"
    group="vgwd_main_grouped_cv_fold${fold}_${condition}_seed${SEED}"
    guard_config "${config}" "${fold}" "${condition}"
    if [[ -d "${root}" ]] && checkpoint_for "${fold}" "${condition}" >/dev/null 2>&1; then
        echo "Reusing completed checkpoint: fold=${fold}, condition=${condition}"
        return
    fi
    if [[ -d "${root}" ]]; then
        latest="$(find "${root}" -type d -name 'checkpoint-*' -printf '%f %p\n' | sort -t- -k2,2n | tail -1 | cut -d' ' -f2-)"
        [[ -n "${latest}" ]] || { echo "No resumable checkpoint below ${root}" >&2; exit 1; }
        echo "Resuming fold=${fold}, condition=${condition} from ${latest}"
        "${PYTHON_CMD[@]}" spectus/train_spectus_lora.py \
            --config-file "${config}" --checkpoint "${BASE_CHECKPOINT}" \
            --resume-checkpoint "${latest}" --checkpoints-dir checkpoints \
            --additional-info "_main_grouped_cv_fold${fold}_${condition}_seed${SEED}" \
            --wandb-group "${group}"
    else
        echo "Training fold=${fold}, condition=${condition}"
        "${PYTHON_CMD[@]}" spectus/train_spectus_lora.py \
            --config-file "${config}" --checkpoint "${BASE_CHECKPOINT}" \
            --checkpoints-dir checkpoints \
            --additional-info "_main_grouped_cv_fold${fold}_${condition}_seed${SEED}" \
            --wandb-group "${group}"
    fi
    checkpoint_for "${fold}" "${condition}" >/dev/null
}

prediction_root_for() {
    echo "predictions/vgwd_main_grouped_cv/fold${1}/${2}"
}

prediction_file_for() {
    local root expected
    root="$(prediction_root_for "$1" "$2")"
    expected="$(wc -l < "${DATA_ROOT}/fold_${1}/oof.jsonl")"
    require_path "${root}"
    "${PYTHON_CMD[@]}" -c 'import sys; from pathlib import Path; root=Path(sys.argv[1]); expected=int(sys.argv[2]); p=[x for x in root.glob("**/predictions.jsonl") if sum(1 for _ in x.open())==expected]; assert len(p)==1,f"Expected one complete OOF prediction, found {len(p)}"; print(p[0])' "${root}" "${expected}"
}

predict_one() {
    local fold="$1" condition="$2" config checkpoint root backup labels
    validate_fold "${fold}"
    validate_condition "${condition}"
    config="$(predict_config_for "${fold}" "${condition}")"
    checkpoint="$(checkpoint_for "${fold}" "${condition}")"
    root="$(prediction_root_for "${fold}" "${condition}")"
    labels="${DATA_ROOT}/fold_${fold}/oof.jsonl"
    require_path "${config}"
    require_path "${labels}"
    [[ "${config,,}" != *holdout* && "${labels,,}" != *holdout* && "${root,,}" != *holdout* ]] || {
        echo "Locked holdout guard rejected prediction request" >&2
        exit 1
    }
    if [[ -d "${root}" ]] && prediction_file_for "${fold}" "${condition}" >/dev/null 2>&1; then
        echo "Reusing completed OOF prediction: fold=${fold}, condition=${condition}"
        return
    fi
    if [[ -e "${root}" ]]; then
        backup="${root}.incomplete.$(date +%Y%m%d_%H%M%S)"
        mv "${root}" "${backup}"
        echo "Moved incomplete OOF prediction to ${backup}"
    fi
    "${PYTHON_CMD[@]}" spectus/predict_lora.py \
        --checkpoint "${checkpoint}" --output-folder "${root}" \
        --config-file "${config}"
    prediction="$(prediction_file_for "${fold}" "${condition}")"
    "${PYTHON_CMD[@]}" analysis/audit_exact_topk.py \
        --predictions "${prediction}" --labels "${labels}" --max-k 10
}

static_smoke() {
    verify_lock
    generate_configs
    for condition in qv allattn; do
        guard_config "$(config_for 0 "${condition}")" 0 "${condition}"
    done
    "${PYTHON_CMD[@]}" -c 'import torch,yaml,sys; sys.path.insert(0,"spectus"); from peft import LoraConfig,get_peft_model; from model.modeling_spectus import SpectusForConditionalGeneration; base=sys.argv[1]; configs=sys.argv[2:]; m=SpectusForConditionalGeneration.from_pretrained(base); found={t:sum(n.endswith("."+t) for n,_ in m.named_modules()) for t in ("q_proj","k_proj","v_proj","out_proj")}; assert found=={t:36 for t in found},found; results={};
for path in configs:
 c=yaml.safe_load(open(path)); a=c["lora_args"]; wrapped=get_peft_model(SpectusForConditionalGeneration.from_pretrained(base),LoraConfig(r=a["r"],lora_alpha=a["lora_alpha"],lora_dropout=a["lora_dropout"],target_modules=a["target_modules"],bias=a["bias"],use_rslora=a["use_rslora"])); results[path]={"targets":a["target_modules"],"trainable":sum(p.numel() for p in wrapped.parameters() if p.requires_grad)}
print({"static_smoke":"PASS","module_counts":found,"models":results,"cuda_available":torch.cuda.is_available()})' "${BASE_CHECKPOINT}" "$(config_for 0 qv)" "$(config_for 0 allattn)"
}

pipeline_smoke() {
    local condition config group output_root checkpoint prediction_config
    verify_lock
    generate_configs
    for condition in qv allattn; do
        config="configs/main_grouped_cv/smoke_fold0_${condition}_seed${SEED}.yaml"
        group="vgwd_main_grouped_cv_smoke_fold0_${condition}_seed${SEED}"
        output_root="checkpoints/${group}"
        if [[ -e "${output_root}" ]]; then
            echo "Reusing existing smoke output: ${output_root}"
            continue
        fi
        "${PYTHON_CMD[@]}" spectus/train_spectus_lora.py \
            --config-file "${config}" --checkpoint "${BASE_CHECKPOINT}" \
            --checkpoints-dir checkpoints \
            --additional-info "_main_grouped_cv_smoke_fold0_${condition}_seed${SEED}" \
            --wandb-group "${group}"
        checkpoint="$(find "${output_root}" -type d -name checkpoint-2 | head -1)"
        require_path "${checkpoint}"
        prediction_config="$(predict_config_for 0 "${condition}")"
        "${PYTHON_CMD[@]}" spectus/predict_lora.py \
            --checkpoint "${checkpoint}" \
            --output-folder "predictions/vgwd_main_grouped_cv_smoke/fold0/${condition}" \
            --config-file "${prediction_config}" --data-range 0:2
    done
    echo "End-to-end two-step smoke test passed for qv and allattn"
}

run_fold() {
    local fold="$1" condition
    verify_lock
    generate_configs
    for condition in qv allattn; do
        train_one "${fold}" "${condition}"
        predict_one "${fold}" "${condition}"
    done
}

case "${1:-}" in
    verify) [[ $# -eq 1 ]] || exit 2; verify_lock; generate_configs ;;
    static-smoke) [[ $# -eq 1 ]] || exit 2; static_smoke ;;
    pipeline-smoke) [[ $# -eq 1 ]] || exit 2; pipeline_smoke ;;
    train) [[ $# -eq 3 ]] || exit 2; verify_lock; generate_configs; train_one "$2" "$3" ;;
    predict) [[ $# -eq 3 ]] || exit 2; verify_lock; generate_configs; predict_one "$2" "$3" ;;
    fold) [[ $# -eq 2 ]] || exit 2; validate_fold "$2"; run_fold "$2" ;;
    full) [[ $# -eq 1 ]] || exit 2; for fold in 0 1 2 3 4; do run_fold "${fold}"; done ;;
    summarize) [[ $# -eq 1 ]] || exit 2; verify_lock; "${PYTHON_CMD[@]}" analysis/main_model_grouped_cv/summarize_oof.py --allow-partial ;;
    *) echo "Usage: $0 verify | static-smoke | pipeline-smoke | train FOLD {qv|allattn} | predict FOLD {qv|allattn} | fold FOLD | full | summarize" >&2; exit 2 ;;
esac
