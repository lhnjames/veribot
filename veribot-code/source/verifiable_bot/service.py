"""Inference service (design spec section 7).

Single forward pass -> (E, y_hat, C, s). A program checks reference legality (format_check).
Optionally (RECHECK_NECESSITY=1), a single-call (K=1) necessity recheck against the judge model
downgrades evidence_status to "insufficient" if removing the cited evidence does not actually
weaken support for the prediction -- the model is never asked to fabricate evidence to compensate.
No external LLM is required at inference time unless this recheck is enabled.
"""
from __future__ import annotations

import os

import torch
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .format_check import check_format
from .llm_client import DeepSeekJudge
from .prompting import inference_messages
from .schema import AccountRecord
from .units import build_units, hidden_unit_metadata, render_masked, render_units, unit_map

MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/grpo/final")
BASE_MODEL = os.environ.get("BASE_MODEL", "/path/to/data_root/cache/huggingface/hub/"
                                              "models--Qwen--Qwen3-0.6B/snapshots/"
                                              "c1899de289a04d12100db370d81485cdf75e47ca")
RECHECK_NECESSITY = os.environ.get("RECHECK_NECESSITY", "0") == "1"
NECESSITY_RECHECK_THRESHOLD = float(os.environ.get("NECESSITY_RECHECK_THRESHOLD", "0.05"))

app = FastAPI(title="VeriRoute-Lite Bot Detector", version="3.0")
tokenizer = model = judge = None


@app.on_event("startup")
def load_model() -> None:
    global tokenizer, model, judge
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True
    )
    try:
        model = PeftModel.from_pretrained(base, MODEL_PATH).eval()
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True
        ).eval()
    if RECHECK_NECESSITY:
        # k=1 deliberately: this is an online, single-request, low-latency inference-time
        # sanity check, NOT part of training or any sufficiency/necessity screening metric --
        # it must never be conflated with or averaged into the K=5 SFT-screening / RL-reward
        # judge statistics, which all use the DeepSeekJudge class default (see llm_client.py).
        judge = DeepSeekJudge(cache_dir="outputs/judge_cache_service", k=1)


@app.get("/health")
def health() -> dict:
    return {"ok": model is not None, "model": MODEL_PATH, "recheck_necessity": RECHECK_NECESSITY}


@app.post("/predict")
def predict(record: AccountRecord) -> dict:
    units = build_units(record)
    valid_ids = unit_map(units)
    prompt = tokenizer.apply_chat_template(inference_messages(record), tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=640, do_sample=False)
    raw = tokenizer.decode(output_ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

    fc = check_format(raw, valid_ids, hidden_unit_metadata(units))
    if not fc.ok:
        raise HTTPException(422, f"Model produced an invalid or ungrounded output: {fc.reason}")

    result = fc.output.model_dump()
    prediction = (record.upstream_verdict if fc.output.agreement == "agree"
                  else ("human" if record.upstream_verdict == "bot" else "bot"))
    if RECHECK_NECESSITY and judge is not None and result["evidence_status"] == "sufficient":
        full_text = render_units(units)
        masked_text = render_masked(units, set(fc.output.evidence_ids))
        approve_rate_full = judge.label_approve_rate(full_text, prediction)
        approve_rate_masked = judge.label_approve_rate(masked_text, prediction)
        if (approve_rate_full - approve_rate_masked) < NECESSITY_RECHECK_THRESHOLD:
            result["evidence_status"] = "insufficient"

    return {"prediction": prediction, "analysis": result}
