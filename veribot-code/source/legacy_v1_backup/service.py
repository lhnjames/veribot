from __future__ import annotations

import json
import os

import torch
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompting import inference_messages
from .schema import AccountRecord, EvidencePrediction
from .verifier import EvidenceVerifier

MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/sft/final")
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-0.6B")
app = FastAPI(title="Verifiable Bot Detector", version="1.0")
tokenizer = model = None


@app.on_event("startup")
def load_model():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, MODEL_PATH).eval()


@app.get("/health")
def health(): return {"ok": model is not None, "model": MODEL_PATH}


@app.post("/predict")
def predict(record: AccountRecord):
    prompt = tokenizer.apply_chat_template(inference_messages(record), tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad(): output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    raw = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    try: prediction = EvidencePrediction.model_validate_json(raw)
    except Exception as exc: raise HTTPException(422, f"Invalid model JSON: {exc}")
    verification = EvidenceVerifier().verify(record, prediction)
    return {"prediction": prediction, "verification": verification}
