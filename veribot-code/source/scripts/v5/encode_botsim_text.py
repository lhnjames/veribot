"""Encode BotSim-24's description_list.pt (raw text) into text_tensor.pt using the same RoBERTa-base
encoder as TB22's des/tweet embeddings, for internal consistency across datasets."""
import torch
from transformers import AutoTokenizer, AutoModel

ENCODER = "/path/to/data_root/cache/modelscope/models/AI-ModelScope--roberta-base/snapshots/master"
OUT = "/path/to/data_root/data/botsim_split"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

texts = torch.load(f"{OUT}/description_list.pt", weights_only=False)
texts = [str(t) if t else "" for t in texts]
tok = AutoTokenizer.from_pretrained(ENCODER)
model = AutoModel.from_pretrained(ENCODER).to(DEV).eval()

def encode(texts, bs=256, max_len=64):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(DEV)
        with torch.no_grad():
            h = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            vec = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        out.append(vec.cpu())
        print(f"  encoded {min(i+bs,len(texts))}/{len(texts)}", flush=True)
    return torch.cat(out).float()

t = encode(texts)
torch.save(t, f"{OUT}/text_tensor.pt")
print(f"[botsim] text_tensor.pt saved, shape={t.shape}")
