# %% [markdown]
# # StreetMath Linear Probes — one‑cell, low‑RAM, robust
# 
# This single Colab cell trains **streaming linear probes** that read "near‑10" (within 1 of a multiple of 10)
# from model hidden states, with improvements:
# 
# - **Span pooling** over multi‑token numbers (digits or words)
# - **Two‑pass online standardization + SGD (logistic)** for stability
# - **Streaming** (no giant activation matrices)
# - **Template & surface‑form robustness checks** (digits vs words; paraphrases)
# - **Layer stride / subset** probing for speed
# - **Optional second model** (e.g., "thinking" vs "non‑thinking"), loaded **sequentially**
# 
# Safe defaults use **GPT‑2 (124M)** to avoid OOM. Swap `MODEL_ID_*` if you have more RAM.
# 
# ---
# 
# **Usage:** just run this cell. To compare two models, set `MODEL_ID_2` to another HF id.

# %%
# Quiet installs
#%pip -q install "transformers==4.44.2" "accelerate>=0.34.2" "scikit-learn==1.5.1" "num2words==0.5.13"

import os, re, gc, math, random, json, numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from num2words import num2words

# -----------------------
# Config (tune here)
# -----------------------
SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Small & safe defaults. Replace with your models (loaded sequentially).
MODEL_ID_1 = "Qwen/Qwen3-4B-Instruct-2507"                     # e.g., "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_ID_2 = None                        # e.g., "Qwen/Qwen3-4B-Instruct-2507" (only if your Colab has RAM/VRAM)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

# Data sizes
TRAIN_N = 4000
VAL_N   = 1500

# Efficiency
BATCH_SIZE  = 16     # reduce if OOM
MAX_LEN     = 96     # reduce if OOM
LAYER_STRIDE = 1     # probe every k-th layer (1=all)

# Probing target: near‑10 = distance to nearest multiple of 10 <= 1 (covers last digits {0,1,9})
NEAR_THRESHOLD = 1

# -----------------------
# Data generation
# -----------------------
TEMPLATES_A = [
    "Consider the number {n}.",
    "Let x = {n}.",
    "Value: {n}",
    "n: {n}",
    "The integer {n} appears below.",
]
TEMPLATES_B = [
    "Here is {n}.",
    "We study the scalar {n}.",
    "Write down {n} and continue.",
    "Take {n} as the value.",
]

def dist_to_nearest_10(n: int) -> int:
    last = n % 10
    return min(abs(last-0), abs(last-10))

def rounding_direction(n: int) -> int:
    # -1 if closer by rounding down to a multiple of 10, +1 up, 0 exact
    d0 = abs((n//10)*10 - n)
    d1 = abs(((n+9)//10)*10 - n)  # nearest up multiple (ceil to multiple of 10)
    if d0 == 0 or d1 == 0:
        return 0
    return +1 if d1 < d0 else -1


def make_dataset(N: int, templates: List[str], surface: str = "digits", lo: int = 0, hi: int = 9999, seed: int = 1337):
    rng = random.Random(seed)
    texts, labels, metas, spans = [], [], [], []
    for _ in range(N):
        n = rng.randint(lo, hi)
        d = dist_to_nearest_10(n)
        y = 1 if d <= NEAR_THRESHOLD else 0
        rd = rounding_direction(n)
        if surface == "digits":
            n_str = str(n)
        elif surface == "words":
            # e.g., "one thousand, forty-six" → normalize to lower, no commas for robust matching
            n_str = num2words(n).replace('-', ' ').replace(',', '').lower()
        else:
            raise ValueError("surface must be 'digits' or 'words'")
        tmpl = rng.choice(templates)
        text = tmpl.format(n=n_str)
        texts.append(text)
        labels.append(y)
        metas.append((d, rd, n))  # (distance, direction, raw n)
    return texts, np.array(labels, dtype=np.int64), metas

# Train on digits + Template A; validate on digits (B) and words (A,B) to detect shortcuts
train_texts, y_train, meta_train = make_dataset(TRAIN_N, TEMPLATES_A, surface="digits", seed=SEED)
valA_texts,  y_valA,  meta_valA  = make_dataset(VAL_N,   TEMPLATES_B, surface="digits", seed=SEED+1)
valW_texts,  y_valW,  meta_valW  = make_dataset(VAL_N,   TEMPLATES_A, surface="words",  seed=SEED+2)

# -----------------------
# Token span finder (multi‑token aware)
# -----------------------

def find_last_subseq(hay: List[int], needle: List[int]) -> Optional[Tuple[int,int]]:
    if not needle: return None
    found = None
    for i in range(0, len(hay)-len(needle)+1):
        if hay[i:i+len(needle)] == needle:
            found = (i, i+len(needle))
    return found


def pool_number_span(tok, input_ids_row: List[int], layer_h: torch.Tensor, n_str: str) -> np.ndarray:
    """Mean‑pool features over the token span of `n_str`. Fallback: last non‑pad token.
    - input_ids_row: list of token ids for one example
    - layer_h: Tensor [T, D] for that example at a specific layer
    """
    n_ids = tok(n_str, add_special_tokens=False)["input_ids"]
    span = find_last_subseq(input_ids_row, n_ids)
    if span is None:
        # fallback: last non‑pad position
        T = layer_h.shape[0]
        return layer_h[T-1].detach().cpu().numpy()
    s,e = span
    s = min(s, layer_h.shape[0]-1); e = min(e, layer_h.shape[0])
    vec = layer_h[s:e].mean(dim=0)
    return vec.detach().cpu().numpy()

# -----------------------
# Streaming probe pipeline (two‑pass: scalers then classifiers)
# -----------------------

def iter_batches(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i:i+bs], slice(i, min(i+bs, len(lst)))


def model_layer_meta(model) -> Tuple[int,int]:
    # Tiny forward to read (n_layers, d_model)
    with torch.no_grad():
        tmp_tok = AutoTokenizer.from_pretrained(model.name_or_path)
        if tmp_tok.pad_token_id is None: tmp_tok.pad_token_id = tmp_tok.eos_token_id
        tmp = tmp_tok("n: 1", return_tensors="pt")
        out = model(**{k:v.to(DEVICE) for k,v in tmp.items()}, use_cache=False, output_hidden_states=True)
        L = len(out.hidden_states)  # includes embeddings
        D = out.hidden_states[-1].shape[-1]
        del tmp, out
        if DEVICE == "cuda": torch.cuda.empty_cache()
        gc.collect()
        return L, D


def stream_hidden_features(model, tok, texts: List[str], metas: List[Tuple[int,int,int]], layer_ids: List[int], batch_size: int, max_len: int, surface: str) -> Tuple[Dict[int, np.ndarray], np.ndarray, List[Tuple[int,int,int]]]:
    """Yield features per layer as we stream. Here we *return* nothing (streaming),
    but this helper centralizes the per‑batch extraction.
    """
    for chunk, s in iter_batches(texts, batch_size):
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        input_ids = enc["input_ids"].to(DEVICE)
        attn      = enc["attention_mask"].to(DEVICE)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False, output_hidden_states=True)
            hs = out.hidden_states  # tuple length L
            ids_list = enc["input_ids"].tolist()
            # for each layer, build X_batch
            Xb_per_layer = {L: [] for L in layer_ids}
            # Find the numeric string we inserted for each example
            # We parse it back using regex from the *original* chunk text.
            for b, text in enumerate(chunk):
                # Extract the n_str we inserted (digits or words). It's the last number/word number in text.
                if surface == "digits":
                    m = re.findall(r"-?\d+", text)
                    n_str = m[-1] if m else "0"
                else:
                    # for words, re‑compose the substring between placeholder and punctuation
                    # heuristic: grab longest group of letter/space around where templates put {n}
                    # fallback: take full text (then span match just finds nothing and we fallback to last token)
                    mm = re.findall(r"[a-z\s]+", text.lower())
                    n_str = max(mm, key=len).strip() if mm else text.lower()
                row_ids = ids_list[b]
                for L in layer_ids:
                    vec = pool_number_span(tok, row_ids, hs[L][b], n_str)
                    Xb_per_layer[L].append(vec)
            # Stack per layer
            for L in layer_ids:
                Xb_per_layer[L] = np.stack(Xb_per_layer[L], axis=0)
        # Labels & meta slice
        yb = None  # returned by caller via closure; we don't return here to keep memory low
        yield Xb_per_layer, s
        del out, hs, enc, input_ids, attn, Xb_per_layer
        if DEVICE == "cuda": torch.cuda.empty_cache()
        gc.collect()


def run_probes_for_model(model_id: str, tag: str, train_pack, eval_packs):
    print(f"\n========== {tag}: {model_id} ==========")
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f'Trying slow tokenizer due to error: {e}')
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE=="cuda" else {"": "cpu"},
        output_hidden_states=True,
        trust_remote_code=True,
    )
    model.eval(); torch.set_grad_enabled(False)

    total_layers, width = model_layer_meta(model)
    layer_ids = list(range(1, total_layers, LAYER_STRIDE))  # skip embeddings (0)
    print(f"Layers (incl. emb): {total_layers} | width: {width} | probing layers: {layer_ids[:8]}{'...' if len(layer_ids)>8 else ''}")

    # ---- First pass: fit StandardScalers per layer (train set only) ----
    scalers = {L: StandardScaler(with_mean=False) for L in layer_ids}
    texts_tr, y_tr, meta_tr, surf_tr = train_pack
    for Xb_per_layer, s in stream_hidden_features(model, tok, texts_tr, meta_tr, layer_ids, BATCH_SIZE, MAX_LEN, surf_tr):
        for L in layer_ids:
            scalers[L].partial_fit(Xb_per_layer[L])

    # ---- Second pass: fit SGD (logistic) per layer (train set only) ----
    probes = {L: SGDClassifier(loss="log_loss", learning_rate="optimal", alpha=1e-4, tol=None, max_iter=1) for L in layer_ids}
    inited = set()
    for Xb_per_layer, s in stream_hidden_features(model, tok, texts_tr, meta_tr, layer_ids, BATCH_SIZE, MAX_LEN, surf_tr):
        yb = y_tr[s]
        for L in layer_ids:
            Xb = scalers[L].transform(Xb_per_layer[L])
            if L not in inited:
                probes[L].partial_fit(Xb, yb, classes=np.array([0,1], dtype=np.int64))
                inited.add(L)
            else:
                probes[L].partial_fit(Xb, yb)

    # ---- Evaluations on multiple packs ----
    results = {}
    for name, (texts_ev, y_ev, meta_ev, surf_ev) in eval_packs.items():
        acc = {L: [] for L in layer_ids}
        # Also aggregate per‑bucket errors & probabilities for the *best* layer later
        store_logits = {L: {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}}  # distance buckets for near-10
        # Stream eval
        for Xb_per_layer, s in stream_hidden_features(model, tok, texts_ev, meta_ev, layer_ids, BATCH_SIZE, MAX_LEN, surf_ev):
            yb = y_ev[s]
            for L in layer_ids:
                Xb = scalers[L].transform(Xb_per_layer[L])
                pb = probes[L].predict(Xb)
                acc[L].extend((pb == yb).tolist())
        # Summarize
        acc_mean = {L: float(np.mean(acc[L])) if len(acc[L]) else 0.0 for L in layer_ids}
        bestL = max(acc_mean, key=acc_mean.get)
        print(f"Eval[{name}] best layer {bestL} acc={acc_mean[bestL]:.3f}")

        # Error breakdown for the best layer
        # For near-10, distances can be 0, 1, 2, 3, 4, or 5
        errs_by_dist = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}
        errs_by_dir  = {-1:[], 0:[], +1:[]}
        # Re‑stream only best layer to compute buckets + proba
        for Xb_per_layer, s in stream_hidden_features(model, tok, texts_ev, meta_ev, [bestL], BATCH_SIZE, MAX_LEN, surf_ev):
            Xb = scalers[bestL].transform(Xb_per_layer[bestL])
            yb = y_ev[s]
            proba = probes[bestL].predict_proba(Xb)[:,1]
            preds = (proba >= 0.5).astype(np.int64)
            for i, idx in enumerate(range(s.start, s.stop)):
                d, rd, n = meta_ev[idx]
                d_bucket = d  # For near-10, keep all distances 0-5 separate
                errs_by_dist[d_bucket].append(int(preds[i] != yb[i]))
                errs_by_dir[rd].append(int(preds[i] != yb[i]))
        dist_view = {k: (float(np.mean(v)) if len(v) else None, len(v)) for k,v in errs_by_dist.items()}
        dir_view  = {k: (float(np.mean(v)) if len(v) else None, len(v)) for k,v in errs_by_dir.items()}
        results[name] = {"acc_per_layer": acc_mean, "best_layer": bestL, "err_by_dist": dist_view, "err_by_dir": dir_view}

    # Clean up model to free VRAM
    del model, tok
    if DEVICE=="cuda": torch.cuda.empty_cache()
    gc.collect()
    return results

# -----------------------
# Run: model 1 (and optional model 2)
# -----------------------

train_pack = (train_texts, y_train, meta_train, "digits")
eval_packs = {
    "digits_paraphrase": (valA_texts, y_valA, meta_valA, "digits"),
    "words":             (valW_texts, y_valW, meta_valW, "words"),
}

all_results = {}
all_results["model1"] = run_probes_for_model(MODEL_ID_1, tag="MODEL_1", train_pack=train_pack, eval_packs=eval_packs)

if MODEL_ID_2:
    all_results["model2"] = run_probes_for_model(MODEL_ID_2, tag="MODEL_2", train_pack=train_pack, eval_packs=eval_packs)

# Pretty print a compact summary
print("\n================ SUMMARY ================")
for mid, res in all_results.items():
    print(f"\n--- {mid} ---")
    for split, info in res.items():
        Lbest = info["best_layer"]
        acc = info["acc_per_layer"][Lbest]
        dist = info["err_by_dist"]; dire = info["err_by_dir"]
        dist_str = " | ".join([f"d={k}: err={v[0]:.3f} (n={v[1]})" for k,v in sorted(dist.items())])
        dir_str  = " | ".join([f"dir={k:+d}: err={v[0]:.3f} (n={v[1]})" for k,v in sorted(dire.items())])
        print(f"{split:18s} -> best layer {Lbest:2d} acc={acc:.3f}\n  by distance: {dist_str}\n  by direction: {dir_str}")

# Tip: to save JSON for later analysis, uncomment:
with open("streetmath_probe_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
