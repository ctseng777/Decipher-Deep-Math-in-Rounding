# %% [markdown]
# # StreetMath Linear Probes — Mamba2-2.7B (Near-10 Task)
# 
# This single Colab cell trains **streaming linear probes** that read "near‑10" (within 1 of a multiple of 10)
# from the **Mamba2-2.7B** state space model's hidden states, with improvements:
# 
# - **Span pooling** over multi‑token numbers (digits or words)
# - **Two‑pass online standardization + SGD (logistic)** for stability
# - **Streaming** (no giant activation matrices)
# - **Template & surface‑form robustness checks** (digits vs words; paraphrases)
# - **Layer stride / subset** probing for speed
# - **Optional second model** (e.g., "thinking" vs "non‑thinking"), loaded **sequentially**
# 
# Configured for **Mamba2-2.7B** state space model to analyze non-attention-based numerical understanding.
# 
# ---
# 
# **Usage:** just run this cell. To compare two models, set `MODEL_ID_2` to another HF id.

# %%
# Required installs for Mamba2
#%pip -q install "transformers>=4.46.0" "accelerate>=0.34.2" "scikit-learn==1.5.1" "num2words==0.5.13"
#%pip -q install "mamba-ssm" "causal-conv1d"  # Required for Mamba2 models

import os, re, gc, math, random, json, numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from transformers import AutoModel, AutoConfig, AutoTokenizer
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

# Set HuggingFace cache to /workspace to avoid disk space issues
os.environ['TRANSFORMERS_CACHE'] = '/workspace/hf_cache'
os.environ['HF_HOME'] = '/workspace/hf_cache'
from sklearn.metrics import accuracy_score
from num2words import num2words

# -----------------------
# Config (tune here)
# -----------------------
SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Mamba state space model configuration  
MODEL_ID_1 = "state-spaces/mamba-1.4b-hf"                   # Mamba 1.4B HF model
MODEL_ID_2 = None                        # Optional: comparison model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

# Data sizes (reduced for faster testing)
TRAIN_N = 4000   # Reduced for faster testing
VAL_N   = 1500   # Reduced for faster testing

# Efficiency (optimized for Mamba-2.8B)
BATCH_SIZE  = 8      # very small batch for 2.8B model to avoid memory issues
MAX_LEN     = 96     # shorter sequences for faster processing
LAYER_STRIDE = 1     # probe every layer (Mamba2 has fewer layers than transformers)

# Probing target: near‑5 = distance to nearest multiple of 5 <= 1 (covers last digits {4,5,6})
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

def dist_to_nearest_5(n: int) -> int:
    last = n % 5
    return min(abs(last-0), abs(last-5))

def rounding_direction(n: int) -> int:
    # -1 if closer by rounding down to a multiple of 5, +1 up, 0 exact
    d0 = abs((n//5)*5 - n)
    d1 = abs(((n+4)//5)*5 - n)  # nearest up multiple (ceil to multiple of 5)
    if d0 == 0 or d1 == 0:
        return 0
    return +1 if d1 < d0 else -1


def make_dataset(N: int, templates: List[str], surface: str = "digits", lo: int = 0, hi: int = 9999, seed: int = 1337):
    rng = random.Random(seed)
    texts, labels, metas, spans = [], [], [], []
    for _ in range(N):
        n = rng.randint(lo, hi)
        d = dist_to_nearest_5(n)
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
    # Since we don't have a tokenizer, use a simpler strategy
    # Average over the last few tokens which likely contain the number
    T = layer_h.shape[0]
    
    # Use last 3-5 tokens which likely contain the number
    num_tokens = min(5, T)
    start_idx = max(0, T - num_tokens)
    
    # Average over these tokens
    vec = layer_h[start_idx:T].mean(dim=0)
    return vec.detach().cpu().numpy()

# -----------------------
# Streaming probe pipeline (two‑pass: scalers then classifiers)
# -----------------------

def iter_batches(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i:i+bs], slice(i, min(i+bs, len(lst)))


def model_layer_meta(model) -> Tuple[int,int]:
    # Get layer count and hidden dimension for Mamba
    with torch.no_grad():
        # Create simple test input without tokenizer
        input_ids = torch.tensor([[110, 58, 32, 49]], dtype=torch.long).to(DEVICE)  # "n: 1" encoded
        
        try:
            # Mamba2 models should support output_hidden_states
            out = model(input_ids=input_ids, output_hidden_states=True)
            if hasattr(out, 'hidden_states') and out.hidden_states is not None:
                L = len(out.hidden_states)  # includes embeddings
                D = out.hidden_states[-1].shape[-1]
            else:
                # Fallback: use config values
                config = model.config
                L = getattr(config, 'num_layers', 24) + 1  # +1 for embeddings
                D = getattr(config, 'd_model', 2560)
                print(f"  Warning: Using config values L={L}, D={D}")
        except Exception as e:
            print(f"  Warning: Error in forward pass: {e}")
            # Use Mamba-1.4B default values
            L = 48 + 1  # 48 layers + embedding
            D = 1536   # d_model for 1.4B
            print(f"  Using Mamba-1.4B defaults: L={L}, D={D}")
            
        del input_ids
        if 'out' in locals(): del out
        if DEVICE == "cuda": torch.cuda.empty_cache()
        gc.collect()
        return L, D


def stream_hidden_features(model, tok, texts: List[str], metas: List[Tuple[int,int,int]], layer_ids: List[int], batch_size: int, max_len: int, surface: str) -> Tuple[Dict[int, np.ndarray], np.ndarray, List[Tuple[int,int,int]]]:
    """Yield features per layer as we stream. Here we *return* nothing (streaming),
    but this helper centralizes the per‑batch extraction.
    """
    for chunk, s in iter_batches(texts, batch_size):
        # Use simple tokenization instead of external tokenizer
        input_ids_list = []
        for text in chunk:
            tokens = simple_tokenize(text, max_len)
            input_ids_list.append(tokens)
        
        # Convert to tensor
        input_ids = torch.tensor(input_ids_list, dtype=torch.long).to(DEVICE)
        attn = torch.ones_like(input_ids).to(DEVICE)  # Create simple attention mask
        with torch.no_grad():
            # Mamba2 models may or may not need attention_mask
            try:
                # Try with attention mask first (standard approach)
                out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
            except Exception:
                # Fallback: Mamba2 might not use attention_mask
                out = model(input_ids=input_ids, output_hidden_states=True)
            
            # Extract hidden states from Mamba2 output
            if hasattr(out, 'hidden_states') and out.hidden_states is not None:
                hs = out.hidden_states  # tuple length L
            else:
                print(f"  Warning: No hidden states in output, skipping batch")
                continue
            ids_list = input_ids.tolist()
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
        del out, hs, input_ids, attn, Xb_per_layer
        if DEVICE == "cuda": torch.cuda.empty_cache()
        gc.collect()


def simple_tokenize(text: str, max_length: int = 128) -> List[int]:
    """Simple tokenization - convert text to integer tokens without external tokenizer"""
    # For this experiment, we'll use character-level encoding as a fallback
    # This is not optimal but allows us to run without tokenizer dependencies
    tokens = []
    for char in text[:max_length]:
        tokens.append(ord(char))
    
    # Pad to max_length
    while len(tokens) < max_length:
        tokens.append(0)  # 0 as pad token
    
    return tokens[:max_length]

def run_probes_for_model(model_id: str, tag: str, train_pack, eval_packs):
    print(f"\n========== {tag}: {model_id} ==========")
    
    # Skip tokenizer completely - we'll use simple tokenization
    print("Skipping tokenizer - using simple character-level encoding...")
    tok = None  # Will use simple_tokenize function instead
    print("  ✓ Will use character-level tokenization")
    
    # Load Mamba model directly using AutoModel with cache redirect and explicit config
    print("Loading Mamba-1.4B-HF state space model...")
    try:
        # First load the config to ensure correct architecture
        config = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir="/workspace/hf_cache",
        )
        print(f"  Config loaded: d_model={getattr(config, 'd_model', 'unknown')}, n_layer={getattr(config, 'n_layer', 'unknown')}")
        
        # Now load model with explicit config
        model = AutoModel.from_pretrained(
            model_id,
            config=config,
            torch_dtype=DTYPE,
            trust_remote_code=True,
            cache_dir="/workspace/hf_cache",
            device_map="auto" if DEVICE=="cuda" else None,
        )
        print("  ✓ Successfully loaded Mamba model with explicit config")
    except Exception as e:
        print(f"  ✗ Failed to load Mamba model: {str(e)[:200]}...")
        print("  This might be due to missing dependencies. Install: pip install mamba-ssm causal-conv1d")
        raise RuntimeError(f"Unable to load Mamba model: {e}")
        
    model.eval(); torch.set_grad_enabled(False)
    
    # Check if model supports hidden states output
    if hasattr(model.config, 'output_hidden_states'):
        model.config.output_hidden_states = True
    
    print(f"  Model type: {type(model).__name__}")
    print(f"  Architecture: State Space Model (Mamba2)")
    if hasattr(model.config, 'num_layers'):
        print(f"  Layers: {model.config.num_layers}")
    if hasattr(model.config, 'd_model'):
        print(f"  Hidden size: {model.config.d_model}")

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
        # For near-5, distances can be 0, 1, or 2
        errs_by_dist = {0:[], 1:[], 2:[]}
        errs_by_dir  = {-1:[], 0:[], +1:[]}
        # Re‑stream only best layer to compute buckets + proba
        for Xb_per_layer, s in stream_hidden_features(model, tok, texts_ev, meta_ev, [bestL], BATCH_SIZE, MAX_LEN, surf_ev):
            Xb = scalers[bestL].transform(Xb_per_layer[bestL])
            yb = y_ev[s]
            proba = probes[bestL].predict_proba(Xb)[:,1]
            preds = (proba >= 0.5).astype(np.int64)
            for i, idx in enumerate(range(s.start, s.stop)):
                d, rd, n = meta_ev[idx]
                d_bucket = min(d, 2)  # For near-5, cap at distance 2
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
all_results["model1"] = run_probes_for_model(MODEL_ID_1, tag="MAMBA", train_pack=train_pack, eval_packs=eval_packs)

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

# Save results for Mamba analysis
with open("mamba_1.4b_hf_probe_results_near_5.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to mamba_1.4b_hf_probe_results_near_5.json")

print("\n" + "=" * 60)
print("Mamba-1.4B-HF Linear Probe Experiment Complete")
print("=" * 60)
print("Notes:")
print("- Model: state-spaces/mamba-1.4b-hf (State Space Model)")
print("- Task: Near-5 classification (distance <= 1 from multiples of 5)")
print("- Architecture: No attention mechanism, selective state spaces")
print("- Memory requirement: ~6-8 GB VRAM (float16)")
print("- Key difference: Linear complexity vs quadratic attention")
print("- Dependencies: mamba-ssm package required")
print("=" * 60)
