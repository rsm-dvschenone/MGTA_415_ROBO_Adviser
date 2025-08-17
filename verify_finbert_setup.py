
"""
verify_finbert_setup.py
-----------------------
Prints Python/env info, verifies transformers/torch imports,
and attempts to load ProsusAI/finbert (first from local cache, then online).
"""
import os, sys, importlib, traceback
from pprint import pprint

print("=== Python ===")
print("sys.executable:", sys.executable)
print("sys.version:", sys.version.splitlines()[0])
print()

def show_pkg(name):
    try:
        m = importlib.import_module(name)
        path = getattr(m, "__file__", "<namespace>")
        ver = getattr(m, "__version__", "<no __version__>")
        print(f"{name}: version={ver} path={path}")
        return m
    except Exception as e:
        print(f"{name}: IMPORT FAILED -> {e.__class__.__name__}: {e}")
        return None

print("=== Packages ===")
t = show_pkg("transformers")
th = show_pkg("torch")
show_pkg("safetensors")
show_pkg("huggingface_hub")
show_pkg("tokenizers")
print()

print("=== Env (HF cache) ===")
print("HF_HOME:", os.getenv("HF_HOME"))
print("HF_HUB_CACHE:", os.getenv("HF_HUB_CACHE"))
print()

if t is None:
    print("Transformers not importable in THIS interpreter.")
    print("Fix: run `python -m pip install transformers` with the same `python` shown above.")
    sys.exit(2)

if th is None:
    print("Torch not importable in THIS interpreter. FinBERT requires torch (CPU is fine).")
    print("Fix (CPU-only): python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu")
else:
    try:
        print("torch.cuda.is_available():", th.cuda.is_available())
    except Exception as e:
        print("torch.cuda.is_available() check failed:", e)

print("\n=== FinBERT test (local cache first) ===")
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert", local_files_only=True)
    mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", local_files_only=True)
    clf = pipeline("text-classification", model=mdl, tokenizer=tok, truncation=True, return_all_scores=True)
    print("Local cache OK ✅ (ProsusAI/finbert found).")
except Exception as e:
    print("Local cache not found or failed:", f"{e.__class__.__name__}: {e}")
    print("\nTrying to fetch online (this requires internet & no firewall blocking huggingface.co)...")
    try:
        tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        clf = pipeline("text-classification", model=mdl, tokenizer=tok, truncation=True, return_all_scores=True)
        print("Online download OK ✅ (ProsusAI/finbert).")
    except Exception as e2:
        print("Online fetch FAILED:", f"{e2.__class__.__name__}: {e2}")
        print("\nCommon fixes:")
        print("  1) Network blocks: try from an unblocked network or pre-download the model:")
        print("     huggingface-cli download ProsusAI/finbert --local-dir models/finbert")
        print("     ...then point your code to model_name='models/finbert'")
        print("  2) Torch wheel mismatch: install CPU-only torch wheel if CUDA is unavailable:")
        print("     python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu")
        print("  3) Interpreter mismatch: ensure `python` used to run scripts matches the one where you installed packages.")
        sys.exit(3)

print("\n=== Mini inference ===")
try:
    outs = clf(["NVIDIA beats on revenue; guidance raised.", "Valuation risk could cap upside."])
    print("Pipeline ran. Example output (first item):", outs[0])
except Exception as e:
    print("Inference failed:", f"{e.__class__.__name__}: {e}")
    traceback.print_exc(limit=1)
    sys.exit(4)

print("\nAll good ✅")
