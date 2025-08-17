# prepare_models.py — downloads FinBERT into models/finbert if missing
import os, sys
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

TARGET = "models/finbert"
if not os.path.isdir(TARGET) or not os.listdir(TARGET):
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print("Install deps first: pip install -r requirements.txt"); sys.exit(1)
    print("Downloading ProsusAI/finbert to", TARGET, "...")
    snapshot_download(
        repo_id="ProsusAI/finbert",
        local_dir=TARGET,
        local_dir_use_symlinks=False,
        revision="main",
    )
    print("Done.")
else:
    print("Model already present at", TARGET)
