from dotenv import load_dotenv
load_dotenv(override=True)

import os

# Download embedding model if not found
EMBEDDING_SAVE_PATH = os.path.join(os.getenv("TOOLCALL_DATA_DIR"), "embedding_model", "paraphrase-multilingual-MiniLM-L12-v2-onnx")
def download_embedding_model():
    print("Downloading EMBEDDING MODEL")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="onnx-models/paraphrase-multilingual-MiniLM-L12-v2-onnx", 
        local_dir=EMBEDDING_SAVE_PATH, 
        force_download=False, resume_download=True)

if not os.path.exists(EMBEDDING_SAVE_PATH):
    download_embedding_model()