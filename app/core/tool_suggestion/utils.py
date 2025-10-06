import os
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from app.core.tool_suggestion.constant import CHROMADB_DIR, EMBEDDING_MODEL_DIR

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from chromadb import EmbeddingFunction, Documents, Embeddings

from threading import Thread

os.environ["TOKENIZERS_PARALLELISM"]="true" # silence huggingface error

def get_chromadb_client(path, reset=False):
    return chromadb.PersistentClient(path)

class CustomONNXEmbeddingFunction(EmbeddingFunction):    
    def __init__(self, model_path: str):
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._session = ort.InferenceSession(f"{model_path}/model.onnx", providers=["CPUExecutionProvider"])
 
        # List of input node names (e.g., ['input_ids', 'attention_mask'])
        self._input_names = [input.name for input in self._session.get_inputs()]
 
    def __call__(self, inputs: Documents) -> Embeddings:
        # Tokenize input to NumPy tensors
        encoded = self._tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="np"
        )
 
        # Filter only the inputs that the ONNX model expects
        onnx_inputs = {k: v for k, v in encoded.items() if k in self._input_names}
 
        # Run inference
        outputs = self._session.run(None, onnx_inputs)
        last_hidden_state = outputs[0]  # (B, L, H)
 
        # Mean pooling with attention mask
        attention_mask = encoded["attention_mask"].astype(np.float32)[..., None]  # (B, L, 1)
        summed = (last_hidden_state * attention_mask).sum(axis=1)                 # (B, H)
        lengths = attention_mask.sum(axis=1)                                      # (B, 1)
        lengths = np.maximum(lengths, 1e-9)
        emb = summed / lengths                                                    # mean-pooled
 
        # Normalize embeddings
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
 
        return emb

embedding_function = CustomONNXEmbeddingFunction(EMBEDDING_MODEL_DIR + "/paraphrase-multilingual-MiniLM-L12-v2-onnx")

def create_collection(name, path=CHROMADB_DIR):
    print("Creating collection:", name)
    chromadb_client = get_chromadb_client(path=path, reset=True)
    return chromadb_client.create_collection(name=name, embedding_function=embedding_function)

def get_collection(name, path=CHROMADB_DIR, reset=False):
    chromadb_client = get_chromadb_client(path=path, reset=reset)
    collection = chromadb_client.get_collection(name=name, embedding_function=embedding_function)
    return collection
    
def delete_collection(name, path=CHROMADB_DIR):
    try:
        chromadb_client = get_chromadb_client(path=path, reset=True)
        return chromadb_client.delete_collection(name=name)
    except Exception as e:
        print("Error deleting collection:", e)
        return None

def get_or_create_collection(name, path=CHROMADB_DIR):
    try:
        collection = get_collection(name, path)
    except Exception as e:
        print("Collection not found. Creating new collection.")
        collection = create_collection(name, path)
    if collection is None:
        print("Collection is None, creating new collection.")
        collection = create_collection(name, path)
    return collection