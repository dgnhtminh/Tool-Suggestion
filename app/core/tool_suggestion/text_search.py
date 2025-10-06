import re
import os
import json
import bm25s
import unicodedata
import fasttext
import urllib.request
from app.core.tool_suggestion.constant import BM25_INDEX_DOC_TYPES, COLLECTION_NAME, CHROMADB_DIR, TOOLCALL_DATA_DIR, STOPWORDS_EN
from app.core.tool_suggestion import utils as chroma_utils


# Constants
INDEX_PATH = TOOLCALL_DATA_DIR + "bm25_index_path"
os.makedirs(INDEX_PATH, exist_ok=True)

INDEX_METADATA_PATH = TOOLCALL_DATA_DIR + "bm25_metadata"
os.makedirs(INDEX_METADATA_PATH, exist_ok=True)

# Fasttext model
fasttext_model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
fasttext_local_path = os.path.join(TOOLCALL_DATA_DIR, "lid.176.ftz")

# Download model only if it doesn't exist
if not os.path.exists(fasttext_local_path):
    print("Downloading FastText language ID model...")
    urllib.request.urlretrieve(fasttext_model_url, fasttext_local_path)

fasttext_model = fasttext.load_model(fasttext_local_path)

# Memory
bm25_retriever = None
bm25_info = {}
try:
    if os.path.exists(INDEX_PATH):
        bm25_retriever = bm25s.BM25.load(INDEX_PATH, load_corpus=True, allow_pickle=True)
    if os.path.exists(INDEX_METADATA_PATH + ".json"):
        with open(INDEX_METADATA_PATH + ".json", "r", encoding="utf-8") as f:
            bm25_info = json.load(f)
except Exception as e:
    print(f"BM25 Index Not Loaded: {e}")

def fast_lang_detect(text):
    predictions = fasttext_model.predict(text.strip().replace("\n", " "), k=1)
    return predictions[0][0].replace('__label__', '')

def remove_accents(s):
    s = re.sub('\u0110', 'D', s)
    s = re.sub('\u0111', 'd', s)
    return unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8')

def tokenize(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)  # Remove punctuation, keep all unicode letters (including accents)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in STOPWORDS_EN]
    return words

def create_index():
    import shutil
    print("Creating index...")
    if os.path.exists(INDEX_PATH):
        print("Removing existing index...")
        shutil.rmtree(INDEX_PATH)
        os.makedirs(INDEX_PATH, exist_ok=True)
    if os.path.exists(INDEX_METADATA_PATH):
        print("Removing existing metadata...")
        shutil.rmtree(INDEX_METADATA_PATH)
        os.makedirs(INDEX_METADATA_PATH, exist_ok=True)
    # get documents from chromadb
        
    collection = chroma_utils.get_collection(name=COLLECTION_NAME, path=CHROMADB_DIR)

    if not collection:
        return "Failed: the Chromadb server may be not running."

    all_entries = collection.get()

    documents = all_entries["documents"]
    metadatas = all_entries["metadatas"]
    ids = all_entries["ids"]

    # Tokenize documents
    tokenized_documents = []
    corpus_metadatas = []

    for i, doc in enumerate(documents):
        metadata = metadatas[i]
        doc_type = metadata["type"]
        if doc_type not in BM25_INDEX_DOC_TYPES:
            continue

        tokenized_documents.append(tokenize(doc))

        corpus_metadatas.append({**metadata, "id": ids[i], "text": doc})

        lang = fast_lang_detect(doc[:100])
        if lang == 'vi':
            no_accents = remove_accents(doc)
            tokenized_documents.append(tokenize(no_accents))
            corpus_metadatas.append({**metadata, "id": ids[i], "text": no_accents})

    print(f"Creating index for {len(tokenized_documents)} documents")
    global bm25_retriever
    bm25_retriever = bm25s.BM25(
        corpus=corpus_metadatas,
        k1=1.5, 
        b=0.75
    )
    bm25_retriever.index(tokenized_documents, show_progress=True, leave_progress=True)
    bm25_retriever.save(INDEX_PATH, corpus=corpus_metadatas, allow_pickle=True)

    global bm25_info
    bm25_info = {
        "num_docs": len(tokenized_documents)
    }
    with open(INDEX_METADATA_PATH + ".json", 'w', encoding="utf-8") as f:
        json.dump(bm25_info, f, indent=4, ensure_ascii=False)

    return "Success!"

def search_bm25(query, top_k = 50):
    if bm25_retriever is None:
        print("BM25 Index Not Created")
        return [], []

    tokenized_query = tokenize(query)

    n_to_retrieve = min(bm25_info["num_docs"], top_k)
    r, s = bm25_retriever.retrieve([tokenized_query], k=n_to_retrieve)
    doc_results = r[0]
    doc_scores = s[0]
    return doc_results, doc_scores