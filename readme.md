# Tool Suggestion
## Setup
0. Install Python and pip
1. `python -m venv venv`
2. `source venv\Scripts\activate` (Run each time after opening a new terminal)
3. `pip install -r requirements.txt`
4. `pip install transformers --no-deps`
5. Set `TOOLCALL_DATA_DIR` in .env to the path of ./data
6. `python download_model.py`
7. `python run.py`

## Overview
Suggest relevant tools / agents given a user prompt. The `data` folder contains a snapshot of the current agents on maxflow.ai.

The `ToolSuggestion.suggest()` method first queries the documents from ChromaDB (e.g. tool name, description, sample prompts) and scales the cosine distance by a quadratic function with a scaling constant s [-0.5, 0.5] depending on the type of the document. If a document has a distance of less than 0.1, the resulting tool IDs are returned immediately. If no documents have a distance less than 0.1, the embedding search results are merged with BM25 search results (using the top 10 results from each). The set of tool IDs for the merged results is returned.

To add a new tool, call `ToolSuggestion.add()`, which embeds the tool name, description, and sample prompts as documents in ChromaDB. Then run `python re_index.py` to recreate the BM25 index. The BM25 index is created by `text_search.create_index()` using the documents in ChromaDB. The documents are first tokenized (using simple regex, text splitting, and stop word removal). Then the tokenized documents are added to the BM25 corpus. If a document is in Vietnamese (detected by fasttext), an additional document is added with the accents removed. The details of the BM25 algorithm are handled by the `bm25s` library.


    