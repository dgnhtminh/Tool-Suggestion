from app.core.tool_suggestion.constant import COLLECTION_NAME, CHROMADB_DIR
from app.core.tool_suggestion import ToolSuggestion
from app.core.tool_suggestion import utils as chroma_utils

def re_embed():
    # import os
    # import shutil
    collection = chroma_utils.get_collection(COLLECTION_NAME, path=CHROMADB_DIR)

    res = collection.get()

    chroma_utils.delete_collection(COLLECTION_NAME, path=CHROMADB_DIR)

    # if os.path.exists(CHROMADB_DIR):
    #     shutil.rm(CHROMADB_DIR)
    #     os.mkdir(CHROMADB_DIR, exists=True)


    new_collection = chroma_utils.create_collection(COLLECTION_NAME, path=CHROMADB_DIR)

    if len(res["ids"]) == 0:
        return
    new_collection.add(
        ids=res["ids"],
        documents=res["documents"],
        metadatas=res["metadatas"],
    )
if __name__ == "__main__":
    re_embed()