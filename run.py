from dotenv import load_dotenv
load_dotenv(override=True)

import os
import json
import time

from app.core.tool_suggestion.constant import COLLECTION_NAME, CHROMADB_DIR
from app.core.tool_suggestion import utils as chroma_utils
from app.core.tool_suggestion import ToolSuggestion

def get_all_tool_ids():
    collection = chroma_utils.get_collection(name=COLLECTION_NAME, path=CHROMADB_DIR)
    all_res = collection.get()
    all_tools = {}
    for i in range(len(all_res["metadatas"])):
        meta = all_res["metadatas"][i]
        doc = all_res["documents"][i]
        if meta["tool_id"] not in all_tools:
            all_tools[meta["tool_id"]] = []
        all_tools[meta["tool_id"]].append({"document": doc, **meta})
    return all_tools

def save_tools_to_file(tools, filename="all_tools.json"):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(tools, f, indent=4, ensure_ascii=False)
        print(f"Đã lưu thành công thông tin của {len(tools)} tool vào file '{filename}'")
    except Exception as e:
        print(f"Error: {e}")

all_tools = get_all_tool_ids()
print("Num Tools", len(all_tools.keys()))
# save_tools_to_file(all_tools)
# print(json.dumps(all_tools, indent=4))


def print_tool_documents(tool_id):
    for doc in all_tools[tool_id]:
        print(f"({doc['type']}) {doc['document']}")


def get_suggestion(prompt):
    suggestions = ToolSuggestion.suggest(prompt, all_tools.keys(), [])
    return suggestions
    

if __name__ == "__main__":
    while True:
        prompt = input("\n\nEnter a prompt: ")
        t0 = time.time()
        suggestions = get_suggestion(prompt)
        t1 = time.time()
        print(f"Time: {t1 - t0:.5f}")
        print("Confident", suggestions.is_confident_in_top_tool)
        print("Tool Results:\n")
        for i, tool in enumerate(suggestions.tool_choices):
            print(f"RESULT {i + 1}")
            print("ID: ", tool.tool_id)
            print_tool_documents(tool.tool_id)
            print()
