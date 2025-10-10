import re
import itertools
import uuid
import numpy as np
from pydantic import BaseModel
from typing import List, Optional

from app.core.tool_suggestion.constant import DocumentTypes, COLLECTION_NAME, CHROMADB_DIR, BM25_PARAMS
from app.core import constant
from app.core.tool_suggestion import utils as chroma_utils
from app.core.tool_suggestion.text_search import search_bm25


class Message(BaseModel):
    role: str
    content: Optional[str]

class ToolChoice(BaseModel):
    tool_id: int
    type: str
    gpt_name: Optional[str] = None
    gpt_link: Optional[str] = None
    gpt_description: Optional[str] = None
    sample_question: Optional[str] = None
    metadata: Optional[dict] = None

class ToolSuggestionModel(BaseModel):
    is_confident_in_top_tool: bool
    tool_choices: List[ToolChoice]
    nickname: Optional[str] = None
    first_word: Optional[str] = None

class ToolSuggestion:

    @staticmethod
    def add(tool_id: int,
            name: str,
            description: str,
            sample_user_prompt: str,
            predefined_qas: str):
        collection = chroma_utils.get_collection(name=COLLECTION_NAME, path=CHROMADB_DIR, reset=True)
        documents = []
        if sample_user_prompt and isinstance(sample_user_prompt, str):
            for sample_prompt in sample_user_prompt.splitlines():
                if sample_prompt.strip() != "":
                    documents.append((sample_prompt.strip(), DocumentTypes.SAMPLE_PROMPT_TRIGGER))
        if description:
            documents.append((description, DocumentTypes.TOOL_DESCRIPTION))
        if name:
            documents.append((name, DocumentTypes.TOOL_NAME))

        for document, doc_type in documents:
            ToolSuggestion.add_tool_document(collection, tool_id=tool_id, document=document, doc_type=doc_type, is_user_prompt=False)
        
        for qa in predefined_qas:
            if qa.strip() != "":
                question = qa["question"]
                answer = qa["answer"]
                threshold = qa["threshold"]
                ToolSuggestion.add_tool_document(collection, 
                                                tool_id=tool_id, 
                                                document=question, 
                                                doc_type=DocumentTypes.PRE_DEFINED_QA,
                                                metadata={"answer": answer, "threshold": threshold})
        return constant.SUCCESS, 200, "Added tool suggestion successfully"

    @staticmethod
    def update(tool_id: int,
               name: str,
               description: str,
               sample_user_prompt: str,
               predefined_qas: str=None):
        collection = chroma_utils.get_collection(name=COLLECTION_NAME, path=CHROMADB_DIR, reset=True)
        if ToolSuggestion.check_exist_tool(collection, tool_id) is False:
            return ToolSuggestion.add(tool_id, name, description, sample_user_prompt, predefined_qas)
        
        entries = collection.get(where={"tool_id": tool_id})
        sample_prompts = [t.strip() for t in sample_user_prompt.splitlines() if t.strip() != ""] if isinstance(sample_user_prompt, str) else []
        predefined_qa_questions = [t.strip() for t in predefined_qas if t.strip() != ""] if isinstance(predefined_qas, list) else []

        for (id, doc, metadata) in zip(entries["ids"], entries["documents"], entries["metadatas"]):
            prompt_type = metadata["type"]
            if predefined_qas is None and prompt_type == DocumentTypes.PRE_DEFINED_QA:
                continue

            if prompt_type == DocumentTypes.PRE_DEFINED_QA and doc in predefined_qa_questions:
                predefined_qa_questions.remove(doc)
            elif prompt_type == DocumentTypes.TOOL_NAME and doc == name.strip():
                name = None
            elif prompt_type == DocumentTypes.TOOL_DESCRIPTION and doc == description.strip():
                description = None
            elif prompt_type == DocumentTypes.SAMPLE_PROMPT_TRIGGER and doc in sample_prompts:
                sample_prompts.remove(doc)
            elif prompt_type != DocumentTypes.NONE and prompt_type != DocumentTypes.UNUSED:
                collection.delete(ids=[id])

        documents = []
        if len(sample_prompts) > 0:
            for sample_prompt in sample_prompts:
                if sample_prompt.strip() != "":
                    documents.append((sample_prompt.strip(), DocumentTypes.SAMPLE_PROMPT_TRIGGER))
        if description:
            documents.append((description, DocumentTypes.TOOL_DESCRIPTION))
        if name:
            documents.append((name, DocumentTypes.TOOL_NAME))
        if len(predefined_qa_questions) > 0:
            for qa in predefined_qa_questions:
                question = qa["question"]
                answer = qa["answer"]
                threshold = qa["threshold"]
                ToolSuggestion.add_tool_document(collection, 
                                                tool_id=tool_id, 
                                                document=question, 
                                                doc_type=DocumentTypes.PRE_DEFINED_QA,
                                                metadata={"answer": answer, "threshold": threshold})
        for document, doc_type in documents:
            ToolSuggestion.add_tool_document(collection, tool_id=tool_id, document=document, doc_type=doc_type, is_user_prompt=False)
        return constant.SUCCESS, 200, "Updated tool suggestion successfully"

    @staticmethod
    def delete(tool_id: int):
        collection = chroma_utils.get_collection(name=COLLECTION_NAME, path=CHROMADB_DIR, reset=True)
        collection.delete(where={"tool_id": tool_id})

    @staticmethod
    def reciprocal_rank_fusion(ranked_lists: list, k: int= 60) -> list:
        rrf_scores = {}
        for ranked_list in ranked_lists:
            for rank, tool_id in enumerate(ranked_list):
                if tool_id not in rrf_scores:
                    rrf_scores[tool_id] = 0
                rrf_scores[tool_id] += 1 / (k + rank + 1) # rank bat dau tu 0

        sorted_tools = sorted(rrf_scores.keys(), key= lambda x: rrf_scores[x], reverse=True)
        return sorted_tools

    @staticmethod
    # def suggest(
    #     prompt: str,
    #     allowed_tools: Optional[List[int]],
    #     general_tool_ids: Optional[List[int]] = None
    # ) -> ToolSuggestionModel:
    #     allowed_tools = allowed_tools if isinstance(allowed_tools, list) else []
    #     if not isinstance(prompt, str):
    #         prompt = str(prompt) if prompt is not None else ""

    #     if prompt and prompt.startswith("#debug:"):
    #         prompt = prompt.removeprefix("#debug:").lstrip()

    #     if prompt is None or prompt == "":
    #         return ToolSuggestionModel(is_confident_in_top_tool=False, tool_choices=[])

    #     # Query tools
    #     embedding_results, embedding_distances = ToolSuggestion._query_tools_sorted_by_scaled_distance(prompt, allowed_tools)

    #     # shortcut if embedding distance is small
    #     if len(embedding_distances) > 0 and embedding_distances[0] < 0.1:
    #         suggestions = ToolSuggestion._get_tool_choices_from_embeddings_query(embedding_results, embedding_distances, 0.10)
    #         suggestions = ToolSuggestion.limit_n_tools_and_include_all_general_tools(suggestions, general_tool_ids)
    #         return suggestions
       
    #     bm25_results, bm25_scores = search_bm25(prompt)

    #     embedding_ids = embedding_results["ids"][0]
    #     embedding_metas = embedding_results["metadatas"][0]
    #     embedding_docs = embedding_results["documents"][0]

    #     embedding_tools = {}
    #     for i, doc_id in enumerate(embedding_ids):
    #         if len(embedding_tools) >= 10 or embedding_distances[i] > 1.5:
    #             break
    #         tool_id = embedding_metas[i]["tool_id"]
    #         if tool_id not in embedding_tools:
    #             embedding_tools[tool_id] = {
    #                 "document": embedding_docs[i],
    #                 "metadata": embedding_metas[i],
    #                 "embedding_distance": embedding_distances[i]
    #             }

    #     bm25_tools = {}
    #     for metadata, bm25_score in zip(bm25_results, bm25_scores):
    #         tool_id = metadata.get("tool_id")
    #         if tool_id not in allowed_tools:
    #             continue
    #         if len(bm25_tools) >= 10 or bm25_score <= 0.0:
    #             break
    #         if tool_id not in bm25_tools:
    #             bm25_tools[tool_id] = {
    #                 "document": metadata.get("text", ""),
    #                 "metadata": metadata,
    #                 "bm25_score": bm25_score
    #             }

    #     # merged_tools = {**bm25_tools, **embedding_tools}
    #     merged_tools = {**embedding_tools}

    #     for k, v in bm25_tools.items():
    #         if k not in merged_tools:
    #             merged_tools[k] = v

    #     # print("BM25")
    #     # for k, v in bm25_tools.items():
    #     #     print("Doc: ", v["document"])
    #     #     print("Meta: ", v["metadata"])
    #     #     print()

    #     # print("EMBEDDING")
    #     # for k, v in embedding_tools.items():
    #     #     print("Doc: ", v["document"])
    #     #     print("Meta: ", v["metadata"])
    #     #     print("Distance: ", v["embedding_distance"])
    #     #     print()

    #     # print("MERGED")
    #     # for k, v in merged_tools.items():
    #     #     print("Doc: ", v["document"])
    #     #     print("Meta: ", v["metadata"])
    #     #     print()

    #     # Create tool suggestions
    #     suggestions = ToolSuggestionModel(is_confident_in_top_tool=False, tool_choices=[])
    #     for tool_id, tool_data in merged_tools.items():
    #         metadata = tool_data["metadata"]
    #         doc_type = metadata["type"]
    #         sample_question = (
    #             tool_data["document"]
    #             if doc_type in [
    #                 DocumentTypes.SAMPLE_PROMPT_TRIGGER,
    #                 DocumentTypes.PRE_DEFINED_QA
    #             ] else None
    #         )
    #         suggestions.tool_choices.append(ToolChoice(
    #             tool_id=tool_id,
    #             type=doc_type,
    #             sample_question=sample_question,
    #             metadata=metadata
    #         ))

    #     suggestions = ToolSuggestion.limit_n_tools_and_include_all_general_tools(suggestions, general_tool_ids)

    #     return suggestions

    @staticmethod
    def suggest(
        prompt: str,
        allowed_tools: Optional[List[int]],
        general_tool_ids: Optional[List[int]] = None
    ) -> ToolSuggestionModel:
        # --- CÁC THAM SỐ TINH CHỈNH ---
        EMBEDDING_DISTANCE_THRESHOLD = 0.8 # Nới lỏng ngưỡng một chút để không bỏ sót
        BM25_SCORE_THRESHOLD = 0.1       # Nới lỏng ngưỡng một chút để không bỏ sót
        FINAL_RESULTS_LIMIT = 7          # Giới hạn số kết quả cuối cùng

        allowed_tools = allowed_tools if isinstance(allowed_tools, list) else []
        if not isinstance(prompt, str):
            prompt = str(prompt) if prompt is not None else ""

        if prompt and prompt.startswith("#debug:"):
            prompt = prompt.removeprefix("#debug:").lstrip()

        if prompt is None or prompt == "":
            return ToolSuggestionModel(is_confident_in_top_tool=False, tool_choices=[])

        # === GIAI ĐOẠN 1: TRUY XUẤT VÀ LỌC THÔ ===
        embedding_results, embedding_distances = ToolSuggestion._query_tools_sorted_by_scaled_distance(prompt, allowed_tools)
        
        # Đường tắt vẫn giữ lại
        if len(embedding_distances) > 0 and embedding_distances[0] < 0.1:
            suggestions = ToolSuggestion._get_tool_choices_from_embeddings_query(embedding_results, embedding_distances, 0.10)
            return ToolSuggestion.limit_n_tools_and_include_all_general_tools(suggestions, general_tool_ids)
        
        bm25_results, bm25_scores = search_bm25(prompt)

        # Lấy danh sách tool_id đã được LỌC và xếp hạng từ mỗi nguồn
        embedding_ranked_ids = []
        seen_in_embedding = set()
        for i, meta in enumerate(embedding_results["metadatas"][0]):
            if embedding_distances[i] > EMBEDDING_DISTANCE_THRESHOLD:
                continue # Bỏ qua nếu không đạt ngưỡng
            tool_id = meta["tool_id"]
            if tool_id not in seen_in_embedding:
                embedding_ranked_ids.append(tool_id)
                seen_in_embedding.add(tool_id)

        bm25_ranked_ids = []
        seen_in_bm25 = set()
        for i, meta in enumerate(bm25_results):
            if bm25_scores[i] < BM25_SCORE_THRESHOLD:
                continue # Bỏ qua nếu không đạt ngưỡng
            tool_id = meta.get("tool_id")
            if tool_id not in seen_in_bm25 and tool_id in allowed_tools:
                bm25_ranked_ids.append(tool_id)
                seen_in_bm25.add(tool_id)
        
        # === GIAI ĐOẠN 2: TÁI XẾP HẠNG VỚI RRF ===
        final_ranked_ids = ToolSuggestion.reciprocal_rank_fusion([embedding_ranked_ids, bm25_ranked_ids])

        # Giới hạn số kết quả cuối cùng
        final_tool_ids = final_ranked_ids[:FINAL_RESULTS_LIMIT]

        # === GIAI ĐOẠN 3: TẠO KẾT QUẢ TRẢ VỀ ===
        # (Giữ nguyên logic tạo suggestions như phiên bản RRF trước)
        all_retrieved_info = {}
        for i, meta in enumerate(embedding_results["metadatas"][0]):
            tool_id = meta["tool_id"]
            if tool_id not in all_retrieved_info:
                all_retrieved_info[tool_id] = {"document": embedding_results["documents"][0][i], "metadata": meta}
        for meta in bm25_results:
            tool_id = meta.get("tool_id")
            if tool_id not in all_retrieved_info:
                 all_retrieved_info[tool_id] = {"document": meta.get("text", ""), "metadata": meta}

        suggestions = ToolSuggestionModel(is_confident_in_top_tool=False, tool_choices=[])
        for tool_id in final_tool_ids:
            if tool_id in all_retrieved_info:
                tool_data = all_retrieved_info[tool_id]
                metadata = tool_data["metadata"]
                doc_type = metadata["type"]
                sample_question = (
                    tool_data["document"]
                    if doc_type in [DocumentTypes.SAMPLE_PROMPT_TRIGGER, DocumentTypes.PRE_DEFINED_QA] else None
                )
                suggestions.tool_choices.append(ToolChoice(
                    tool_id=tool_id, type=doc_type, sample_question=sample_question, metadata=metadata
                ))

        return ToolSuggestion.limit_n_tools_and_include_all_general_tools(suggestions, general_tool_ids)

    @staticmethod
    def check_exist_tool(collection, tool_id: int):
        try:
            entries = collection.get(where={"tool_id": tool_id}, limit=1)
            return True if entries is not None and len(entries) > 0 else False
        except Exception as e:
            return False

    @staticmethod
    def add_tool_document(collection, tool_id, document, doc_type, is_user_prompt=False, metadata={}):
        embeddings = None
        if not is_user_prompt and doc_type in [DocumentTypes.SAMPLE_PROMPT_TRIGGER, DocumentTypes.PRE_DEFINED_QA]:
            combinations = ToolSuggestion.generate_string_combinations(document)
            if len(combinations) > 1:
                embeddings = chroma_utils.embedding_function(combinations)
                average_embedding = np.mean(embeddings, axis=0)
                embeddings = [average_embedding.tolist()]

        metadata = {"tool_id": tool_id, "type": doc_type, **metadata}
        if is_user_prompt:
            metadata["is_user_prompt"] = True

        collection.add(
            documents=[document],
            metadatas=[metadata],
            embeddings=embeddings,
            ids=[str(uuid.uuid4())]
        )

    @staticmethod
    def generate_string_combinations(s):
        parts = re.split(r"(\[[^\]]+\])", s)

        choices_lists = []
        format_string = ""

        for part in parts:
            if len(part) >= 2 and part[0] == "[" and part[-1] == "]":
                choices = [choice.strip() for choice in part[1:-1].split(";")]
                choices_lists.append(choices)
                format_string += "{}"
            else:
                format_string += part

        combinations = itertools.product(*choices_lists)
        result_strings = [format_string.format(*combo) for combo in combinations]
        return result_strings
    
    @staticmethod
    def _query_tools_sorted_by_scaled_distance(prompt, allowed_tools):
        query_results = ToolSuggestion._query_tools(prompt=prompt, allowed_tools=allowed_tools)
        scaled_distances = ToolSuggestion._get_query_scaled_distances(query_results)
        if not query_results["documents"][0]:
            return query_results, scaled_distances
        combined_results = zip(query_results["documents"][0], query_results["metadatas"][0], scaled_distances)
        sorted_combined_results = sorted(combined_results, key=lambda x: x[2])
        query_results["documents"][0], query_results["metadatas"][0], scaled_distances = zip(*sorted_combined_results)        
        
        return query_results, scaled_distances

    @staticmethod
    def _query_tools(
        prompt: str, 
        doc_types: List[str]=[
            DocumentTypes.TOOL_NAME,
            DocumentTypes.TOOL_DESCRIPTION,
            DocumentTypes.SAMPLE_PROMPT_TRIGGER,
            DocumentTypes.PRE_DEFINED_QA
        ], 
        allowed_tools: Optional[List[int]] = None):
        doctype_filter = {
            "type": {"$in": doc_types }
        }
        metadata_filter = doctype_filter
        if allowed_tools:
            metadata_filter = {
                "$and": [
                    doctype_filter,
                    {"tool_id": {"$in": allowed_tools}}
                ]
            }

        collection = chroma_utils.get_collection(name=COLLECTION_NAME, path=CHROMADB_DIR)
        results = {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}
        if collection:
            results = collection.query(
                 query_texts=prompt,
                n_results=20,
                where=metadata_filter
            )
        return results
    
    @staticmethod
    def _get_query_scaled_distances(results):
        scaled_distances = []
        for i in range(len(results["documents"][0])):
            tool_type = results["metadatas"][0][i]["type"]
            distance = results["distances"][0][i]

            scale_constant = 0.0
            if tool_type == DocumentTypes.TOOL_NAME:                scale_constant = -0.1
            elif tool_type == DocumentTypes.TOOL_DESCRIPTION:       scale_constant = -0.2
            elif tool_type == DocumentTypes.SAMPLE_PROMPT_TRIGGER:  scale_constant = -0.2

            scaled_distance = ToolSuggestion._scale_distance(distance, scale_constant)
            scaled_distances.append(scaled_distance)
        return scaled_distances
    
    @staticmethod
    def _scale_distance(d, s):
        scaled = d + (1 - (1 - d) * (1 - d)) * s
        return scaled
    
    @staticmethod
    def _get_tool_choices_from_embeddings_query(query_results, scaled_distances, threshold):
        metadatas = query_results["metadatas"][0]
        documents = query_results["documents"][0]
        tool_choices: List[ToolChoice] = []
        seen_tools = set()
        tool_distances_under_threshold = []
        for i in range(len(metadatas)):
            tool_id = int(metadatas[i]["tool_id"])
            prompt_type = str(metadatas[i]["type"])
            sample_question = documents[i] if prompt_type in [DocumentTypes.SAMPLE_PROMPT_TRIGGER, 
                                                            DocumentTypes.PRE_DEFINED_QA] else None
            scaled_distance = scaled_distances[i]

            if scaled_distance < 1.2 and tool_id not in seen_tools:
                seen_tools.add(tool_id)
                tool_choices.append(
                    ToolChoice(
                        tool_id=tool_id, 
                        type=prompt_type,
                        sample_question=sample_question,
                        metadata=metadatas[i]
                    )
                )
                if scaled_distance <= threshold:
                    tool_distances_under_threshold.append(scaled_distance)
            if len(seen_tools) >= 16:
                break
        
        is_confident = False
        if len(tool_distances_under_threshold) == 0:
            is_confident = False
        elif len(tool_distances_under_threshold) == 1:
            is_confident = True
        elif (tool_distances_under_threshold[0] / max(tool_distances_under_threshold[1], 0.001)) <= 0.5:
            is_confident = True

        res_tool_choices = ToolSuggestionModel(
            is_confident_in_top_tool=is_confident,
            tool_choices=tool_choices
        )
        return res_tool_choices
    
    @staticmethod
    def limit_n_tools_and_include_all_general_tools(tool_suggestions: ToolSuggestionModel, general_tool_ids: List[int]) -> ToolSuggestionModel:
        tool_choices = tool_suggestions.tool_choices
        new_tool_choices: List[ToolChoice] = []
        for tool in tool_choices:
            if tool.tool_id in general_tool_ids:
                general_tool_ids.remove(tool.tool_id)
            new_tool_choices.append(tool)
        
        for gen_tool_id in general_tool_ids:
            new_tool_choices.append(ToolChoice(
                tool_id=gen_tool_id,
                type=DocumentTypes.SAMPLE_PROMPT_TRIGGER,
            ))
        new_tool_suggestions = tool_suggestions.model_copy()
        new_tool_suggestions.tool_choices = new_tool_choices
        return new_tool_suggestions