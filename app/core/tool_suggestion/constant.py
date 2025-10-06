import os

COLLECTION_NAME = "Genie"
TOOLCALL_DATA_DIR = os.getenv("TOOLCALL_DATA_DIR", "D:\\maxflow\\data\\tool_suggestion\\")
CHROMADB_DIR = TOOLCALL_DATA_DIR + "chromadb"
EMBEDDING_MODEL_DIR = TOOLCALL_DATA_DIR + "embedding_model"

class DocumentTypes:
    TOOL_NAME = "name"
    TOOL_DESCRIPTION = "desc"
    SAMPLE_PROMPT_TRIGGER = "trig"
    SAMPLE_PROMPT_INPUT = "inp"
    NONE = "none"
    UNUSED = "unused"
    PRE_DEFINED_QA = "pre_defined_qa"

BM25_INDEX_DOC_TYPES = ["name", "desc", "trig", "inp", "pre_defined_qa"]

BM25_PARAMS = {
    "name":             {'w': 1.0},
    "desc":             {'w': 0.7},
    "trig":             {'w': 0.5},
    "inp":              {'w': 0.7},
    "pre_defined_qa":   {'w': 0.7}
}

STOPWORDS_EN = [
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "if",
    "in",
    "into",
    "is",
    "not",
    "of",
    "on",
    "or",
    "such",
    "the",
    "their",
    "there",
    "these",
    "they",
    "this",
    "was",
    "will",
    "with"
]