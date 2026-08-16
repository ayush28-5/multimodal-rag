import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EXTRACTED_IMAGES_DIR = os.path.join(BASE_DIR, "extracted_images")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# ============================================
# MODELS
# ============================================
VLM_MODEL = "microsoft/Florence-2-large"
EMBEDDING_MODEL = "nomic-embed-text"

# Primary LLM — switched to Phi for low-latency inference
# phi3:mini = 3.8B, fastest. phi3:medium = 14B, slower but higher quality.
LLM_MODEL = "phi3:mini"

# Optional separate (smaller) model for query rewriting
REWRITER_MODEL = "phi3:mini"

# ============================================
# CHUNKING
# ============================================
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MIN_TEXT_LENGTH = 40
MIN_IMAGE_SIZE = (100, 100)

# ============================================
# RETRIEVAL
# ============================================
# Retrieve more candidates, then trim to top-K relevant after dedup
TOP_K_CANDIDATES = 6      # how many ChromaDB returns
TOP_K_RESULTS = 3         # how many actually go to the LLM

# Distance threshold — chunks above this are dropped as irrelevant
MAX_DISTANCE = 1.4

# ChromaDB
COLLECTION_NAME = "multimodal_rag"

# ============================================
# FLOWCHART DETECTION
# ============================================
FLOWCHART_KEYWORDS = ["arrow", "flow", "process", "step", "decision", "start", "end", "yes", "no"]
FLOWCHART_ASPECT_RATIO_RANGE = (0.5, 2.0)

# ============================================
# LLM INFERENCE PARAMETERS — tuned for speed
# ============================================
LLM_NUM_CTX = 2048         # context window — smaller = faster
LLM_NUM_PREDICT = 350      # max output tokens — keeps answers concise
LLM_TEMPERATURE = 0.2      # low temperature for factual answers
LLM_TOP_P = 0.9
LLM_TOP_K = 40
LLM_REPEAT_PENALTY = 1.1

# Streaming
ENABLE_STREAMING = True

# ============================================
# DEBUG
# ============================================
ENABLE_PROFILING = True    # logs per-stage timings to terminal