import glog as log
# log.setLevel(log.DEBUG)
import os

from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete

WORKING_DIR = "./movies"
INPUT_DIR="./data/"

os.makedirs(WORKING_DIR, exist_ok=True)

os.environ["OPENAI_API_KEY"] = "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"

 
# neo4j
BATCH_SIZE_NODES = 500
BATCH_SIZE_EDGES = 100
os.environ["NEO4J_URI"] = "neo4j+s://49341d95.databases.neo4j.io"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "VRNrnd5iPbLlH1RHqoHiRC9DDlHSTE2oDnLCsBF2_bg"

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,
    # llm_model_func=gpt_4o_complete
    graph_storage="Neo4JStorage",
    
)


for filename in os.listdir(INPUT_DIR):
    filepath = os.path.join(INPUT_DIR, filename)
    if os.path.isfile(filepath):
        log.info(f"Inserting content from file: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            rag.insert(f.read())

# Perform naive search
query = "What are the top themes?"
log.info(f"Executing naive search query: {query}")
print(
    rag.query(query, param=QueryParam(mode="naive"))
)

# Perform local search
log.info(f"Executing local search query: {query}")
print(
    rag.query(query, param=QueryParam(mode="local"))
)

# Perform global search
log.info(f"Executing global search query: {query}")
print(
    rag.query(query, param=QueryParam(mode="global"))
)

# Perform hybrid search
log.info(f"Executing hybrid search query: {query}")
print(
    rag.query(query, param=QueryParam(mode="hybrid"))
)
