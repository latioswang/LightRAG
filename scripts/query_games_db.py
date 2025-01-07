import os
import glog as log
import argparse
import asyncio
from pathlib import Path
from lightrag import LightRAG, QueryParam
from IPython.terminal.embed import InteractiveShellEmbed

from lightrag.llm import gpt_4o_complete, gpt_4o_mini_complete

def create_interactive_shell():
    # Create namespace for the shell
    namespace = {
        'QueryParam': QueryParam,
        'asyncio': asyncio
    }
    
    # Define helper functions with lazy RAG initialization
    def get_rag():
        """Lazily initialize RAG instance"""
        if not hasattr(get_rag, '_instance'):
            get_rag._instance = LightRAG(
                working_dir="./data/games/lightrag",
                llm_model_func=gpt_4o_complete,
                graph_storage="NetworkXStorage",
                enable_llm_cache=True,
                addon_params={
                    "entity_types": [
                        "game", 
                        "game_publisher", 
                        "review", 
                        "review_author", 
                        "platform", 
                        "outlet", 
                        "score_format",
                    ],
                }
            )
        return get_rag._instance

    def sync_query(q: str, level="global"):
        """Query the LightRAG database (synchronous wrapper)"""
        rag = get_rag()
        response = asyncio.run(
            rag.aquery(q, param=QueryParam(mode=level))
        )
        print("\n=== Response ===")
        print(response)
        return response

    def sync_debug(q: str, level="global"):
        """Debug query the LightRAG database (synchronous wrapper)"""
        rag = get_rag()
        # Get prompt
        prompt = asyncio.run(
            rag.aquery(q, param=QueryParam(mode=level, only_need_prompt=True))
        )
        print("=== Prompt ===")
        print(prompt)
        
        # Get response
        print("\n=== Response ===")
        result = asyncio.run(
            rag.aquery(q, param=QueryParam(mode=level))
        )
        print(result)
        return result

    # Add helper functions to namespace
    namespace['q'] = sync_query
    namespace['d'] = sync_debug
    namespace['get_rag'] = get_rag

    # Create and configure shell
    shell = InteractiveShellEmbed(
        banner1="""
Welcome to LightRAG Interactive Shell!

Available commands:
- q(question, level="global")  # Query the database
- d(question, level="global")  # Show prompt and query result
- get_rag()                    # Get RAG instance

Levels: "naive", "local", "global", "hybrid"
""",
        exit_msg="Goodbye!"
    )
    
    return shell, namespace

def main():
    # Start interactive shell without initializing RAG
    shell, namespace = create_interactive_shell()
    shell.mainloop(local_ns=namespace)

if __name__ == "__main__":
    main()
