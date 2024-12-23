import os
import glog as log
import argparse
import asyncio
from pathlib import Path
from lightrag import LightRAG, QueryParam
from IPython.terminal.embed import InteractiveShellEmbed

from lightrag.llm import gpt_4o_mini_complete

def create_interactive_shell(rag: LightRAG):
    # Create namespace for the shell
    namespace = {
        'rag': rag,
        'QueryParam': QueryParam,
        'asyncio': asyncio
    }
    
    # Define helper functions
    def sync_query(q: str, level="global"):
        """Query the LightRAG database (synchronous wrapper)"""
        response = asyncio.run(
            rag.aquery(q, param=QueryParam(mode=level))
        )
        print("\n=== Response ===")
        print(response)
        return response

    def sync_debug(q: str, level="global"):
        """Debug query the LightRAG database (synchronous wrapper)"""
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

    # Create and configure shell
    shell = InteractiveShellEmbed(
        banner1="""
Welcome to LightRAG Interactive Shell!

Available commands:
- q(question, level="global")  # Query the database
- d(question, level="global")  # Show prompt and query result

Levels: "naive", "local", "global", "hybrid"
""",
        exit_msg="Goodbye!"
    )
    
    return shell, namespace

def main():
    # Initialize LightRAG with the same configuration
    rag = LightRAG(
        working_dir="./data/games/lightrag",
        llm_model_func=gpt_4o_mini_complete,
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
    
    # Start interactive shell
    shell, namespace = create_interactive_shell(rag)
    shell.mainloop(local_ns=namespace)

if __name__ == "__main__":
    main()
