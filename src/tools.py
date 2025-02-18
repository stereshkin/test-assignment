import json
import os
from src.vectorstore import VectorStore
import numpy as np
import faiss
from typing import Literal, Optional, List
from src.elasticsearch import ElasticSearchManager
from rich.console import Console
import difflib
from rich.panel import Panel
from rich.syntax import Syntax


class UpdateKnowledgeTool:
    def __init__(self, allowed_api_calls: int):
        self.allowed_api_calls = allowed_api_calls
    
    async def run(
        self,    
        modified_knowledge: List[dict],
        vs: VectorStore,
        elasticsearch_manager: Optional[ElasticSearchManager] = None
        ) -> None:

        # Update vector store
        new_documents = []
        ids = []
        print(type(modified_knowledge))
        print(type(modified_knowledge[0]))
        for doc in modified_knowledge:
            ids.append(doc["ind"])
            new_documents.append(doc["content_after"])
            vs.json_documents[int(doc["ind"])]["content"] = doc["content_after"]
        assert os.path.exists(vs.index_path)
        vs.index.remove_ids(np.array(ids))
        new_embeddings = await vs.extract_embeddings(documents=new_documents, allowed_api_calls=self.allowed_api_calls)
        vs.index.add_with_ids(np.array(new_embeddings), ids)
        #faiss.write_index(vs.index, vs.index_path)
        # Update elasticsearch index if elasticsearch was used
        if elasticsearch_manager is not None:
            documents_to_be_removed = [
                {
                    "ind": doc["ind"],
                    "content": doc["content_before"],
                    "url": vs.json_documents[int(doc["ind"])]["url"]
                }
                for doc in modified_knowledge
            ]
            documents_to_be_inserted = [
                {
                    "ind": doc["ind"],
                    "content": doc["content_after"],
                    "url": vs.json_documents[int(doc["ind"])]["url"]
                }
                for doc in modified_knowledge
            ]

            elasticsearch_manager.remove_from_index(documents=documents_to_be_removed)
            elasticsearch_manager.add_to_index(documents=documents_to_be_inserted)
        print(f"The vector store has been updated successfully!")


class DisplayDiffTool:
    @staticmethod
    def run(modified_knowledge: List[dict], group: Literal['check', 'update']) -> None:
        console = Console()
        console.rule(f"[bold red]CHANGES FROM THE {group.upper()} GROUP[/bold red]")

        for doc in modified_knowledge:
            doc_id, doc_before, doc_after = doc["ind"], doc["content_before"], doc["content_after"]
            # Print the document id
            console.print(f"[bold yellow]DOC ID:[/bold yellow] {doc_id}")
            # Generate a unified diff between the before and after content.
            diff_lines = list(difflib.unified_diff(
                doc_before.splitlines(),
                doc_after.splitlines(),
                fromfile="Before",
                tofile="After",
                lineterm=""
            ))
            diff_text = "\n".join(diff_lines)
            if not diff_text:
                diff_text = "No changes detected."
            
            syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=True)
            panel = Panel(syntax, title="Diff", border_style="green")
            console.print(panel)
            console.print("\n")

        

