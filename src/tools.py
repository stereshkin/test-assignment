import json
import os
from src.vectorstore import VectorStore
import numpy as np
import faiss
from typing import Literal, Optional, List
from src.elasticsearch import ElasticSearchManager


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
        for doc in modified_knowledge:
            ids.append(doc["_id"])
            new_documents.append(doc["content_after"])
            vs.json_documents[int(doc["_id"])] = doc["content_after"]
        assert os.path.exists(vs.index_path)
        vs.index.remove_ids(np.array(ids))
        new_embeddings = await vs.extract_embeddings(documents=new_documents, allowed_api_calls=self.allowed_api_calls)
        vs.index.add_with_ids(np.array(new_embeddings), ids)
        faiss.write_index(vs.index, vs.index_path)
        # Update elasticsearch index if elasticsearch was used
        if elasticsearch_manager is not None:
            documents_to_be_removed = [
                {
                    "_id": doc["_id"],
                    "content": doc["content_before"],
                    "url": doc["url"]
                }
                for doc in modified_knowledge
            ]
            documents_to_be_inserted = [
                {
                    "_id": doc["_id"],
                    "content": doc["content_after"],
                    "url": doc["url"]
                }
                for doc in modified_knowledge
            ]

            elasticsearch_manager.remove_from_index(documents=documents_to_be_removed)
            elasticsearch_manager.add_to_index(documents=documents_to_be_inserted)
        print(f"The vector store has been updated successfully!")


class DisplayDiffTool:
    @staticmethod
    def run(model_response: str, group: Literal['check', 'update']) -> None:
        print(f"CHANGES FROM THE {group.upper()} GROUP: \n\n")
        modified_knowledge = json.loads(model_response)
        for doc in modified_knowledge:
            doc_id, doc_before, doc_after = doc["_id"], doc["content_before"], doc["content_after"]
            print(f"DOC ID: {doc_id}")
            print(f"Before: {doc_before}")
            print(f"After: {doc_after}")
            print('=' * 30)
        

