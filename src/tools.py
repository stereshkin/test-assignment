import json
import os
from vectorstore import VectorStore
import numpy as np
import faiss
from typing import Literal


class UpdateKnowledgeTool:
    @staticmethod
    async def run(model_response: str, vs: VectorStore, group: Literal['check', 'update']) -> None:
        modified_knowledge = json.loads(model_response)
        new_documents = []
        ids = []
        for doc in modified_knowledge:
            ids.append(doc["_id"])
            new_documents.append(doc["content_after"])
            vs.json_documents[int(doc["_id"])] = doc["content_after"]
        assert os.path.exists(vs.index_path)
        vs.index.remove_ids(np.array(ids))
        new_embeddings = await vs.extract_embeddings(documents=new_documents)
        vs.index.add_with_ids(new_embeddings, ids)
        faiss.write_index(vs.index, vs.index_path)
        print(f"The vector store has been updated successfully from the {group} group!")


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
        

