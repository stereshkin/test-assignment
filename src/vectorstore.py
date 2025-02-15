import faiss
from faiss import IndexIDMap2, IndexFlatL2
from openai import OpenAI
from typing import List
from dotenv import load_dotenv
import os
import json
import asyncio
from tenacity import retry, wait_random_exponential, stop_after_attempt
import re
import tqdm
import numpy as np


ENV_PATH = os.path.join(os.path.join(__file__, '..'), '.env')
load_dotenv(ENV_PATH)

class VectorStore:
    def __init__(self,
                 dim: int,
                 embedding_model_name: str,
                 knowledge_file_path: str = "../dune_docs.json",
                 index_file_name: str = "documentation.index"
                 ):
        
        self.index = IndexIDMap2(IndexFlatL2(dim))
        self.index_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            '..',
            index_file_name
        ))
        self.knowledge_file_path = knowledge_file_path
        self.client = OpenAI()
        self.embedding_model = embedding_model_name
        self.load_data()

    @staticmethod
    def normalize_embedding(embedding: List[float]) -> List[float]:
        norm = np.linalg.norm(embedding)
        unit_embedding = np.array(embedding) / norm
        return unit_embedding.tolist()
    
    # Synchronous function with retry logic to get an embedding.
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _get_embedding(self, text: str, model="text-embedding-3-small") -> List[float]:
        response = self.client.embeddings.create(input=[text], model=model)
        embedding = response.data[0].embedding
        return VectorStore.normalize_embedding(embedding)
    
    # Asynchronous wrapper that runs the synchronous function in a thread.
    async def _get_embedding_async(self,
                                   text: str,
                                   semaphore: asyncio.Semaphore,
                                   model="text-embedding-3-small"):
        loop = asyncio.get_running_loop()
        async with semaphore:
            return await loop.run_in_executor(None, self._get_embedding, text, model)
    
    # Wrapper function that updates the progress bar after each task completes.
    async def _get_embedding_async_wrapper(self, text: str, model: str, semaphore, pbar):
        result = await self._get_embedding_async(text, semaphore, model=model)
        pbar.update(1)
        return result
    
    # Main asynchronous function to process a list of documents.
    async def extract_embeddings(self, documents: List[str], max_concurrent: int = 5) -> List[List[float]]:
        # Create a list of tasks; asyncio.gather preserves the order of results.
        semaphore = asyncio.Semaphore(max_concurrent)
        with tqdm.tqdm(total=len(documents), desc="Extracting embeddings") as pbar:
            tasks = [
                self._get_embedding_async_wrapper(doc, "text-embedding-3-small", semaphore, pbar)
                for doc in documents
            ]
            embeddings = await asyncio.gather(*tasks)
        return embeddings
    
    # Load and clean data from Unicode characters
    def load_data(self) -> None:
        self.json_documents = []
        with open(self.knowledge_file_path, "r") as f:
            for ind, line in enumerate(f):
                document = json.loads(line)
                self.json_documents.append(
                    {
                        "_id": ind,
                        "content": re.sub(r'[^\x00-\x7F]+', ' ', document.get("content")),  
                        "url": document.get("url")
                    }
                )
    
    # Populate the FAISS vector store
    async def populate_vector_store(self) -> None:
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            ids = []
            documents = []
            for doc in self.json_documents:
                ids.append(doc["_id"])
                documents.append(doc["content"])

            embeddings = await self.extract_embeddings(documents)
            self.index.add_with_ids(embeddings, ids)
            faiss.write_index(self.index, self.index_path)