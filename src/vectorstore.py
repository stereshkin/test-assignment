import faiss
from faiss import IndexIDMap2, IndexFlatIP
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
import tiktoken

ENV_PATH = os.path.join(os.path.join(__file__, '..'), '.env')
load_dotenv(ENV_PATH)

class VectorStore:
    def __init__(self,
                 dim: int,
                 embedding_model_name: str,
                 knowledge_file_path: str = "dune_docs.jsonl",
                 index_file_name: str = "documentation.index"
                 ):
        
        self.index = IndexIDMap2(IndexFlatIP(dim))
        self.index_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            '..',
            index_file_name
        ))
        self.knowledge_file_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            '..',
            knowledge_file_path
        ))
        self.client = OpenAI()
        self.embedding_model = embedding_model_name
        self.load_data()
    

    def _get_token_count(self, text: str) -> int:
        encoding = tiktoken.encoding_for_model(self.embedding_model)
        tokens = encoding.encode(text)
        return len(tokens)


    def _batch_documents(self, documents: List[str], token_limit: int = 7000):
        """
        Partition documents into batches such that each batch's total token count is below token_limit.
        """
        batches = []
        current_batch = []
        current_tokens = 0
        
        for doc in documents:
            tokens = self._get_token_count(doc)
            if tokens > token_limit:
                raise ValueError(f"One document has {tokens} tokens, exceeding the token limit of {token_limit}.")
            # If adding the document stays under the limit, add it; otherwise, start a new batch.
            if current_tokens + tokens <= token_limit:
                current_batch.append(doc)
                current_tokens += tokens
            else:
                batches.append(current_batch)
                current_batch = [doc]
                current_tokens = tokens

        if current_batch:
            batches.append(current_batch)
        return batches

    
    # Synchronous function with retry logic to get an embedding.
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _get_embeddings_batch(self, texts: List[str], model: str) -> List[List[float]]:
        response = self.client.embeddings.create(input=texts, model=model)
        embeddings = [np.array(item.embedding) for item in response.data]
        for emb in embeddings:
            faiss.normalize_L2(emb)

        return embeddings
    
    # Asynchronous wrapper that runs the synchronous function in a thread.
    async def _get_embeddings_batch_async(self,
                                   texts: List[str],
                                   semaphore: asyncio.Semaphore,
                                   model: str):
        loop = asyncio.get_running_loop()
        async with semaphore:
            return await loop.run_in_executor(None, self._get_embeddings_batch, texts, model)
    
    # Wrapper function that updates the progress bar after each task completes.
    async def _get_embeddings_batch_async_wrapper(self, texts: List[str], model: str, semaphore: asyncio.Semaphore, pbar) -> List[List[float]]:
        result = await self._get_embeddings_batch_async(texts, semaphore, model=model)
        pbar.update(1)
        return result
    
    # Main asynchronous function to process the list of documents.
    async def extract_embeddings(self, documents: List[str], max_concurrent: int = 5, token_limit: int = 7000) -> list:
        """
        Extract embeddings for all documents in batches that adhere to the token limit.
        Batches are processed asynchronously with a concurrency limit.
        """
        batches = self._batch_documents(documents, token_limit=token_limit)
        semaphore = asyncio.Semaphore(max_concurrent)
        all_embeddings = []
        
        with tqdm.tqdm(total=len(batches), desc="Extracting embeddings (batches)") as pbar:
            tasks = [
                self._get_embeddings_batch_async_wrapper(batch, self.embedding_model, semaphore, pbar)
                for batch in batches
            ]
            # asyncio.gather preserves the order of batches.
            batch_results = await asyncio.gather(*tasks)
            for embeddings in batch_results:
                all_embeddings.extend(embeddings)
        return all_embeddings
    
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
            self.index.add_with_ids(np.array(embeddings), ids)
            faiss.write_index(self.index, self.index_path)