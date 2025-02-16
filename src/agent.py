from openai import OpenAI
from dotenv import load_dotenv
import os
from src.vectorstore import VectorStore
import faiss
from sentence_transformers.cross_encoder import CrossEncoder
import bisect
import torch.nn as nn
from typing import List, Tuple, Literal, Optional
import tiktoken
from src.binary_search import find_partitions
import json
from src.prompts import rewrite_query_prompt, prompt_updated_documents, prompt_checked_documents
from tenacity import retry, wait_random_exponential, stop_after_attempt
import asyncio
import tqdm
from src.tools import UpdateKnowledgeTool, DisplayDiffTool
from elasticsearch import Elasticsearch
from src.elasticsearch import ElasticSearchManager
from collections import defaultdict


ENV_PATH = os.path.join(os.path.join(__file__, '..'), '.env')
load_dotenv(ENV_PATH)

class Agent:
    def __init__(
            self,
            dim: int,
            llm_model_name: str,
            embedding_model_name: str,
            reranker_model_name: str,
            use_elastic_search: bool,
            update_api_calls: int = 8,
            max_api_calls: int = 50,
            weight_elastic: Optional[float] = None,
            *args,
            **kwargs,
        ):
        self.llm = llm_model_name
        self.max_api_calls = max_api_calls
        self.update_api_calls = update_api_calls
        self.client = OpenAI()
        self.reranker = CrossEncoder(reranker_model_name, default_activation_function=nn.Sigmoid())
        self.vs = VectorStore(
            dim,
            embedding_model_name,
            *args,
            **kwargs
        )
        if use_elastic_search:
            if weight_elastic is None:
                raise ValueError("You must specify weight value for elastic search scores if you want to use elastic search. \
                                  This value should be between 0 and 1.")
            
            self.weight_elastic = weight_elastic
            self.elastic_index_name = "documents"
            self.elasticsearch = ElasticSearchManager(elastic_instance=Elasticsearch(os.environ.get('ELASTICSEARCH_URL')), index=self.elastic_index_name)
            self.elasticsearch.add_to_index(documents=self.vs.json_documents)

    def _calculate_number_of_tokens(self, text: str):
        encoding = tiktoken.encoding_for_model(self.llm)
        tokens = encoding.encode(text)
        return len(tokens)
                    

    def rewrite_and_analyze_user_query(self, query: str) -> Tuple[str, str, str]:
        completion = self.client.chat.completions.create(
            model=self.llm,
            messages=[
                {"role": "developer", "content": rewrite_query_prompt},
                {"role": "user", "content": query}
            ]
        )
        self.max_api_calls -= 1

        response = json.loads(completion.choices[0].message.content)
        rewritten_query, feature_changed, changes = response["rewritten_query"], response["changed_feature"], response["changes"]
        return rewritten_query, feature_changed, changes
    
    async def extract_relevant_documents(self,
                                         query: str,
                                         feature_changed: str,
                                         cosine_min: float = 0.1,
                                         score_min: float = 0.1,
                                         score_confident: float = 0.5,
                                         batch_size=100
                                         ) -> Tuple[List[dict], List[dict]]:
        # embed query vector
        response = self.client.embeddings.create(input=[query], model=self.vs.embedding_model)
        self.max_api_calls -= 1
        query_embedding = response.data[0].embedding
        faiss.normalize_L2(query_embedding)
        # populate vector store
        await self.vs.populate_vector_store()
        # retrieve all documents within radius
        threshold = cosine_min
        lims, distances, indices = self.vs.index.range_search(query_embedding, threshold)

        retrieved_ids = set()
        for i in range(lims[0], lims[1]):
            vector_id = indices[i]
            # double-check the threshold here
            if cosine_min <= distances[i] <= 1.0:
                retrieved_ids.add(vector_id)
        
        # Reranking retrieved documents with a cross encoder
        retrieved_documents = [(i, x["content"]) for i, x in enumerate(self.vs.json_documents) if i in retrieved_ids]
        reranker_results = []
        for i in range(0, len(retrieved_documents), batch_size):
            batch_documents = retrieved_documents[i:i+batch_size]
            reranker_scores = self.reranker.predict([[query, doc[1]] for doc in batch_documents])
            indices = [doc[0] for doc in retrieved_documents]
            reranker_results.extend(list(zip(indices, reranker_scores)))

        reranker_results.sort(key=lambda x: x[1])

        if hasattr(self, "elasticsearch"):
            elastic_scores = self.elasticsearch.query_index(query=feature_changed)
            fused_scores = defaultdict(float, lambda: 0.)
            reranker_scores = {_id: score for _id, score in reranker_results}
            for _id in list(elastic_scores.keys() | reranker_scores.keys()):
                fused_scores[_id] = elastic_scores.get(_id, 0.) * self.weight_elastic + reranker_scores.get(_id, 0.) * (1 - self.weight_elastic)

            scores = list(fused_scores.values()).sort()
            results = list(fused_scores.items()).sort(key=lambda x: x[1])

        else:
            scores = [x[1] for x in reranker_results]
            results = reranker_results
        # find the documents with cross encoder/fused scores between 0.1 and 0.5
        # those are sent to be checked with llm
        # the documents with scores above 0.5 are sent to be updated right away
        cutoff_index = bisect.bisect_left(scores, score_min)
        index_confident = bisect.bisect_left(scores, score_confident)
        check_documents_ids = {x[0] for x in results[cutoff_index:index_confident]}
        update_documents_ids = {x[0] for x in results[index_confident:]}
        documents_to_be_checked = [x for i, x in enumerate(self.vs.json_documents) if i in check_documents_ids]
        documents_to_be_updated = [x for i, x in enumerate(self.vs.json_documents) if i in update_documents_ids]
        print(f"Extracted {len(documents_to_be_checked)} documents to be checked, and {len(documents_to_be_updated)} to be updated.")
        return documents_to_be_checked, documents_to_be_updated
    

    def _prepare_token_num_list(self, system_prompt: str, documents: List[dict]) -> List[int]:
        """
        Helper method to form a list with number of tokens for each additional document in the prompt.
        """
        prompt_num_tokens = self._calculate_number_of_tokens(system_prompt)
        documents.sort(key=lambda x: self._calculate_number_of_tokens(json.dumps(x)))
        tokens_num_list = [self._calculate_number_of_tokens(json.dumps(documents[0])) + prompt_num_tokens]
        tokens_num_list.extend(
            list(map(lambda x: self._calculate_number_of_tokens('\n\n' + json.dumps(x)), documents[1:]))
        )
        return tokens_num_list, prompt_num_tokens
    
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _call_llm(self, messages: List[dict]) -> str:
        completion = self.client.chat.completions.create(
            model=self.llm,
            messages=messages
        )
        return completion.choices[0].message.content
    

    async def _call_llm_async(self, messages: List[dict], semaphore: asyncio.Semaphore):
        loop = asyncio.get_running_loop()
        async with semaphore:
            return await loop.run_in_executor(None, self._call_llm, messages)
        

    async def _call_llm_async_wrapper(self, messages: List[dict], semaphore: asyncio.Semaphore, pbar) -> str:
        response = await self._call_llm_async(messages=messages, semaphore=semaphore)
        pbar.update(1)
        return response
    
    async def _process_messages(self, messages_list: List[List[dict]], task: Literal["update", "check"], max_concurrent: int = 5) -> List[str]:
        semaphore = asyncio.Semaphore(max_concurrent)
        with tqdm.tqdm(total=len(messages_list), desc=f"Processing messages, task: {task}") as pbar:
            tasks = [
                self._call_llm_async_wrapper(messages, semaphore, pbar) for messages in messages_list
            ]
            responses = await asyncio.gather(*tasks)
        return responses


    @staticmethod
    def _create_messages_from_partitions(system_prompt: str,
                                        partition: List[Tuple[int, int]],
                                        documents: List[dict],
                                        query: str
                                        ) -> List[List[dict]]:
        all_messages = []
        for start, end in partition:
            documents_selected = list(map(json.dumps, documents[start:end]))
            query += '\n\nHere are the documents: \n\n' + '\n\n'.join(documents_selected)
            messages_partition = [
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            all_messages.append(messages_partition)

        return all_messages


    async def update_documents(self,
                               documents_to_be_checked: List[dict],
                               documents_to_be_updated: List[dict],
                               query: str
                               ) -> None:
        # Create lists with with token counts for both groups
        tokens_updated_group, prompt_updated_token_count = self._prepare_token_num_list(
            system_prompt=prompt_updated_documents,
            documents=documents_to_be_updated
        )

        tokens_checked_group, prompt_checked_token_count = self._prepare_token_num_list(
            system_prompt=prompt_checked_documents,
            documents=documents_to_be_checked
        )
        # Find partitions for both groups
        partition_checked_group, partition_updated_group = find_partitions(
            max_partitions=self.max_api_calls - self.update_api_calls,
            tokens_updated_group=tokens_updated_group,
            tokens_checked_group=tokens_checked_group,
            prompt_updated_token_count=prompt_updated_token_count,
            prompt_checked_token_count=prompt_checked_token_count
        )
        # Create list of messages for both groups
        messages_updated_group = Agent._create_messages_from_partitions(
            system_prompt=prompt_updated_documents,
            partition=partition_updated_group,
            documents=documents_to_be_updated,
            query=query
        )

        messages_checked_group = Agent._create_messages_from_partitions(
            system_prompt=prompt_checked_documents,
            partition=partition_checked_group,
            documents=documents_to_be_checked,
            query=query
        )
        # Get reponses for all the messages for both groups
        updated_documents_response = await self._process_messages(messages_updated_group, task='update')
        checked_documents_response = await self._process_messages(messages_checked_group, task='check')
        # Update knowledge
        kwargs_update = {
            "model_response": updated_documents_response,
            "vs": self.vs,
            "group": 'update',
        }

        if hasattr(self, "elasticsearch"):
            kwargs_update["elasticsearch_manager"] = self.elasticsearch

        await UpdateKnowledgeTool.run(**kwargs_update)
        kwargs_check = kwargs_update.copy()
        kwargs_check["model_response"] = checked_documents_response
        kwargs_check["group"] = 'check'
        await UpdateKnowledgeTool.run(**kwargs_check)
        # Display diff for changed documents
        print(f"QUERY: {query} \n\n")
        DisplayDiffTool.run(updated_documents_response, group='update')
        DisplayDiffTool.run(checked_documents_response, group='check')

    # Method that runs the whole pipeline
    async def run(self, query):
        rewritten_user_query, feature_changed, changes = self.rewrite_and_analyze_user_query(query)
        print(f"Initial user query: \n{query}\n")
        print(f"Feature changed: \n{feature_changed}\n")
        print(f"Identified changes: \n{changes}\n")
        print(f"Rewritten query: \n{rewritten_user_query}\n")
        documents_to_be_checked, documents_to_be_updated = await self.extract_relevant_documents(query=rewritten_user_query, feature_changed=feature_changed)
        await self.update_documents(
            documents_to_be_checked=documents_to_be_checked,
            documents_to_be_updated=documents_to_be_updated,
            query=rewritten_user_query
            )
