from openai import OpenAI
from dotenv import load_dotenv
import os
from src.vectorstore import VectorStore
import faiss
from sentence_transformers.cross_encoder import CrossEncoder
import bisect
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Literal, Optional, Union
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
from src.models import DocumentationChangeResponse, UpdatedDocumentsResponse
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from rich.live import Live
from collections.abc import Callable


basedir = os.path.abspath(os.path.dirname(__file__))
ENV_PATH = os.path.join(os.path.join(basedir, '..'), '.env')
load_dotenv(ENV_PATH)

class Agent:
    def __init__(
            self,
            dim: int,
            llm_model_name: str,
            embedding_model_name: str,
            use_cross_encoder_reranking: bool,
            use_elastic_search: bool,
            update_api_calls: int = 8,
            max_api_calls: int = 50,
            weight_elastic: Optional[float] = None,
            reranker_model_name: Optional[str] = None,
            *args,
            **kwargs,
        ):
        self.llm = llm_model_name
        self.max_api_calls = max_api_calls
        self.update_api_calls = update_api_calls
        self.rewrite_query_model = DocumentationChangeResponse
        self.updated_documents_model = UpdatedDocumentsResponse
        self.use_cross_encoder_reranking = use_cross_encoder_reranking
        self.use_elastic_search = use_elastic_search
        self.client = OpenAI()
        self.vs = VectorStore(
            dim,
            embedding_model_name,
            *args,
            **kwargs
        )
        self.documents_changes = {}
        self.final_output_tree = None
        if self.use_cross_encoder_reranking:
            if reranker_model_name is None:
                raise RuntimeError("You must specify cross encoder model name if you want to use cross encoder reranking.")
            self.reranker = CrossEncoder(reranker_model_name, default_activation_function=nn.Sigmoid())

        if self.use_elastic_search:
            if weight_elastic is None:
                raise RuntimeError("You must specify weight value for elastic search scores if you want to use elastic search. \
                                  This value should be between 0 and 1.")
            
            self.weight_elastic = weight_elastic
            self.log_callback = None
            self.elastic_index_name = "documents"
            self.elasticsearch = ElasticSearchManager(elastic_instance=Elasticsearch(os.environ.get('ELASTICSEARCH_URL')), index=self.elastic_index_name)
            self.elasticsearch.add_to_index(documents=self.vs.json_documents)

    def set_log_callback(self, callback: Callable):
        self.log_callback = callback
    
    def log(self, message: Union[Panel, str]):
        # If a callback is set, use it to log messages.
        if self.log_callback:
            self.log_callback(message)
        else:
            raise RuntimeError("Callback function is not set.")
        
    def print_tree(self):
        console = Console()
        console.print(self.final_output_tree)


    def _calculate_number_of_tokens(self, text: str) -> int:
        encoding = tiktoken.encoding_for_model(self.llm)
        tokens = encoding.encode(text)
        return len(tokens)
                    

    def rewrite_and_analyze_user_query(self, query: str) -> Tuple[str, str]:
        completion = self.client.beta.chat.completions.parse(
            model=self.llm,
            messages=[
                {"role": "developer", "content": rewrite_query_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.2,
            response_format=self.rewrite_query_model
        )
        self.max_api_calls -= 1

        response = json.loads(completion.choices[0].message.content)
        feature_changed, changes = response["changed_feature"], response["changes"]
        return feature_changed, changes
    
    async def extract_relevant_documents(self,
                                         query: str,
                                         feature_changed: str,
                                         cosine_min: float = 0.1,
                                         score_min: float = 0.2,
                                         score_confident: float = 0.7,
                                         batch_size=100
                                         ) -> Tuple[List[dict], List[dict]]:
        # embed query vector
        response = self.client.embeddings.create(input=[query], model=self.vs.embedding_model)
        self.max_api_calls -= 1
        query_embedding = np.array(response.data[0].embedding).reshape((1, -1)).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        # populate vector store
        await self.vs.populate_vector_store()
        # retrieve all documents within radius
        threshold = cosine_min
        lims, distances, indices = self.vs.index.range_search(query_embedding, threshold)

        retrieved_ids = set()
        min_cossim_score = float("inf")
        max_cossim_score = float("-inf")
        retrieved_docs_and_scores = []
        for i in range(lims[0], lims[1]):
            vector_id = indices[i]
            distance = distances[i]
            min_cossim_score = float(min(min_cossim_score, distance))
            max_cossim_score = float(max(max_cossim_score, distance))
            # double-check the threshold here
            if cosine_min <= distances[i] <= 1.0:
                retrieved_ids.add(vector_id)
                retrieved_docs_and_scores.append((vector_id, distance))

        log_message = (f"Extracted [bold]{len(retrieved_ids)}[/bold] documents in total"
                       f" with range search. Scores are ranging"
                       f" from [bold]{round(min_cossim_score, 2)}[/bold] to [bold]{round(max_cossim_score, 2)}[/bold].")
        self.log(Panel(log_message, style="green"))

        retrieved_documents = [(i, x["content"]) for i, x in enumerate(self.vs.json_documents) if i in retrieved_ids]

        if self.use_cross_encoder_reranking or self.use_elastic_search:    
            if self.use_cross_encoder_reranking:
            # Reranking retrieved documents with a cross encoder
                reranker_results = []
                for i in range(0, len(retrieved_documents), batch_size):
                    batch_documents = retrieved_documents[i:i+batch_size]
                    reranker_scores = self.reranker.predict([[query, doc[1]] for doc in batch_documents])
                    indices = [doc[0] for doc in retrieved_documents]
                    reranker_results.extend(list(zip(indices, reranker_scores)))

                retrieved_docs_and_scores = reranker_results

            if self.use_elastic_search:
            # Fusing reranker/faiss similarity scores with elastic scores
                elastic_scores = self.elasticsearch.query_index(query=feature_changed)
                fused_scores = defaultdict(float)
                semantic_scores = {_id: score for _id, score in retrieved_docs_and_scores}
                for _id in list(elastic_scores.keys() | semantic_scores.keys()):
                    fused_scores[_id] = elastic_scores.get(_id, 0.) * self.weight_elastic + float(semantic_scores.get(_id, 0.)) * (1 - self.weight_elastic)

                retrieved_docs_and_scores = list(fused_scores.items())

        retrieved_docs_and_scores.sort(key=lambda x: x[1])
        scores = [score for ind, score in retrieved_docs_and_scores]
        results = retrieved_docs_and_scores
        
        reranker_largest_scores = [str(scores[0]), str(scores[-1])]
        reranker_largest_scores_items = ', '.join(reranker_largest_scores)

        if self.use_cross_encoder_reranking or self.use_elastic_search:    
            log_rerank_message = (f"Retrieved documents have been reranked: scores are"
                                  f" in the range: [bold][{reranker_largest_scores_items}][/bold]")
            self.log(Panel(log_rerank_message, style="green"))
        # find the documents with cross encoder/fused scores between 0.1 and 0.5
        # those are sent to be checked with llm
        # the documents with scores above 0.5 are sent to be updated right away
        cutoff_index = bisect.bisect_left(scores, score_min)
        index_confident = bisect.bisect_left(scores, score_confident)

        check_documents_ids = {x[0] for x in results[cutoff_index:index_confident]}
        update_documents_ids = {x[0] for x in results[index_confident:]}
        
        documents_to_be_checked = [x for i, x in enumerate(self.vs.json_documents) if i in check_documents_ids]
        documents_to_be_updated = [x for i, x in enumerate(self.vs.json_documents) if i in update_documents_ids]
        return documents_to_be_checked, documents_to_be_updated
    

    def _prepare_token_num_list(self,
                                system_prompt: str,
                                query: str,
                                documents: List[dict]) -> Tuple[List[int], int]:
        """
        Helper method to form a list with number of tokens for each additional document in the prompt.
        """
        user_prompt = query + f'\n\nHere are the documents: \n'
        prompt_model_json_schema = json.dumps(self.updated_documents_model.model_json_schema(), indent=2)
        prompt_num_tokens = self._calculate_number_of_tokens(system_prompt) + \
              self._calculate_number_of_tokens(user_prompt) + self._calculate_number_of_tokens(prompt_model_json_schema)
        documents.sort(key=lambda x: self._calculate_number_of_tokens(json.dumps(x)))
        tokens_num_list = [self._calculate_number_of_tokens(json.dumps(documents[0])) + prompt_num_tokens]
        tokens_num_list.extend(
            list(map(lambda x: self._calculate_number_of_tokens('\n\n' + json.dumps(x)), documents[1:]))
        )
        return tokens_num_list, prompt_num_tokens
    
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _call_llm(self, messages: List[dict]) -> str:
        completion = self.client.beta.chat.completions.parse(
            model=self.llm,
            messages=messages,
            temperature=0.2,
            response_format=self.updated_documents_model
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
            query += f'\n\nHere are the documents: \n' + '\n\n'.join(documents_selected)
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
        
        if len(documents_to_be_updated) > 0:
            # Create lists with token counts for both groups
            tokens_updated_group, prompt_updated_token_count = self._prepare_token_num_list(
                system_prompt=prompt_updated_documents,
                query=query,
                documents=documents_to_be_updated
            )
        else:
            tokens_updated_group, prompt_updated_token_count = None, None

        if len(documents_to_be_checked) > 0:
            tokens_checked_group, prompt_checked_token_count = self._prepare_token_num_list(
                system_prompt=prompt_checked_documents,
                query=query,
                documents=documents_to_be_checked
            )
        else:
            tokens_checked_group, prompt_checked_token_count = None, None
        # Find partitions for both groups
        partition_checked_group, partition_updated_group = find_partitions(
            max_partitions=self.max_api_calls - self.update_api_calls,
            tokens_updated_group=tokens_updated_group,
            tokens_checked_group=tokens_checked_group,
            prompt_updated_token_count=prompt_updated_token_count,
            prompt_checked_token_count=prompt_checked_token_count
        )
        # Create list of messages for both groups
        if partition_updated_group is not None:
            messages_updated_group = Agent._create_messages_from_partitions(
                system_prompt=prompt_updated_documents,
                partition=partition_updated_group,
                documents=documents_to_be_updated,
                query=query
            )
            updated_documents_response = await self._process_messages(messages_updated_group, task='update')
            updated_documents_data = [json.loads(data)['updated_documents'] for data in updated_documents_response]
            updated_documents_flattened = []
            for data in updated_documents_data:
                updated_documents_flattened.extend(data)
        else:
            updated_documents_flattened = []

        if partition_checked_group is not None:
            messages_checked_group = Agent._create_messages_from_partitions(
                system_prompt=prompt_checked_documents,
                partition=partition_checked_group,
                documents=documents_to_be_checked,
                query=query
            )
        
            checked_documents_response = await self._process_messages(messages_checked_group, task='check')
            checked_documents_data = [json.loads(data)['updated_documents'] for data in checked_documents_response]
            checked_documents_flattened = []
            for data in checked_documents_data:
                checked_documents_flattened.extend(data)
        else:
            checked_documents_flattened = []
        # Update knowledge
        modified_knowledge = [*updated_documents_flattened, *checked_documents_flattened]
        if len(modified_knowledge) == 0:
            self.log(Panel("[bold red]No documents were changed by LLM.[/bold red]", style="red"))
            return

        kwargs_update = {
            "modified_knowledge": modified_knowledge,
            "vs": self.vs
        }

        if self.use_elastic_search:
            kwargs_update["elasticsearch_manager"] = self.elasticsearch

        await UpdateKnowledgeTool(allowed_api_calls=self.update_api_calls).run(**kwargs_update)
        self.log(Panel("[bold green]The documentation was updated![/bold green]", style="green"))
        
        # Save changed documents
        if len(updated_documents_flattened) > 0:
            self.documents_changes['update'] = updated_documents_flattened

        if len(checked_documents_flattened) > 0:
            self.documents_changes['check'] = checked_documents_flattened


    def display_changes(self, group: Literal['check', 'update']) -> None:
        if group not in ['check', 'update']:
            raise ValueError("You must specify either 'check' or 'update' group.")

        console = Console()
        if group in self.documents_changes:
            DisplayDiffTool.run(self.documents_changes[group], group=group)
        else:
            console.print(Panel(f"[bold red]No documents from the group {group} have been changed.[/bold red]",
                                style="red"))
            
    # Method that runs the whole pipeline
    async def run(self, query: str):
        console = Console()
        # Create a tree with a root node representing the entire pipeline.
        tree = Tree("[bold cyan]Pipeline Execution[/bold cyan]", guide_style="bold bright_blue")

        def update_node(node: Tree, message: Union[Panel, str]):
            node.add(message)

        with Live(tree, refresh_per_second=4, console=console):
            # Step 1: Query Analysis and Rewriting
            step1 = tree.add("[bold cyan]Step 1: Query Analysis and Rewriting[/bold cyan]")
            self.set_log_callback(lambda message: update_node(step1, message))   # set callback function
            feature_changed, changes = self.rewrite_and_analyze_user_query(query)
            rewritten_user_query = f"Feature {feature_changed} was changed: {changes}"
            
            self.log(Panel(f"[bold]Initial user query:[/bold]\n{query}", style="cyan"))
            self.log(Panel(f"[bold]Feature changed:[/bold]\n{feature_changed}", style="magenta"))
            self.log(Panel(f"[bold]Identified changes:[/bold]\n{changes}", style="green"))
            self.log(Panel(f"[bold]Rewritten query:[/bold]\n{rewritten_user_query}", style="blue"))

            # Step 2: Extraction of Relevant Documents
            step2 = tree.add("[bold yellow]Step 2: Extraction of Relevant Documents[/bold yellow]")
            self.set_log_callback(lambda message: update_node(step2, message))   # update callback function, now nodes are added to step2 subtree via self.log()
            documents_to_be_checked, documents_to_be_updated = await self.extract_relevant_documents(
                query=rewritten_user_query, feature_changed=feature_changed
            )

            if len(documents_to_be_checked) == 0 and len(documents_to_be_updated) == 0:
                self.log(Panel("[bold red]No documents that require update or check have been found.[/bold red]",
                                style="red"))
                
                self.final_output_tree = tree
                return
            else:
                self.log(Panel(f"[bold]Documents to be checked:[/bold] {len(documents_to_be_checked)}", style="yellow"))
                self.log(Panel(f"[bold]Documents to be updated:[/bold] {len(documents_to_be_updated)}", style="yellow"))

            # Step 3: Update of Documents
            step3 = tree.add("[bold green]Step 3: Updating/Checking Documents[/bold green]")
            self.set_log_callback(lambda message: update_node(step3, message))  # update callback function
            self.log(Panel("[bold]Proceeding to check/update documents...[/bold]", style="green"))
            
            await self.update_documents(
                documents_to_be_checked=documents_to_be_checked,
                documents_to_be_updated=documents_to_be_updated,
                query=rewritten_user_query
            )
        
        # Save the entire tree structure
        self.final_output_tree = tree
