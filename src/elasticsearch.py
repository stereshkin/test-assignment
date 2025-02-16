from typing import List
from elasticsearch import Elasticsearch

class ElasticSearchManager:
    def __init__(self, elastic_instance: Elasticsearch, index: str) -> None:
        self.elastic_instance = elastic_instance
        self.index = index
        
    def add_to_index(self, documents: List[dict]) -> None:
        for doc in documents:
            self.elastic_instance.index(index=self.index, id=doc["_id"], document=doc)

    def remove_from_index(self, documents: List[dict]) -> None:
        for doc in documents:
            self.elastic_instance.delete(index=self.index, id=doc["_id"])
    
    def query_index(self, query: str, elastic_score_threshold: float = 0.2, scale_factor: float = 0.5) -> dict:
            
        search_body = {
            "query": {
                "script_score": {
                    "query": {
                        "match": { "content": query }
                        },
                    "script": {
                        "source": "1 / (1 + Math.exp(-params.scale * _score))",
                        "params": { "scale": scale_factor }
                    }
                }
            },
            "min_score": elastic_score_threshold
        }

        response = self.elastic_instance.search(index=self.index, body=search_body)

        elastic_scores = {}
        for hit in response["hits"]["hits"]:
            elastic_scores[hit["_id"]] = hit["_score"]

        return elastic_scores