# RAG Pipeline for Documentation Updates

## Overview

The **RAG Pipeline for Documentation Updates** is a Retrieval Augmented Generation (RAG) system that consists of the following components:
- **Analysis** of the user query describing changes to product documentation. In this step, the changed feature and the exact changes to this
feature are extracted from the user query. The query is then rewritten using the extracted information.

- **Retrieval** of the relevant documentation from a FAISS vector store using range search (only documents with cosine similarity above a certain threshold are retrieved).

- **Re-ranking** of the retrieved documents using a cross-encoder and/or Elastic search BM25 scores that are normalized to the range [0, 1]. If Elasticsearch is used, the scores are then fused using a weighted sum and used to re-rank the documents. In this case, Elasticsearch uses the extracted in the step 1 changed feature as a query for a full-text search. Then, the documents above the minimal score threshold (set to 0.2) but below confidence threshold (set to 0.9) are checked and, if necessary, updated by the LLM, whereas documents with a score above confidence threshold are updated by LLM right away.

- **Update** of documentation based on LLM-driven modifications.

- **Display** of intermediate and final results interactively using the [Rich](https://github.com/Textualize/rich) library.


## Architecture

The project is structured into several key components:

- **Agent:**  
  The central class orchestrating the pipeline. It handles query rewriting, document retrieval, re-ranking, and updates, while also managing live visualizations of pipeline execution.

- **UpdateKnowledgeTool:**  
  A tool for updating documents in both the vector store and Elasticsearch index (if Elasticsearch was used).

- **ElasticSearchManager:**  
  A manager class that abstracts interactions with Elasticsearch, including adding and deleting documents.

- **DisplayDiffTool:**  
  A utility to render differences (diffs) between original and updated document contents. This uses the Rich library to produce color-coded, easy-to-read diffs.

- **VectorStore:**  
  Manages document embeddings and integrates with FAISS for efficient vector search. This component ensures that document updates are reflected in the underlying vector index.

- **Message Construction & LLM Calls:**  
  The pipeline partitions all retrieved documents to minimize the amount of tokens the LLM receives in each call while ensuring that constraint of <= 50 API calls per query is satisfied. The pipeline builds messages (with detailed system prompts and JSON schema) to instruct the LLM to process each document independently, ensuring that modifications are correctly applied without carrying over previous changes.


## Possible improvements

- **Human-in-the-loop:**
  It would be beneficial if the model decided if it should ask a human for the approval of the changes it has introduced in case it is not confident that these
  changes are necessary.

- **Embeddings' Finetuning:** 
  It would be beneficial for the retrieval quality to finetune the embeddings on the documentation. Using a simple linear adapter would already yield some gains in terms of retrieval quality.
  
- **Elasticsearch Hybrid Search:**
  Using Elasticsearch Hybrid Search could possibly improve the score quality. There should be a better way to fuse semantic and BM25 scores.
