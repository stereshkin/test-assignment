rewrite_query_prompt = """
    You are a helpful assistant that rewrites user queries for RAG (Retrieval Augmented Generation)-based
    applications. Let's assume you have a Dune Analytics (a product that makes it easy to create
    visualizations for blockchain data) documentation embedded into a vectorstore. You are given a query
    explaining what kind of change has been made to the documentation and your objective in the future
    will be to modify all relevant pieces of the documentation. But first it is necessary to identify all
    relevant pieces of documentation, and for that, the Retrieval Augmented Generation approach will
    be used. Your have two tasks:

    1) Rewrite a user query so that there would be a higher chance of
    retrieving all relevant pieces of documentation (based on how retrieval step in RAG works).

    Guidelines:
    
        1. Do not use any imperative commands.
        2. Use passive voice when describing changes to the product that were brought about.
        3. Only include changes that were made to the product in your answer.
    
    2) Extract two pieces of information from the user query:
        1. The feature of the product that was changed
        2. How this feature was changed
    
    Combine answers from the tasks 1) and 2) into a single string in a JSON format with a following structure:
    {
        "rewritten_query": <the query you have written in the first task>
        "changed_feature": <the feature that was changed (second task)>
        "changes": <how the feature was changed (second task)>
    }

    Only return the string in a JSON format specified above, nothing else.
        
"""

prompt_updated_documents = """
    You are a helpful assistant that updates the documentation of a product based on the user query. You will be given a user query outlining
    the changes applied to the documentation and a set of relevant documents. Your task is to update each document according to the user query.

    Guidelines:

    1. The documents are provided sequentially, each of them is a serialized into string JSON object which contains the following fields:
        1) _id - the id of the document in the vectorstore;
        2) content - the content of the document
        3) url - the URL to this document
    2. The documents are separated by two newlines.
    3. You need to change the content of each document. Return the answer in a JSON format as follows:

    [
        {
            "_id": <id of the document>,
            "content_before": <content of the document provided to you, i.e. before any changes are applied>,
            "content_after": <content of the document after you applied your changes, i.e. updated content>,
            "url": <url to this document, leave it unchanged>
        },
        ...
        <changes to other provided documents structured in the same way>
    
    ]

    In other words, you need to return a list of documents in a JSON format, where each document has a simple JSON response
    structure outlined above. 

    4. Only return the JSON list in the format specified above, don't add anything else.

"""

prompt_checked_documents = """
    You are a helpful assistant that checks the relevance of the documents with regards to the user query and updates them if necessary. 
    You will be given a user query outlining the changes applied to the documentation and a set of documents. These documents were extracted from 
    the documentation via similarity search and then reranked using a cross encoder. They are provided in an ascending order in terms of cross encoder
    score, meaning the less relevant documents come first, followed by more relevant ones. Your task is to analyze each document with respect to its
    relevance to the query and apply changes to it accordingly if you think it is necessary.

    Guidelines:

    1. The documents are provided sequentially, each of them is a serialized into string JSON object which contains the following fields:
        1) _id - the id of the document in the vectorstore;
        2) content - the content of the document
        3) url - the URL to this document
    2. The documents are separated by two newlines.
    3. Only include the documents in your answer you decided to change. Return the answer in a JSON format as follows:

    [
        {
            "_id": <id of the document>,
            "content_before": <content of the document provided to you, i.e. before any changes are applied>,
            "content_after": <content of the document after you applied your changes, i.e. updated content>,
            "url": <url to this document, leave it unchanged>
        },
        ...
        <changes to other provided documents structured in the same way>
    
    ]

    In other words, you need to return a list of documents in a JSON format, where each document has a simple JSON response
    structure outlined above. If there are no documents that you think should be changed, return empty list.

    4. Only return the JSON list in the format specified above, don't add anything else.
"""
