import os
import openai
from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

openai.api_key = os.getenv('OPENAI_API_KEY')

Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

def build_automerging_index(
    documents,
    save_dir="merging_index",
    chunk_sizes=None,
):
    """
    Build an automerging index from the given documents.

    Parameters:
        documents (list): List of documents to build the index from.
        save_dir (str): Directory to save the index. Defaults to "merging_index".
        chunk_sizes (list): List of chunk sizes for hierarchical node parsing. Defaults to None.

    Returns:
        None
    """
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    # Build the vector store index
    automerging_index = VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
        llm=Settings.llm,
        embed_model=Settings.embed_model,
    )
    # Persist the index
    automerging_index.storage_context.persist(persist_dir=save_dir)
    print("Index built and saved.")

def get_automerging_query_engine(
    automerging_index,
    similarity_top_k=12,
    rerank_top_n=6,
):
    """
    Get an automerging query engine.

    Parameters:
        automerging_index: Automerging index to use for retrieval.
        similarity_top_k (int): Number of similar documents to retrieve initially. Defaults to 12.
        rerank_top_n (int): Number of documents to rerank. Defaults to 6.

    Returns:
        RetrieverQueryEngine: Automerging query engine.
    """
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank]
    )
    return auto_merging_engine

def get_automerging_index(llm, save_dir, embed_model):
    """
    Load an automerging index from storage.

    Parameters:
        llm: OpenAI model instance.
        save_dir (str): Directory where the index is saved.
        embed_model: OpenAIEmbedding model instance.

    Returns:
        VectorStoreIndex: Automerging index loaded from storage.
    """
    # Load index from storage
    automerging_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=save_dir),
        llm=llm,
        embed_model=embed_model,
    )
    return automerging_index

def get_query_engine():
    """
    Get an automerging query engine.

    Returns:
        RetrieverQueryEngine: Automerging query engine.
    """
    print("Getting the query engine...")
    # Get automerging index
    index = get_automerging_index(
        llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
        save_dir="./merging_index",
        embed_model=OpenAIEmbedding(model="text-embedding-3-large"),
    )
    # Get automerging query engine
    query_engine = get_automerging_query_engine(index, similarity_top_k=6)
    print("Query engine ready.")
    return query_engine

def chat(query, query_engine):
    """
    Get response for a given query using the provided query engine.

    Parameters:
        query (str): Query string.
        query_engine: Query engine to use for retrieving response.

    Returns:
        str: Response to the query.
    """
    response = query_engine.query(query)
    return str(response)
