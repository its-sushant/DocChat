import os
import io
import tempfile
from pathlib import Path

import openai
import pandas as pd
from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.postprocessors.rankgpt_rerank import RankGPTRerank
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

openai.api_key = os.getenv("OPENAI_API_KEY")

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")


def _load_excel(path: str, filename: str) -> list[Document]:
    xls = pd.ExcelFile(path)
    docs = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        text = f"File: {filename} | Sheet: {sheet}\n\n{df.to_string(index=False)}"
        docs.append(Document(text=text, metadata={"source": filename, "sheet": sheet}))
    return docs


def _load_csv(path: str, filename: str) -> list[Document]:
    df = pd.read_csv(path)
    text = f"File: {filename}\n\n{df.to_string(index=False)}"
    return [Document(text=text, metadata={"source": filename})]


def load_documents_from_uploaded_files(uploaded_files) -> list[Document]:
    """Load documents from Streamlit UploadedFile objects (PDF, DOCX, XLSX, CSV)."""
    documents = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        saved: dict[str, str] = {}
        for uf in uploaded_files:
            dest = Path(tmp_dir) / uf.name
            dest.write_bytes(uf.getbuffer())
            saved[str(dest)] = uf.name

        for path_str, filename in saved.items():
            ext = Path(filename).suffix.lower()
            if ext in (".pdf", ".docx", ".txt"):
                try:
                    from llama_index.readers.file import DocxReader
                    file_extractor = {".docx": DocxReader()}
                except ImportError:
                    file_extractor = {}
                reader = SimpleDirectoryReader(
                    input_files=[path_str],
                    file_extractor=file_extractor,
                )
                docs = reader.load_data()
                for doc in docs:
                    doc.metadata["source"] = filename
                documents.extend(docs)
            elif ext in (".xlsx", ".xls"):
                documents.extend(_load_excel(path_str, filename))
            elif ext == ".csv":
                documents.extend(_load_csv(path_str, filename))

    return documents


def build_automerging_index(
    documents: list[Document],
    save_dir: str = "merging_index",
    chunk_sizes: list[int] | None = None,
) -> VectorStoreIndex:
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)

    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)
    index.storage_context.persist(persist_dir=save_dir)
    return index


def build_query_engine(
    index: VectorStoreIndex,
    similarity_top_k: int = 12,
    rerank_top_n: int = 6,
) -> RetrieverQueryEngine:
    base_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, index.storage_context, verbose=False
    )
    rerank = RankGPTRerank(top_n=rerank_top_n, llm=Settings.llm)
    return RetrieverQueryEngine.from_args(retriever, node_postprocessors=[rerank])


def process_uploaded_files(uploaded_files) -> RetrieverQueryEngine:
    """End-to-end pipeline: uploaded files → query engine."""
    documents = load_documents_from_uploaded_files(uploaded_files)
    if not documents:
        raise ValueError("No text could be extracted from the uploaded files.")
    index = build_automerging_index(documents)
    return build_query_engine(index)


def chat(query: str, query_engine: RetrieverQueryEngine) -> str:
    response = query_engine.query(query)
    return str(response)
