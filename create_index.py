from langchain_core.embeddings import Embeddings
from langchain_core.documents.base import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embeddings(pipeline: dict) -> Embeddings:
    return HuggingFaceEmbeddings(
            model_name=pipeline['embeddings']['hf_name'],
            **pipeline['embeddings']['args']
        )


def get_vector_store(pipeline: dict, index_name: str = None, documents: list[Document] = None) -> VectorStore:
    embeddings = get_embeddings(pipeline)
    vector_store_class: VectorStore = globals()[pipeline['vector_store']['class_name']]
    if not documents:
        return vector_store_class(embedding_function=embeddings, index_name=index_name, **pipeline['vector_store'][
            'args'])
    elif documents:
        return vector_store_class.from_documents(documents, embeddings, index_name=index_name, **pipeline[
            'vector_store']['args'])


class Indexer:
    def __init__(self, pipeline: dict):
        self.pipeline = pipeline

    def create_index(self, docs_path: str, index_name: str) -> None:
        db = get_vector_store(self.pipeline, index_name)

        # load items
        # while True:
        #     db.add_documents(documents=[])

        db.client.close()
