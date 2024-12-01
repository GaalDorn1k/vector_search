import os
from typing import Any
from vector_search import HybridOpenSearch
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


class SearchService:
    def __init__(self, config: dict):
        embedder = HuggingFaceEmbeddings(
            model_name=config['embeddings']['hf_name'],
            **config['embeddings']['args']
        )
        self.vector_search = HybridOpenSearch(
            opensearch_url=config['vector_store']['args']['opensearch_url'],
            index_name=config['vector_store']['index_name'],
            embedding_function=embedder
        )

    def _dict_convert(self, docs: list[Document]) -> list[dict]:
        for i in range(len(docs)):
            docs[i] = docs[i].model_dump()

    def search(self, query: str) -> list[dict]:
        result = self.vector_search.similarity_search(query)
        return self._dict_convert(result)

    def add_item(self, item: dict) -> None:
        self.vector_search.add_texts([item['description']],
                                     metadatas=[item['metadata']])

    def filter_search(self, query: str, filter: dict) -> list[dict]:
        result = self.search(query)
        key, value = list(filter.items())[0]
        filter_result = []

        for res in result:
            if res['metadatas'][key] == value:
              filter_result.append(res)

        return filter_result   

        
        