import requests

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from typing import Any, Optional, List, Iterable
from langchain_community.vectorstores import OpenSearchVectorSearch


class HybridOpenSearch(OpenSearchVectorSearch):
    def __init__(self,
                 opensearch_url: str,
                 index_name: str,
                 embedding_function: Embeddings,
                 **kwargs: Any):
        super().__init__(opensearch_url=opensearch_url,
                         index_name=index_name,
                         embedding_function=embedding_function,
                         **kwargs)
        self.search_pipeline_name = 'test-pipeline'
        self.opensearch_url = opensearch_url if opensearch_url.endswith('/') else opensearch_url + '/'
        self.index = None
        response = requests.get(f'{self.opensearch_url}_ingest/pipeline/')
        pipelines = response.json()

        if self.search_pipeline_name not in pipelines:
            request = {
                'description': 'Hybrid search pipeline',
                'processors': []
            }
            requests.put(f'{self.opensearch_url}_ingest/pipeline/{self.search_pipeline_name}', json=request)
            request = {
                'description': "Post processor for hybrid search",
                'phase_results_processors': [
                    {
                        'normalization-processor': {
                            'normalization': {
                                'technique': 'min_max'
                            },
                            'combination': {
                                'technique': 'arithmetic_mean',
                                'parameters': {
                                    'weights': [0.8, 0.2]
                                }
                            }
                        }
                    }
                ]
            }
            requests.put(f'{self.opensearch_url}_search/pipeline/{self.search_pipeline_name}', json=request)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        bulk_size: int = 500,
        **kwargs: Any,
    ) -> List[str]:
        lower_texts = []

        for text in texts:
            lower_texts.append(text.lower())

        embeddings = self.embedding_function.embed_documents(list(lower_texts))
        return self._OpenSearchVectorSearch__add(
            texts,
            embeddings,
            metadatas=metadatas,
            ids=ids,
            bulk_size=bulk_size,
            **kwargs,
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any
    ) -> List[Document]:
        chunk_min_characters_count = 10
        query_embeddings = self.embeddings.embed_documents([query])
        request = {
            'size': k,
            'query': {
                'hybrid': {
                    'queries': [
                        {
                            'multi_match': {
                                'query': query,
                                'fields': 'metadata.description'
                            }
                        },
                        {
                            'script_score': {
                                'query': {"bool": {}},
                                'script': {
                                    'source': 'knn_score',
                                    'lang': 'knn',
                                    'params': {
                                        'field': 'vector_field',
                                        'query_value': query_embeddings[0],
                                        'space_type': kwargs.get('space_type')
                                    }
                                }
                            }
                        }
                    ]
                }
            }
        }

        response = requests.get(
            f'{self.opensearch_url}{self.index}/_search?search_pipeline={self.search_pipeline_name}',
            json=request)
        search_results = response.json()['hits']['hits']
        chunks = []

        for chunk in search_results:
            if len(chunk['_source']['text']) >= chunk_min_characters_count:
                metadata = chunk['_source']['metadata']
                metadata['score'] = chunk['_score']
                chunks.append(Document(
                    page_content=chunk['_source']['text'],
                    metadata=metadata
                ))

        return chunks
