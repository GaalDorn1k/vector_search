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
                 search: dict,
                 **kwargs: Any):
        super().__init__(opensearch_url=opensearch_url,
                         index_name=index_name,
                         embedding_function=embedding_function,
                         **kwargs)
        self.search_config = search
        self.search_pipeline_name = 'hybrid_pipeline_0'
        self.opensearch_url = opensearch_url if opensearch_url.endswith('/') else opensearch_url + '/'
        self.index = index_name
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
                                    'weights': self.search_config['weights']
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
            ids=[str(metadatas[0]['option_id'])],
            bulk_size=bulk_size,
            **kwargs,
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any
    ) -> List[Document]:
        query_embeddings = self.embeddings.embed_documents([query])
        request = {
            'size': self.search_config.get('k'),
            'query': {
                'hybrid': {
                    'queries': [
                        {
                            'multi_match': {
                                'query': query,
                                'fields': 'metadata.option_all_text'
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
                                        'space_type': self.search_config.get('space_type')
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

        return search_results
