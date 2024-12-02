from vector_search import HybridOpenSearch
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
            embedding_function=embedder,
            search=config['search']
        )

    def result_convert(self, result: list[dict]) -> list[dict]:
        result_new = []
        for item in result:
            result_new.append(item['_source']['metadata'])
        return result_new

    def search(self, query: str) -> list[dict]:
        result = self.vector_search.similarity_search(query)
        return self.result_convert(result)

    # TODO: добавление элемента торчит наружу
    def add_item(self, item: dict) -> None:
        self.vector_search.add_texts([item['option_all_text']],
                                     metadatas=[item])

    # TODO: удаление элемента торчит наружу
    def delete_item(self, id: int) -> None:
        self.vector_search.delete([id])

    # TODO: move to OpenSearch filter
    def filter_search(self, query: str, filter: dict) -> list[dict]:
        result = self.search(query)
        key, value = list(filter.items())[0]
        filter_result = []

        for res in result:
            if res[key] == value:
              filter_result.append(res)

        return filter_result   
