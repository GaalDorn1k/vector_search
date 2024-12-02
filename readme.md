# Сервис смешанного поиска в OpenSearch

Реализует векторный поиск + полнотекстовый поиск (BM25)

В качестве эмбеддера использует [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)

Модель загружается из репозитория только один раз при первом запуске сервиса, далее хранится в **/.cache**

ВАЖНО!!! логика обработки полей привязана к структуре полей в индексе **options-index**


## Запуск

`python run.py`

## Конфигурирование

`файл config.json`

### Конфигурирование поиска

`config.search`

param      | type | default     | description
-----------|------|-------------|--------------------------------------
k          | int  | 10          | количество найденных элементов
space-type | str  | cosinesimil | тип векторного пространства (см. [документацию OpenSearch](https://opensearch.org/docs/latest/search-plugins/knn/approximate-knn/))
weights    | list | [0.8, 0.2]  | соотношение BM25/VectorSearch

## API

прилагается postman-collection: `VectorSearchRequests.postman_collection.json`

request                         | description                | params                                  |additional info
--------------------------------|----------------------------|-----------------------------------------|----------------
`GET /api/search`               | поиск                      | **param query: str**                    |
`POST /api/search_with_filter`  | поиск с фильром            | **param query: str; json filter: dict** | filter_example: `{"option_area": 4066.8}`
`POST /api/add_item`            | добавить элемент в индекс  | **json item: dict**                     | Торчит наружу!
`POST /api/delete_item`         | удалить элемент из индекса | **param id: str**                       | Торчит Наружу!
