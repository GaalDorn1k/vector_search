import json
import argparse

from search_service import SearchService
from flask import Flask, request, jsonify


app = Flask(__name__)


@app.route('/api/search', methods=['GET'])
def search() -> dict:
    query = request.args.get('query')
    result = service.search(query)
    return jsonify(result)


@app.route('/api/search_with_filter', methods=['POST'])
def search_with_filter() -> dict:
    query = request.args.get('query')
    filter = json.loads(request.values.get('filter'))
    result = service.filter_search(query, filter)
    return jsonify(result)

# TODO: добавление элемента торчит наружу
@app.route('/api/add_item', methods=['POST'])
def add_item() -> str:
    item = json.loads(request.values.get('item'))
    service.add_item(item)
    return ''

# TODO: удаление элемента торчит наружу
@app.route('/api/delete_item', methods=['GET'])
def delete_item() -> str:
    id = request.args.get('id')
    service.delete_item(id)
    return ''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='pipeline name',
                        default='config.json')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as jf:
        config = json.load(jf)

    service = SearchService(config)
    app.run(port=5000)
