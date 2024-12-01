import json
import argparse

from search_service import SearchService
from flask import Flask, request, jsonify


app = Flask(__name__)


@app.route('/api/search', methods=['GET'])
def request_process() -> dict:
    query = request.args.get('search')
    result = service.search(query)
    return jsonify(result)


@app.route('/api/search_with_filter', methods=['POST'])
def request_process() -> dict:
    query = request.args.get('search')
    filter = json.loads(request.values.get('filter'))
    result = service.filter_search(query, filter)
    return jsonify(result)


@app.route('/api/add_item', methods=['POST'])
def request_process() -> str:
    item = json.loads(request.values.get('item'))
    service.add_item(item)
    return ''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='pipeline name',
                        default='configs.json')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as jf:
        config = json.load(jf)

    service = SearchService(config)
    app.run(port=5000)
