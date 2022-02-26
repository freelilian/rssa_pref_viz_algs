"""

app.py

Lijie Guo
Clemson University
10/17/2021

Server for running the prediction and diversification algorithms. See
`models.py` for information about the input and outputs.

"""

from pathlib import Path
import json

from flask import Flask, abort
from flask import request
from flask import render_template

from compute import predict_diverse_items

from models import Rating

app = Flask(__name__)


@app.route('/')
def show_readme():
    return render_template('README.html')

@app.route('/preferences', methods=['POST'])
def predict_preferences():
    req = request.json
    ratings = None

    try:
        ratings = req['ratings']
    except KeyError:
        abort(400)

    funcs = {
        'diverse_items': predict_diverse_items
    }

    ratings = [Rating(**rating) for rating in ratings]
    diversified_recs = {k: f(ratings=ratings, user_id='Bart') for k, f in funcs.items()}

    return dict(preferences=diversified_recs)


if __name__ == '__main__':
    config_path = Path(__file__).parent / 'config.json'
    with open(config_path) as f:
        settings = json.load(f)
    app.run(port=settings['port'],
            host=settings['host'],
            debug=settings['debug'])
