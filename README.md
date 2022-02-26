# Algorithms

## Overview

This directory contains code for running the recommender algorithms server. This server has been
separated from the main server to isolate large dependencies. The `/preferences` endpoints
accepts a series of ratings and outputs diverse items. 
See `tests/test_ratings.json` for example ratings schema.

## Usage

Start by installing all the dependencies (it is recommended to use `conda`):


|    Type     |        Location       |
|-------------|-----------------------|
| Algorithms  |  src/algs/lenskit11.yml |
| Server      |  requirements.txt     |
| Testing     |  test/     |

Then configure `src/config.json` 
and finally start the server with `python src/app.py`. 

## Data files
Please download the pre-trained data from https://drive.google.com/drive/folders/1OQw2oTX6no-tNNQ_oQUO9wnApV5alztQ?usp=sharing,
place them in this location: src/algs/data/



