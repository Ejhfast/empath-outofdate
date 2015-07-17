import fileinput
from collections import defaultdict
from numpy import zeros
from numpy import add
from numpy import dot
from numpy import matrix
from numpy.linalg import norm
import math
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
from flask import Flask, request, render_template
import flask
from bson.json_util import dumps
from bson.objectid import ObjectId
from flask_cors import *
from clarifai.client import ClarifaiApi
import json
from gensim.models import Word2Vec

app = Flask(__name__)

app.debug = True

app.config['CORS_ORIGINS'] = ['*']
app.config['CORS_HEADERS'] = ['Content-Type']

model = Word2Vec.load("deepmodel3")

@app.route("/query/<words>")
@cross_origin(headers=['Content-Type'])
def query(words):
  words = [x.lstrip().rstrip() for x in words.split(",")]
  results = model.most_similar(positive=words,topn=100)
  return json.dumps(results)

@app.route("/")
@cross_origin(headers=['Content-Type'])
def index():
  return render_template("index.html")

if(__name__ == "__main__"):
  app.run(debug = True,port=8080)
