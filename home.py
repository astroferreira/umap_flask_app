from flask import Flask, render_template, request, jsonify

import numpy as np
import pandas as pd
import glob

import glob
import re 
import math
from pathlib import Path 
file_pattern = re.compile(r'.*?(\d+).*?')

from sklearn.neighbors import KDTree

def get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])

app = Flask(__name__)
embeddings = np.load('embeddings.npy')
labels = np.load('labels.npy')
files = [int(f.split('/')[-1].split('.png')[0]) for f in np.array(sorted(glob.glob('static/pngs/*.png'), key=get_order))]


df = pd.DataFrame()
df['ids'] = files
df['x'] = embeddings[files, 0]
df['y'] = embeddings[files, 1]
df['labels'] = labels[files]    

tree = KDTree(embeddings[files], leaf_size=2)

@app.route("/UMAP")
def UMAP():
    opacity = np.array(['0.05']*len(files))
    opacity[np.where(df.labels == 1)] = 0.6
    return render_template("index.html", axis=1, col_index=2, col_name='labels', columns=df.columns.tolist(), s=df['labels'].values.tolist(), filter=filter, x=list(df.x.values), y=list(df.y.values), ids=list(files), opacity=opacity.tolist(), zip=zip)

@app.route("/neighbours")
def neighbours():
    
    id = int(request.args.get('id'))
    #k  is the number_of_neighbours
    dist, ind = tree.query(df.loc[df.ids == id][['x', 'y']].values.reshape(1, 2), k=100)     
    index = np.array(files)[ind[0]]
    return jsonify(ids=index.tolist())