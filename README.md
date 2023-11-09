# umap_flask_app

Flask APP to explore UMAP embeddings. It makes an interactive plot of the UMAP in 'embeddings.npy' color-coded by the labels.npy file, hooking each datapoint to a png in static/pngs.

When the user clicks a point, the server side makes a search in a KDTree for the k nearest neighbours. Layout is very barebones.

DECALS imaging for testing can be downloaded from here: https://drive.google.com/file/d/1LTU7nl9K0KR7dPZAVPvOikT0VgOt2c-h/view?usp=sharing

Run with 
``flask --app home run --host=0.0.0.0 --debug --port=5001``

Requirements:

flask
pandas
sklearn
numpy
