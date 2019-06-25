from flask import Flask, render_template, request
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

minnow = "static/minnow.csv"
trained_minnow = pd.read_csv(minnow)
trained_minnow.fillna(trained_minnow.mean(), inplace=True)

labelEncoder = LabelEncoder()
labelEncoder.fit(trained_minnow['Survived'])
trained_minnow['Survived'] = labelEncoder.transform(trained_minnow['Survived'])

# print(trained_minnow.info())

# EB looks for an 'application' callable by default.
application = Flask(__name__)

@application.route('/', methods=['GET'])
def myapp():
    return render_template('index.html')

@application.route('/kmeans', methods=['GET'])
def kmeans():
    x_column = request.args['x_value']
    y_column = request.args['y_value']
    k_input = int(request.args['k_input'])
    start_time = datetime.datetime.now()
    kmeans = KMeans(n_clusters=k_input)
    kmeans = kmeans.fit(trained_minnow[[x_column, y_column]])
    centroids = kmeans.cluster_centers_
    clusters = kmeans.fit_predict(trained_minnow[[x_column, y_column]])
    end_time = datetime.datetime.now()
    mydict = {i: np.where(kmeans.labels_ == i)[0] for i in range(k_input)}
    total_time = end_time - start_time
    return render_template('index.html', result=mydict, time_run=total_time.total_seconds())

@application.route('/myrest', methods=['GET'])
def myrest():
    sse = []
    myrange = range(1, 10, 1)
    graph_x = []
    x_column = request.args['x_value']
    y_column = request.args['y_value']
    for k in myrange:
        km = KMeans(n_clusters=k)
        km.fit(trained_minnow[[x_column, y_column]])
        graph_x.append(k)
        sse.append(km.inertia_)
    graph_data = { 'x': graph_x, 'y': sse }
    return render_template('newindex.html', graph_data=graph_data)


# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run()