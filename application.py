from flask import Flask, render_template, request, session
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances_argmin_min
from flask.ext.session import Session

minnow = "static/minnow.csv"
trained_minnow = pd.read_csv(minnow)
trained_minnow.fillna(trained_minnow.mean(), inplace=True)
sess = Session()

labelEncoder = LabelEncoder()
labelEncoder.fit(trained_minnow['Survived'])
trained_minnow['Survived'] = labelEncoder.transform(trained_minnow['Survived'])
global_kmeans = ''
global_x = ''
global_mydict = ''

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
    session['kmeans'] = kmeans
    centroids = kmeans.cluster_centers_
    clusters = kmeans.fit_predict(trained_minnow[[x_column, y_column]])
    session['X'] = trained_minnow[[x_column, y_column]]
    distances = []
    max_dist = []
    for i, (cx, cy) in enumerate(centroids):
        mean_distance = k_mean_distance(trained_minnow[[x_column, y_column]], cx, cy, i, clusters)
        distances.append(mean_distance)
    for distance_array in distances:
        max_dist.append(max(distance_array))
    print(max_dist)
    end_time = datetime.datetime.now()
    mydict = {i: np.where(kmeans.labels_ == i)[0] for i in range(k_input)}
    session['mydict'] = mydict
    total_time = end_time - start_time
    return render_template('index.html', centroids=centroids, result=mydict, max_indices=max_dist)

def k_mean_distance(data, cx, cy, i_centroid, cluster_labels):
    distances = []
    for (x, y) in data[cluster_labels == i_centroid].values:
        distances.append(np.sqrt((x-cx)**2+(y-cy)**2))
    return distances

@application.route('/kmeansglobal', methods=['GET'])
def globalkmeans():
    x_co = float(request.args['x_co'])
    y_co = float(request.args['y_co'])
    centroids = session.get('kmeans').cluster_centers_
    closest, _ = pairwise_distances_argmin_min(centroids, session.get('X'))
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = trained_minnow.index.values
    cluster_map['cluster'] = session.get('kmeans').labels_
    # print(cluster_map[cluster_map.cluster == np.where(centroids == [x_co, y_co])[0]-1])
    # print(closest[np.where(centroids == [x_co, y_co])[0]-1])
    mylist = np.where(centroids == [x_co, y_co])
    print(mylist[0][0])
    return render_template("newindex.html", result=trained_minnow['Fname'][mylist[0][0]])

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
    application.secret_key = 'super secret key'
    application.config['SESSION_TYPE'] = 'filesystem'

    sess.init_app(application)
    application.debug = True
    application.run()