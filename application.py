from flask import Flask, render_template, request
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

titanic_data = "static/titanic3.xls"
trained_titanic = pd.read_excel(titanic_data)
trained_titanic.fillna(trained_titanic.mean(), inplace=True)

labelEncoder = LabelEncoder()
labelEncoder.fit(trained_titanic['sex'])
trained_titanic['sex'] = labelEncoder.transform(trained_titanic['sex'])
trained_titanic['embarked'] = labelEncoder.fit_transform(trained_titanic['embarked'].astype(str))

# print(trained_titanic.info())

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
    kmeans = KMeans(n_clusters=k_input)
    kmeans = kmeans.fit(trained_titanic[[x_column, y_column]])
    centroids = kmeans.cluster_centers_
    clusters = kmeans.fit_predict(trained_titanic[[x_column, y_column]])
    mydict = {i: np.where(kmeans.labels_ == i)[0] for i in range(k_input)}
    sse = []
    graph_y = []
    myrange = [5, 20, 100]
    distances = []
    my_euclidean_range = []
    for i in range(k_input):
        if (i+1 >= k_input):
            distances.append(np.linalg.norm(centroids[0]-centroids[k_input - 1]))
            my_euclidean_range.append(str(0)+"-"+str(k_input - 1))
            break
        my_euclidean_range.append(str(i)+"-"+str(i+1))
        distances.append(np.linalg.norm(centroids[i]-centroids[i+1]))
    for k in myrange:
        km = KMeans(n_clusters=k)
        km.fit(trained_titanic[[x_column, y_column]])
        graph_y.append(k)
        sse.append(km.inertia_)
    graph_data = { 'x': sse, 'y': graph_y }
    distance_dict = dict(zip(my_euclidean_range, distances))
    return render_template('index.html', result=mydict, centroids=centroids, graph_data=graph_data, distance_dict=distance_dict)

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run()