import numpy as np
from time import time

from skmultiflow.core import BaseSKMObject, ClassifierMixin

from sklearn.cluster import KMeans


class Minas(BaseSKMObject, ClassifierMixin):

    def __init__(self,
                 kini=3,
                 cluster_algorithm='kmeans',
                 random_state=0):
        super().__init__()
        self.kini = kini
        self.random_state = random_state

        accepted_algos = ['kmeans']  # TODO: define list of accepted algos
        if cluster_algorithm not in accepted_algos:
            print('Available algorithms: {}'.format(', '.join(accepted_algos)))
        else:
            self.cluster_algorithm = cluster_algorithm

        self.microclusters = []  # list of microclusters

    def fit(self, X, y, classes=None, sample_weight=None):
        self.microclusters = self.offline(X, y)
        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        # TODO
        self.fit(X, y)
        return self

    def predict(self, X):
        """X is an array"""
        # TODO: remove this ugly loop
        predictions = []
        for point in X:
            # find closest centroid
            closest_cluster = min(self.microclusters,
                                  key=lambda cl: cl.distance_to_centroid(point))
            if closest_cluster.is_inside(point):  # classify in this cluster
                predictions.append(closest_cluster.label)
            else:  # classify as unknown
                predictions.append(0)
        return np.asarray(predictions)

    def predict_proba(self, X):
        # TODO
        pass

    def offline(self, X_train, y_train):

        microclusters = []
        # in offline phase, consider all instances arriving at the same time in the microclusters:
        timestamp = time()
        if self.cluster_algorithm == 'kmeans':
            for y_class in np.unique(y_train):
                # subset with instances from each class
                X_class = X_train[y_train == y_class]

                class_cluster_clf = KMeans(n_clusters=self.kini,
                                           random_state=self.random_state)
                class_cluster_clf.fit(X_class)

                for class_cluster in np.unique(class_cluster_clf.labels_):
                    # get instances in cluster
                    cluster_instances = X_class[class_cluster_clf.labels_ == class_cluster]

                    microclusters.append(
                        MicroCluster(y_class, cluster_instances, timestamp)
                    )

        return microclusters


class MicroCluster(object):

    def __init__(self,
                 label,  # the class the microcluster belongs to
                 instances,
                 timestamp=0):

        # TODO: remove instances so it doesn't need to be stored in memory
        super(MicroCluster, self).__init__()
        self.label = label
        self.instances = instances
        self.n = len(instances)
        self.linear_sum = instances.sum(axis=0)
        self.squared_sum = np.square(instances).sum(axis=0)
        self.timestamp = timestamp

        self.centroid = self.linear_sum / self.n
        self.radius = self.get_radius()

    def __str__(self):
        return f"""
        Microcluster for target class {self.label}
        Number of instances: {self.n}
        Linear sum: {self.linear_sum}
        Squared sum: {self.squared_sum}
        Centroid: {self.centroid}
        Radius: {self.radius}
        Timestamp of last change: {self.timestamp}"""

    def get_radius(self):
        """Return radius of the subcluster"""
        # from sklearn:
        # dot_product = -2 * np.dot(self.linear_sum, self.centroid)
        # return np.sqrt(
        #   ((self.squared_sum + dot_product) / self.n) + self.squared_sum
        # )
        # from BIRCH paper:
        return np.sqrt(
            np.square(self.distance_to_centroid(self.instances)).sum() / self.n
        )
        # from MINAS paper:
        # factor = 1.1
        # return factor*np.std(self.distance_to_centroid(self.instances))

    def distance_to_centroid(self, X):
        """X is a numpy array of instances"""
        if len(X.shape) == 1:  # X is only one point
            return np.linalg.norm(X - self.centroid)
        else:  # X contains several points
            return np.linalg.norm(X - self.centroid, axis=1)

    def is_inside(self, X):
        """Check if points in X are inside this microcluster.
        X is an array."""
        return np.less(self.distance_to_centroid(X), self.radius)
