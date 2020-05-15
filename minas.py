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
        self.before_offline_phase = True

        self.short_mem = []
        self.sleep_mem = []

    def fit(self, X, y, classes=None, sample_weight=None):
        """fit means fitting in the OFFLINE phase"""
        self.microclusters = self.offline(X, y)
        self.before_offline_phase = False
        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        if self.before_offline_phase:
            self.fit(X, y)
        else:
            y_preds, cluster_preds = self.predict(X, ret_cluster=True)
            timestamp = time()
            # TODO: remove this ugly loop too
            for point_x, y_pred, cluster in zip(X, y_preds, cluster_preds):
                if y_pred != -1:  # the model can explain point_x
                    self.update_cluster(cluster, point_x, y_pred, timestamp)
                else:  # the model cannot explain point_x
                    pass  # TODO
        return self

    def predict(self, X, ret_cluster=False):
        """X is an array"""
        # TODO: remove this ugly loop
        pred_labels = []
        pred_clusters = []
        for point in X:
            # find closest centroid
            closest_cluster = min(self.microclusters,
                                  key=lambda cl: cl.distance_to_centroid(point))
            if closest_cluster.is_inside(point):  # classify in this cluster
                pred_labels.append(closest_cluster.label)
                pred_clusters.append(closest_cluster)
            else:  # classify as unknown
                pred_labels.append(-1)
                pred_clusters.append(None)
        if ret_cluster:
            return np.asarray(pred_labels), pred_clusters
        else:
            return np.asarray(pred_labels)

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

    def update_cluster(self, cluster, X, y, timestamp):
        """

        Parameters
        ----------
        cluster :minas.MicroCluster
        X :numpy.ndarray
            X is one point.
        y :numpy.int64

        Returns
        -------

        """
        assert len(X.shape) == 1  # it's just one point
        cluster.n += 1
        cluster.linear_sum = np.sum([cluster.linear_sum, X], axis=0)
        cluster.squared_sum = np.sum([cluster.squared_sum, np.square(X)], axis=0)
        cluster.timestamp = timestamp
        cluster.instances = np.append(cluster.instances, [X], axis=0)  # TODO: remove later when dropping instances from class
        cluster.update_properties()


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

        self.update_properties()

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

    def update_properties(self):
        """
        Update centroid and radius based on current cluster properties.

        Returns
        -------
        None

        """
        self.centroid = self.linear_sum / self.n
        self.radius = self.get_radius()
