import numpy as np
from time import time

import pandas as pd
from matplotlib import pyplot as plt
# from celluloid import Camera

from skmultiflow.core import BaseSKMObject, ClassifierMixin

from sklearn.cluster import KMeans


class Minas(BaseSKMObject, ClassifierMixin):

    def __init__(self,
                 kini=3,
                 cluster_algorithm='kmeans',
                 random_state=0,
                 min_short_mem_trigger=50,
                 min_examples_cluster=10):
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
        self.min_short_mem_trigger = min_short_mem_trigger
        self.min_examples_cluster = min_examples_cluster

        # TODO use Camera
        # self.fig = plt.figure()
        # self.camera = Camera(self.fig)
        self.animation_frame_num = 0

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
                    self.short_mem.append(ShortMemInstance(point_x, timestamp))
                    if len(self.short_mem) >= self.min_short_mem_trigger:
                        self.novelty_detect()

        self.plot_clusters()

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
        cluster.instances = np.append(cluster.instances, [X],
                                      axis=0)  # TODO: remove later when dropping instances from class
        cluster.update_properties()

    def novelty_detect(self):
        possible_clusters = []
        X = np.array([instance.point for instance in self.short_mem])
        if self.cluster_algorithm == 'kmeans':
            cluster_clf = KMeans(n_clusters=self.kini,
                                 random_state=self.random_state)
            cluster_clf.fit(X)
            for cluster_label in np.unique(cluster_clf.labels_):
                cluster_instances = X[cluster_clf.labels_ == cluster_label]
                possible_clusters.append(
                    MicroCluster(-1, cluster_instances, 0))  # TODO: check later, is timestamp important here?
            for cluster in possible_clusters:
                if cluster.is_cohesive(self.microclusters) and cluster.is_representative(self.min_examples_cluster):
                    closest_cluster = cluster.find_closest_cluster(self.microclusters)
                    closest_distance = cluster.distance_to_centroid(closest_cluster.centroid)

                    threshold = self.best_threshold(closest_cluster)

                    if closest_distance < threshold:  # the new microcluster is an extension
                        cluster.label = closest_cluster.label
                    else:  # the new microcluster is a novelty pattern
                        cluster.label = max([cluster.label for cluster in self.microclusters]) + 1

                    # add the new cluster to the model
                    self.microclusters.append(cluster)

                    # remove these examples from short term memory
                    for instance in cluster.instances:
                        self.short_mem.remove(instance)
                else:
                    # keep these examples in short term memory
                    pass

    @staticmethod
    def best_threshold(cluster, strategy=1):
        # TODO implement other strategies according to paper
        # factor = 1.1
        factor = 3
        if strategy == 1:
            return factor * np.std(cluster.distance_to_centroid(cluster.instances))

    def plot_clusters(self):
        """Simplistic plotting, assumes elements in cluster have two dimensions"""
        points = pd.DataFrame(columns=['x', 'y', 'label'])
        cluster_info = pd.DataFrame(columns=['label', 'centroid', 'radius'])
        for cluster in self.microclusters:
            cluster_info = cluster_info.append(pd.Series({'label': cluster.label,
                                                          'centroid': cluster.centroid,
                                                          'radius': cluster.radius}),
                                               ignore_index=True)
            for point in cluster.instances:
                points = points.append(pd.Series({'x': point[0],
                                                  'y': point[1],
                                                  'label': cluster.label}),  # TODO turn into int
                                       ignore_index=True)

        color_names = ['k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                       'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        assert len(cluster_info.label.unique()) <= len(color_names)  # limited to these colors for now
        # colormap to be indexed by label id
        colormap = pd.DataFrame({'name': color_names}, index=range(-1, len(color_names) - 1))
        mapped_label_colors = colormap.loc[points['label']].values[:, 0]
        plt.scatter(points['x'], points['y'], c=mapped_label_colors, s=10, alpha=0.3)
        plt.gca().set_aspect('equal', adjustable='box')  # equal scale for both axes

        circles = []
        for label, centroid, radius in cluster_info.values:
            circles.append(plt.Circle((centroid[0], centroid[1]), radius,
                                      color=colormap.loc[label].values[0], alpha=0.1))
        for circle in circles:
            plt.gcf().gca().add_artist(circle)

        # self.camera.snap()
        plt.savefig(f'animation/clusters_{self.animation_frame_num:05}.png', dpi=300)
        plt.close()
        self.animation_frame_num += 1

    def plot_animation(self):
        pass
        # TODO
        # animation = self.camera.animate()
        # animation.save('animation.mp4')


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
        return f"""Target class {self.label}
# of instances: {self.n}
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
        """Returns distance from X to centroid of this cluster.

        Parameters
        ----------
        X : numpy.ndarray
        """
        if len(X.shape) == 1:  # X is only one point
            return np.linalg.norm(X - self.centroid)
        else:  # X contains several points
            return np.linalg.norm(X - self.centroid, axis=1)

    def is_inside(self, X):
        """Check if points in X are inside this microcluster.

        Parameters
        ----------
        X : numpy.ndarray

        Returns
        -------
        numpy.bool_"""
        return np.less(self.distance_to_centroid(X), self.radius)

    def find_closest_cluster(self, clusters):
        """Finds closest cluster to this one among passed clusters.

        Parameters
        ----------
        clusters : List[minas.MicroCluster]

        Returns
        -------
        minas.MicroCluster

        """
        return min(clusters, key=lambda cl: cl.distance_to_centroid(self.centroid))

    def update_properties(self):
        """
        Update centroid and radius based on current cluster properties.

        Returns
        -------
        None

        """
        self.centroid = self.linear_sum / self.n
        self.radius = self.get_radius()

    def is_cohesive(self, clusters):
        """Verifies if this cluster is cohesive for novelty detection purposes.

        A new micro-cluster is cohesive if its silhouette coefficient is larger than 0.
        b represents the Euclidean distance between the centroid of the new micro-cluster and the centroid of its
        closest micro-cluster, and a represents the standard deviation of the distances between the examples of the
        new micro-cluster and the centroid of the new micro-cluster.

        Parameters
        ----------
        clusters : List[minas.MicroCluster]

        Returns
        -------

        """
        b = self.distance_to_centroid(self.find_closest_cluster(clusters).centroid)
        a = np.std(self.distance_to_centroid(self.instances))
        silhouette = (b - a) / max(a, b)  # hm, this is always positive if b > a
        return silhouette > 0

    def is_representative(self, min_examples):
        """Verifies if this cluster is representative for novelty detection purposes.

        A new micro-cluster is representative if it contains a minimal number of examples,
        where this number is a user-defined parameter.

        Parameters
        ----------
        min_examples : int

        Returns
        -------
        bool

        """
        return self.n >= min_examples


class ShortMemInstance:
    def __init__(self, point, timestamp):
        self.point = point
        self.timestamp = timestamp

    def __eq__(self, other):
        """
        I'm considering elements equal if they have the same values for all variables.
        This currently does not consider the timestamp.
        """
        if type(other) == np.ndarray:
            return np.all(self.point == other)
