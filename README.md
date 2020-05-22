# minas

Python implementation of the [MINAS](http://www.liaad.up.pt/area/jgama/MINAS.pdf) novelty detection algorithm for data streams.

## Quick-Start Guide

This documents presents an overview of how to run a simple example using the `Minas` classifier. It was implemented by extending the base estimator from `scikit-multiflow`, so it works similarly to the popular `scikit-learn` API.

The `Minas` classifier defines the following methods:

- `fit` – Trains a model in the offline phase, in a batch fashion.
- `partial_fit` – Incrementally trains the stream model.
- `predict` – Predicts the target’s value.

You can also run the [Quick-Start jupyter notebook](Quick-Start.ipynb) to follow the steps described below.

### Train and test a stream classification model using `Minas`

#### 1. Import module

Before we start, we have to import the classifier.


```python
from minas import Minas
```

#### 2. Create a stream

We create a stream using `RandomRBFGenerator`, the Random Radial Basis Function stream generator from `scikit-multiflow`. In this example, we will create a data stream with 3 classes, 4 features, and 6 centroids.

Also, before using the stream, we need to prepare it by calling `prepare_for_use()`.


```python
from skmultiflow.data.random_rbf_generator import RandomRBFGenerator

stream = RandomRBFGenerator(model_random_state=123,
                            sample_random_state=12,
                            n_classes=3,
                            n_features=4,
                            n_centroids=6)
stream.prepare_for_use()
```

#### 3. Instantiate the `Minas` classifier

The classifier takes the following parameters:

- `kini`: Number of clusters for each class to be found during clustering (offline phase and novelty detection process).
- `cluster_algorithm`: A string containing the clustering algorithm to use. Currently only supports `'kmeans'`.
- `random_state`: Seed to use for random number generation.
- `min_short_mem_trigger`: Minimum number of samples in the short term memory required to trigger a novelty detection process.
- `min_examples_cluster`: Minimum number of examples required to form a cluster.
- `threshold_strategy`: Strategy used to compute the threshold for differentiating between novelty classes and concept extensions. Accepts `1`, `2`, or `3`. The strategies are defined in the [MINAS paper](http://www.liaad.up.pt/area/jgama/MINAS.pdf).
- `threshold_factor`: Factor to use for calculating thresholds.
- `window_size`: Window size (an integer representing the number of samples) used by the forgetting mechanism.
- `update_summary`: Defaults to `False`. If `True`, the summary statistics for a cluster are updated when a new point is added to it.
- `animation`: Defaults to `False`. If `True`, a plot is created showing the current state of the model (points and clusters). It only works if the examples have two dimensions.

For this example, we will set 10 clusters per class, with at least 30 examples required in the short term memory before triggering a novelty detection procedure, and a minimum of 10 examples per cluster.


```python
clf = Minas(kini=10,
            min_short_mem_trigger=30,
            min_examples_cluster=10)
```

#### 4. Get data from the stream

Next, we will get the data from the stream. For this example, we use 500 samples to train our model in the offline phase. Then, we will use the next 500 samples for the online phase.


```python
n_samples = 1000
offline_size = 500

X_all, y_all = stream.next_sample(n_samples)
X_train = X_all[:offline_size]
y_train = y_all[:offline_size]
X_test = X_all[offline_size:n_samples]
y_test = y_all[offline_size:n_samples]
```

#### 5. Offline phase

The next step corresponds to the offline phase. We run it by calling `fit()` with the training data from the last step.


```python
# OFFLINE phase
clf.fit(X_train, y_train)
```




#### 6. Online phase

Now we get to the online phase. We feed each example at a time to the model, and collect the results from the `predict()` calls. In each iteration, we call `partial_fit()` to update the model with the data from the sample it has just seen.


```python
# ONLINE phase
y_preds = []
for X, y in zip(X_test, y_test):
    y_preds.append(clf.predict([X])[0])
    clf.partial_fit([X], [y])
```

#### 7. Evaluate performance

Finally, we will see how our model performed. We create a confusion matrix with the following definition:

- Each row represents the true label of the examples from the stream seen during the online phase.
- Each column represents the label predicted by the model. The numbers greater than the highest label shown in the rows represent the novelty patterns detected. Unknown samples are represented by `-1`.

In the confusion matrix below, for our example, the labels from our data set are 0, 1, and 2. The columns 3, 4, and 5 represent 3 novelty patterns discovered by the model. We can see that the model identified classes 0 and 2 quite well, and had more trouble distinguishing examples from class 1.


```python
clf.confusion_matrix(X_test, y_test)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>-1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>52</td>
      <td>4</td>
      <td>15</td>
      <td>2</td>
      <td>0</td>
      <td>10</td>
      <td>23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>39</td>
      <td>6</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
      <td>26</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>24</td>
      <td>237</td>
      <td>9</td>
      <td>11</td>
      <td>2</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
</div>

#### Putting it all together

```
from minas import Minas
from skmultiflow.data.random_rbf_generator import RandomRBFGenerator

# Set up stream
stream = RandomRBFGenerator(model_random_state=123,
                            sample_random_state=12,
                            n_classes=3,
                            n_features=4,
                            n_centroids=6)
stream.prepare_for_use()

# Create classifier
clf = Minas(kini=10,
            min_short_mem_trigger=30,
            min_examples_cluster=10)

# Get stream data
n_samples = 1000
offline_size = 500

X_all, y_all = stream.next_sample(n_samples)
X_train = X_all[:offline_size]
y_train = y_all[:offline_size]
X_test = X_all[offline_size:n_samples]
y_test = y_all[offline_size:n_samples]

# OFFLINE phase
clf.fit(X_train, y_train)

# ONLINE phase
y_preds = []
for X, y in zip(X_test, y_test):
    y_preds.append(clf.predict([X])[0])
    clf.partial_fit([X], [y])

# View confusion matrix 
clf.confusion_matrix(X_test, y_test)
```