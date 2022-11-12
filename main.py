import numpy as np
import pandas as pd
from collections import Counter

# read in csv
iris = pd.read_csv('iris.csv')

# describe data
print(iris.shape)
print(iris.head())
print(iris.describe())

# divide data
features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
X = iris[features].values
y = iris['Name'].values


# define euclidean distance
def euclidean_distance(X_train, X_test_point):
    distances = []
    for row in range(len(X_train)):
        current_train_point = X_train[row]
        current_distance = 0

        for col in range(len(current_train_point)):
            current_distance += (current_train_point[col] - X_test_point[col]) ** 2

        current_distance = np.sqrt(current_distance)

        distances.append(current_distance)

    distances = pd.DataFrame(data=distances, columns=['dist'])
    return distances


# define nearest neighbor
def nearest_neighbors(point, k):
    nearest = point.sort_values(by=['dist'], axis=0)
    nearest = nearest[:k]

    return nearest


# define majority vote
def vote(nearest, y_train):
    vote_counter = Counter(y_train[nearest.index])
    y_pred = vote_counter.most_common()[0][0]

    return y_pred


# define k nearest neighbor
def knn(X_train, y_train, X_test, k):
    y_pred = 0

    for X_test_point in X_test:
        distance_point = euclidean_distance(X_train, X_test_point)
        nearest_point = nearest_neighbors(distance_point, k)
        y_pred = vote(nearest_point, y_train)

    return y_pred


# define input checker
def input_check(value, min_value, max_value):
    valid = False
    while not valid:
        if min_value <= value <= max_value:
            valid = True
            return value
        else:
            print('Invalid input.')
            value = float(input('Please enter a valid number between {} and {}: '.format(min_value, max_value)))


# get sepal length
sepal_length = float(input('Enter Sepal Length (between 4 and 8): '))
sepal_length = input_check(sepal_length, 4, 8)

# get sepal width
sepal_width = float(input('Enter Sepal Width (between 1 and 5): '))
sepal_width = input_check(sepal_width, 1, 5)

# get petal length
petal_length = float(input('Enter Petal Length (between 0 and 8): '))
petal_length = input_check(petal_length, 0, 8)

# get petal width
petal_width = float(input('Enter Petal Width (between 0 and 3): '))
petal_width = input_check(petal_width, 0, 3)

X_test = [[sepal_length, sepal_width, petal_length, petal_width]]

# predict flower type (k = 3)
pred_type = knn(X, y, X_test, 3)

# print prediction
print('The type of flower is:', pred_type)
