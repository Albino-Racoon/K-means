import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
    """Distance between two vectors."""
    distance = [(p - q) ** 2 for p, q in zip(x1, x2)]
    return sum(distance) ** .5
    ###distance = np.sqrt(np.sum((x1 - x2) ** 2))


    ###return distance

""""""""" print(x1)
    print(x2)
    distance = []
    for i in range(0, len(x1)):
        distance.append(x1[i] - x2[i])
    distance = tuple(distance)"""""""""



class KNNimp:
    def __init__(self, k=4):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # racuna razdaljo
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # dobi najblizji k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # večina
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
