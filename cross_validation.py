from nearest_neighbors import KNNClassifier
from sklearn.model_selection import KFold

def knn_cross_val_score(X, y, k_list, score, cv=None, **kwargs):
    res = {}
    for k in k_list:
        knn = KNNClassifier(k, **kwargs)
        res[k] = []
        kf = KFold(k)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            knn.fit(X_train, y_train)
            my_y = knn.predict(X_test)
            correct = sum([1 for i,v in enumerate(my_y) if y_test[i] is v])
            res[k].append(correct/len(my_y))

    return res