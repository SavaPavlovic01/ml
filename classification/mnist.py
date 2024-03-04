from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV


def fetch_data():
    return fetch_openml('mnist_784', version=1)

def split_data(data):
    X, y = data['data'], data['target']
    return X[:60000], X[60000:], y[:60000], y[60000:]

def draw_from_index(data, index):
    some_digit = data.iloc[index]
    some_digit_image = some_digit.to_numpy().reshape(28,28)
    plt.imshow(some_digit_image, cmap=mpl.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    data = fetch_data()
    X_train, X_test, y_train, y_test = split_data(data)
   
   

    model = KNeighborsClassifier()
    #model.fit(X_train, y_train)

    param_grid = [{"n_neighbors":[1,2,3,4,5,6,7,8,9,10]}]

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", return_train_score=True, n_jobs=10)
    grid_search.fit(X_train, y_train)
    final_model = grid_search.best_estimator_
    print("done")
    y_train_pred = cross_val_predict(final_model, X_train, y_train)
    print(precision_score(y_train, y_train_pred))
    print(recall_score(y_train, y_train_pred))
    print(precision_score(y_train, y_train_pred))