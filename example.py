import numpy as np

from sklearn import datasets
from decision_tree.decision_tree_classifier import DecisionTreeClassifier


def main():
    iris = datasets.load_iris()
    X_train = iris.data
    y_train = iris.target

    X_test = np.array([[5.2, 3.3, 1.2, 0.3],
                        [4.8, 3.1 , 1.6, 0.2],
                        [4.75, 3.1, 1.32, 0.1],
                        [5.9, 2.6, 4.1 , 1.2],
                        [5.1 , 2.2, 3.3, 1.1],
                        [5.2, 2.7, 4.1, 1.3],
                        [6.6, 3.1 , 5.25, 2.2],
                        [6.3, 2.5, 5.1 , 2.],
                        [6.5, 3.1 , 5.2, 2.1]])
    y_test = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    decision_tree_clf = DecisionTreeClassifier(
        criterion_option = "gini",
        split_policy = "best",
    )

    decision_tree_clf.fit(X_train, y_train)

    y_proba = decision_tree_clf.predict_proba(X_test)

    print(y_proba)


if __name__ == "__main__":
    main()  


