from sklearn import datasets
from dtree.decision_tree import DecisionTreeClassifier


def main():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    decision_tree_clf = DecisionTreeClassifier(
        criterion_option = "gini",
        split_policy = "best",
    )

    decision_tree_clf.fit(X, y)


if __name__ == "__main__":
    main()  


