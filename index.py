#!/usr/bin/env python3

import numpy as np
import pandas as pd
import mglearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split

iris = load_iris()

# データフレームに格納
iris_dataframe = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)


iris_datalabel = pd.Series(data=iris.target)

print("===========iris_data=================")
print(iris_dataframe.head(10))
print("===========iris_datalabel=================")
print(iris_datalabel.tail(10))

# plot
plt.figure()  # キャンバスを用意
br = pd.plotting.scatter_matrix(iris_dataframe, c=iris_datalabel, figsize=(10, 10), marker="o",
                                hist_kwds={'bins': 20}, s=30, alpha=8, cmap=mglearn.cm3)
plt.savefig("plotting_scatter_matrix.png")

iris_dataset = sns.load_dataset("iris")
sns.pairplot(iris_dataset, hue='species',
             palette="husl").savefig('seaborn_iris.png')


# 決定木
def main():
    # モデル作成
    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf = clf.fit(iris.data, iris.target)

    # 作成したモデルを用いて予測を実行
    predicted = clf.predict(iris.data)
    print(predicted)
    print("------------------------")
    print(iris.target)

    # モデルの可視化
    f = tree.export_graphviz(clf, out_file='iris_model.dot', feature_names=iris.feature_names,
                             class_names=iris.target_names, filled=True, rounded=True)


if __name__ == '__main__':
    main()
