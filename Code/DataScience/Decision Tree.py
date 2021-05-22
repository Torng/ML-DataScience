from sklearn.datasets import load_iris
from sklearn import tree
from sklearn import neighbors
from sklearn import ensemble
from sklearn import metrics
from sklearn.model_selection import train_test_split
# 讀入鳶尾花資料
iris = load_iris()
iris_X = iris.data
iris_y = iris.target
print(iris_X)
print(iris_y)

# 切分訓練與測試資料
train_X, test_X, train_y, test_y = train_test_split(iris_X, iris_y, test_size = 0.3,shuffle=True)

# 建立分類器
# clf = tree.DecisionTreeClassifier()
clf = ensemble.RandomForestClassifier()
iris_clf = clf.fit(train_X, train_y)


# 預測
test_y_predicted = iris_clf.predict(test_X)
print(test_y_predicted)

# 標準答案
print(test_y)

accuracy = metrics.accuracy_score(test_y, test_y_predicted)
print(accuracy)