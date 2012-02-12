from sklearn.mixture.semisupervised_gmm import SemisupervisedGMM
import sklearn.datasets
iris = sklearn.datasets.load_iris()
X = iris.data
y = iris.target_names[iris.target]
model = SemisupervisedGMM(3, "full")
model.fit(X[::2], y[::2], X[1::2])
model.precision(X, y)