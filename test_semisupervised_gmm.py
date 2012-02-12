from sklearn.mixture.semisupervised_gmm import SupervisedGMM, SemisupervisedGMM
import sklearn.datasets
iris = sklearn.datasets.load_iris()
X = iris.data
y = iris.target_names[iris.target]
covtype = "full"
supervised_model = SupervisedGMM(3, covtype)
supervised_model.fit(X[::2], y[::2])
print supervised_model.precision(X, y)
semisupervised_model = SemisupervisedGMM(3, covtype)
semisupervised_model.fit(X[::2], y[::2], X[1::2])
print semisupervised_model.precision(X, y)