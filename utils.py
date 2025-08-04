import pickle

from sklearn.decomposition import TruncatedSVD


def truncated_svd(X, n_components):
    model = TruncatedSVD(n_components=n_components)
    model.fit(X)
    return model.transform(X)


def write_pkl(obj, filename: str):
    pickle.dump(obj, open(filename, "wb"))


def read_pkl(filename: str):
    return pickle.load(open(filename, "rb"))
