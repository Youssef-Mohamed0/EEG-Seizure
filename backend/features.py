import pandas as pd


# builds the 13 columns: the 10 original ones + cluster_id + pca_1 + pca_2
# used by both training and the server so they always compute it the same way
def enhance_features(X: pd.DataFrame, scaler, pca, kmeans):
    Xs = scaler.transform(X)
    Xp = pca.transform(Xs)
    out = X.copy()
    out["cluster_id"] = kmeans.predict(Xs)
    out["pca_1"] = Xp[:, 0]
    out["pca_2"] = Xp[:, 1]
    return out, Xp, Xs
