"""Shared feature engineering so training and serving cannot drift apart."""
import pandas as pd


def enhance_features(X: pd.DataFrame, scaler, pca, kmeans):
    """10 raw features -> (13-column frame, PCA array). Uses already-fitted transformers."""
    Xs = scaler.transform(X)
    Xp = pca.transform(Xs)
    out = X.copy()
    out["cluster_id"] = kmeans.predict(Xs)
    out["pca_1"] = Xp[:, 0]
    out["pca_2"] = Xp[:, 1]
    return out, Xp, Xs
