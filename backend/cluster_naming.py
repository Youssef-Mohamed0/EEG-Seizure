"""Name K-Means clusters from what actually distinguishes them (never from a guess).

Centers live in StandardScaler space, so a value of +1.0 means "one standard deviation above
the training average". Only features with |z| >= MIN_Z are mentioned; names therefore change
if the clusters change, and never claim e.g. 'high intensity' unless heart rate really differs.
"""
FEATURE_WORDS = {  # feature: (English, Arabic)
    "Age": ("age", "العمر"), "Weight_kg": ("weight", "الوزن"), "Height_m": ("height", "الطول"),
    "Max_BPM": ("max heart rate", "أقصى نبض"), "Avg_BPM": ("avg heart rate", "متوسط النبض"),
    "Resting_BPM": ("resting heart rate", "نبض الراحة"),
    "Session_Duration_min": ("session duration", "مدة الجلسة"), "Fat_Percentage": ("body fat", "نسبة الدهون"),
    "Water_Intake_liters": ("water intake", "شرب المياه"), "BMI": ("BMI", "BMI"),
}
MIN_Z = 0.5
TOP_N = 2


def describe_clusters(centers, feature_cols, sizes=None):
    """centers: (k, n_features) array in scaled space -> {cluster_id: {en, ar, z, size}}."""
    out = {}
    for k, row in enumerate(centers):
        z = {f: float(v) for f, v in zip(feature_cols, row)}
        top = sorted((f for f in z if abs(z[f]) >= MIN_Z), key=lambda f: -abs(z[f]))[:TOP_N]
        if top:
            en = ", ".join(f"{'high' if z[f] > 0 else 'low'} {FEATURE_WORDS[f][0]}" for f in top)
            ar = "، ".join(f"{FEATURE_WORDS[f][1]} {'أعلى' if z[f] > 0 else 'أقل'}" for f in top)
        else:
            en, ar = "near-average profile", "بروفايل قريب من المتوسط"
        out[k] = {"en": en, "ar": ar, "z": {f: round(v, 2) for f, v in z.items()},
                  "size": int(sizes[k]) if sizes is not None else None}
    return out
