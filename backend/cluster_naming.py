# readable names for each feature (english, arabic)
FEATURE_WORDS = {
    "Age": ("age", "العمر"), "Weight_kg": ("weight", "الوزن"), "Height_m": ("height", "الطول"),
    "Max_BPM": ("max heart rate", "أقصى نبض"), "Avg_BPM": ("avg heart rate", "متوسط النبض"),
    "Resting_BPM": ("resting heart rate", "نبض الراحة"),
    "Session_Duration_min": ("session duration", "مدة الجلسة"), "Fat_Percentage": ("body fat", "نسبة الدهون"),
    "Water_Intake_liters": ("water intake", "شرب المياه"), "BMI": ("BMI", "BMI"),
}
MIN_Z = 0.5   # ignore features that are close to the average
TOP_N = 2     # how many features go into the name


# names every cluster after the features that make it different from the others
def describe_clusters(centers, feature_cols, sizes=None):
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
