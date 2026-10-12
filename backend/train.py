"""Training: RandomForest (risiko) + KMeans (cluster) + Scaler. Output ke backend/artifacts/."""
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, silhouette_score

BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts"
ART.mkdir(parents=True, exist_ok=True)
DATA = BASE.parent / "data" / "dataset_sintetik.csv"

NUM = ["semester", "ipk", "kehadiran_pct", "jam_belajar_minggu",
       "tugas_selesai_pct", "kuis_avg", "forum_aktif", "partisipasi",
       "beasiswa", "ekstrakurikuler"]
SOS_MAP = {"rendah": 0, "menengah": 1, "tinggi": 2}

df = pd.read_csv(DATA)
X_num = df[NUM].copy()
X_num["sosek"] = df["sosek"].map(SOS_MAP)
FEATS = NUM + ["sosek"]

scaler = StandardScaler()
Xs = scaler.fit_transform(X_num[FEATS])

# KMeans cluster persona
km = KMeans(n_clusters=4, n_init=10, random_state=42)
clusters = km.fit_predict(Xs)
sil = silhouette_score(Xs, clusters)

# RandomForest risiko
Xtr, Xte, ytr, yte = train_test_split(Xs, df["risiko"], test_size=0.2, random_state=42, stratify=df["risiko"])
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(Xtr, ytr)
acc = accuracy_score(yte, rf.predict(Xte))
f1 = f1_score(yte, rf.predict(Xte), average="macro")

joblib.dump(rf, ART / "rf.pkl")
joblib.dump(km, ART / "kmeans.pkl")
joblib.dump(scaler, ART / "scaler.pkl")
(ART / "meta.json").write_text(json.dumps({
    "version": "v1.0.0", "features": FEATS,
    "accuracy": round(float(acc), 4), "f1_macro": round(float(f1), 4),
    "silhouette": round(float(sil), 4),
    "importance": dict(sorted(
        {f: round(float(v), 4) for f, v in zip(FEATS, rf.feature_importances_)}.items(),
        key=lambda x: -x[1])),
}, indent=2))

print(f"accuracy={acc:.3f} f1={f1:.3f} silhouette={sil:.3f}")
print("Artefak tersimpan di backend/artifacts/")
