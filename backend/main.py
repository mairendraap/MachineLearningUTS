"""SIAP Backend — FastAPI. Satu-satunya tempat inference ML (lihat design.md)."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts"
DATA = BASE.parent / "data" / "dataset_sintetik.csv"
DB = BASE / "siap.db"

CLUSTER_LABELS = {0: "Deep Learner", 1: "Pejuang Deadline", 2: "Butuh Dukungan", 3: "Pasif"}

KATALOG = [
    {"id": "M1", "judul": "Manajemen Waktu & Pomodoro", "tipe": "Modul", "durasi": "45 mnt", "level": "Dasar", "untuk": ["Pejuang Deadline", "Pasif"]},
    {"id": "M2", "judul": "Remedial Kalkulus: Limit & Turunan", "tipe": "Video", "durasi": "60 mnt", "level": "Dasar", "untuk": ["Butuh Dukungan"]},
    {"id": "M3", "judul": "Python untuk Analisis Data", "tipe": "Praktik", "durasi": "90 mnt", "level": "Menengah", "untuk": ["Deep Learner", "Pejuang Deadline"]},
    {"id": "M4", "judul": "Teknik Catat Cornell", "tipe": "Modul", "durasi": "30 mnt", "level": "Dasar", "untuk": ["Pasif", "Butuh Dukungan"]},
    {"id": "M5", "judul": "Konsultasi Dosen Wali (template)", "tipe": "Aksi", "durasi": "20 mnt", "level": "Semua", "untuk": ["Butuh Dukungan", "Pasif"]},
    {"id": "M6", "judul": "Latihan Soal Kuis Adaptif", "tipe": "Kuis", "durasi": "40 mnt", "level": "Menengah", "untuk": ["Pejuang Deadline", "Deep Learner"]},
    {"id": "M7", "judul": "Studi Kelompok Terpandu", "tipe": "Komunitas", "durasi": "120 mnt", "level": "Semua", "untuk": ["Pasif", "Butuh Dukungan"]},
    {"id": "M8", "judul": "Deep Work untuk Skripsi", "tipe": "Modul", "durasi": "50 mnt", "level": "Lanjut", "untuk": ["Deep Learner"]},
]

app = FastAPI(title="SIAP API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- load artefak ---
def load_artifacts():
    try:
        rf = joblib.load(ART / "rf.pkl")
        km = joblib.load(ART / "kmeans.pkl")
        scaler = joblib.load(ART / "scaler.pkl")
        meta = json.loads((ART / "meta.json").read_text())
        return rf, km, scaler, meta
    except Exception:
        return None, None, None, {"version": "heuristic-v0", "accuracy": 0, "f1_macro": 0, "silhouette": 0}

RF, KM, SCALER, META = load_artifacts()
FEATS = META.get("features", ["semester", "ipk", "kehadiran_pct", "jam_belajar_minggu",
    "tugas_selesai_pct", "kuis_avg", "forum_aktif", "partisipasi", "beasiswa", "ekstrakurikuler", "sosek"])
SOS_MAP = {"rendah": 0, "menengah": 1, "tinggi": 2}

class PredictIn(BaseModel):
    nama: str = "Mahasiswa Baru"
    prodi: str = "Informatika"
    semester: int = Field(5, ge=1, le=14)
    ipk: float = Field(3.0, ge=0, le=4)
    kehadiran_pct: float = Field(80, ge=0, le=100)
    jam_belajar_minggu: float = Field(8, ge=0, le=40)
    tugas_selesai_pct: float = Field(75, ge=0, le=100)
    kuis_avg: float = Field(70, ge=0, le=100)
    forum_aktif: int = Field(5, ge=0)
    partisipasi: float = Field(6, ge=0, le=10)
    sosek: str = "menengah"
    beasiswa: int = 0
    ekstrakurikuler: int = 1

def featurize(p: PredictIn):
    row = {f: getattr(p, f, 0) for f in FEATS if f != "sosek"}
    row["sosek"] = SOS_MAP.get(p.sosek, 1)
    return np.array([[row[f] for f in FEATS]], dtype=float)

def heuristic(feats: np.ndarray):
    ipk, hadir, tugas = feats[0][1], feats[0][2], feats[0][4]
    skor = ipk * 20 + hadir * 0.4 + tugas * 0.2
    if skor > 155: return "Aman", 0.85
    if skor > 130: return "Waspada", 0.65
    return "At-Risk", 0.82

def init_db():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS predictions_log
        (id INTEGER PRIMARY KEY AUTOINCREMENT, nama TEXT, input_json TEXT,
         risiko TEXT, prob REAL, cluster TEXT, created_at TEXT)""")
    con.commit(); con.close()

init_db()

@app.get("/health")
def health():
    return {"status": "ok", "model_version": META.get("version"), "metrics": {
        "accuracy": META.get("accuracy"), "f1_macro": META.get("f1_macro"),
        "silhouette": META.get("silhouette")}}

def recommend_for(cluster_label: str, risiko: str):
    recs = [m for m in KATALOG if cluster_label in m["untuk"]]
    if risiko == "At-Risk":
        recs = [m for m in KATALOG if m["id"] in ("M5", "M2", "M4", "M7")] + recs
    seen, out = set(), []
    for m in recs:
        if m["id"] not in seen:
            out.append(m); seen.add(m["id"])
    return out[:5]

@app.post("/predict")
def predict(p: PredictIn):
    X = featurize(p)
    if RF is not None:
        Xs = SCALER.transform(X)
        risiko = str(RF.predict(Xs)[0])
        proba = float(RF.predict_proba(Xs).max())
        cl = int(KM.predict(Xs)[0])
    else:
        risiko, proba = heuristic(X)
        cl = 2 if risiko == "At-Risk" else 0
    cl_label = CLUSTER_LABELS[cl]
    headline = f"Mahasiswa {p.prodi} • Semester {p.semester} • {cl_label}"
    recs = recommend_for(cl_label, risiko)
    con = sqlite3.connect(DB)
    con.execute("INSERT INTO predictions_log (nama,input_json,risiko,prob,cluster,created_at) VALUES (?,?,?,?,?,?)",
                (p.nama, p.model_dump_json(), risiko, proba, cl_label, datetime.now().isoformat()))
    con.commit(); con.close()
    return {"nama": p.nama, "headline": headline, "risiko": risiko,
            "probabilitas": round(proba, 3), "cluster_id": cl,
            "cluster_label": cl_label, "rekomendasi": recs,
            "model_version": META.get("version")}

@app.get("/recommend/{nama}")
def recommend(nama: str, cluster: str = "Pejuang Deadline", risiko: str = "Waspada"):
    return {"nama": nama, "rekomendasi": recommend_for(cluster, risiko)}

@app.get("/profiles")
def profiles(limit: int = 24, q: Optional[str] = None, prodi: Optional[str] = None):
    if not DATA.exists():
        raise HTTPException(404, "dataset belum dibuat. Jalankan data/generator.py")
    df = pd.read_csv(DATA)
    if q: df = df[df["nama"].str.contains(q, case=False) | df["id"].str.contains(q, case=False)]
    if prodi and prodi != "Semua": df = df[df["prodi"] == prodi]
    return df.head(limit).to_dict(orient="records")

@app.get("/dashboard-stats")
def dashboard_stats():
    if not DATA.exists():
        return {"total": 0, "by_risiko": {}, "avg_ipk": 0, "avg_hadir": 0}
    df = pd.read_csv(DATA)
    return {"total": len(df), "by_risiko": df["risiko"].value_counts().to_dict(),
            "by_prodi": df["prodi"].value_counts().to_dict(),
            "avg_ipk": round(float(df["ipk"].mean()), 2),
            "avg_hadir": round(float(df["kehadiran_pct"].mean()), 1),
            "by_semester": df.groupby("semester")["ipk"].mean().round(2).to_dict()}
