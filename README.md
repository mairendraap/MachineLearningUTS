# SIAP — Sistem Insight & Rekomendasi Akademik

Profil mahasiswa ala LinkedIn + ML (RandomForest + KMeans + Rekomendasi). Lihat `design.md` sebagai acuan tunggal.

## Struktur
```
├── design.md
├── data/generator.py + dataset_sintetik.csv (dibuat saat generate)
├── backend/ (FastAPI :8000) — main.py, train.py, artifacts/
└── frontend/ (Streamlit :8501) — app.py, assets/style.css
```

## Cara jalan (3 perintah)
```bash
# 1. install
pip install -r backend/requirements.txt -r frontend/requirements.txt

# 2. data + model
python data/generator.py
python backend/train.py

# 3a. backend
uvicorn backend.main:app --reload --port 8000
# 3b. frontend (terminal baru)
streamlit run frontend/app.py
```

Buka http://localhost:8501. Backend health: http://localhost:8000/health

## Demo 3 menit ke dosen
1. Tab Beranda → isi form (IPK 2.4, hadir 60%) → Analisis → muncul badge **At-Risk** + cluster.
2. Rekomendasi kanan berubah dinamis.
3. Tab Jelajah → filter prodi + search.
4. Tab Tentang Model → tunjukkan accuracy/F1/silhouette + feature importance.
