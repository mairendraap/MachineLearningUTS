# DOKUMENTASI LENGKAP — SIAP (Sistem Insight & Rekomendasi Akademik)

> Dokumen ini bahan mentah untuk PPT. Teman yang bikin slide tinggal salin per bagian.
> Versi model: `v1.0.0` • Lihat juga `design.md` (aturan desain) dan `README.md` (cara jalan).

---

## 1. Judul & Identitas

- **Judul:** SIAP — Sistem Insight & Rekomendasi Akademik Berbasis Machine Learning
- **Mata kuliah:** Machine Learning (UTS)
- **Jenis aplikasi:** Fullstack — Frontend Streamlit + Backend FastAPI + model scikit-learn
- **Slogan slide:** "Profil mahasiswa ala LinkedIn, dibaca oleh AI."

## 2. Latar Belakang (untuk slide 2)

Kampus punya banyak data akademik (IPK, kehadiran, tugas, aktivitas LMS) tapi jarang dipakai untuk **peringatan dini**. Dosen wali biasanya tahu mahasiswa bermasalah setelah nilai akhir keluar — sudah terlambat.

SIAP menjawab: bagaimana kalau data itu dibaca ML sejak awal semester, lalu sistem otomatis memberi **label risiko + saran belajar yang personal**?

## 3. Rumusan Masalah (untuk slide 3)

1. Bagaimana memprediksi status risiko mahasiswa (Aman / Waspada / At-Risk) dari data akademik?
2. Bagaimana mengelompokkan mahasiswa ke persona belajar agar intervensi tepat sasaran?
3. Bagaimana memberi rekomendasi materi yang relevan dengan profil tiap mahasiswa?

## 4. Tujuan & Manfaat

- **Tujuan:** membangun aplikasi prediksi + rekomendasi yang bisa didemo live.
- **Manfaat dosen wali:** daftar prioritas bimbingan (fokus ke At-Risk dulu).
- **Manfaat mahasiswa:** tahu posisi diri + dapat saran konkret (bukan sekadar "belajar yang rajin").
- **Manfaat akademik:** arsitektur frontend-backend terpisah yang rapi dan bisa dikembangkan.

## 5. Konsep Solusi (analogi LinkedIn — untuk slide 4)

| LinkedIn | SIAP | Maksud |
|---|---|---|
| Profil profesional | Profil mahasiswa (foto, headline, IPK, skill-bar) | Data yang kaku jadi hidup dan mudah dibaca |
| Koneksi | Cluster persona (4 tipe) | Mahasiswa dikelompokkan ke "komunitas" sejenis |
| Feed | Feed rekomendasi materi | Tiap profil dapat saran berbeda |
| Jobs | Prediksi risiko | "Peluang bermasalah" diprediksi di awal |

Empat persona cluster:
- **Deep Learner** — rajin dan stabil.
- **Pejuang Deadline** — pintar tapi kejar tayang.
- **Butuh Dukungan** — nilai + kehadiran turun, prioritas bimbingan.
- **Pasif** — jarang aktif di LMS.

## 6. Dataset (untuk slide 5–6)

- **Jenis:** tabular numerik + kategorikal (kompleks: 1500 baris × 15 kolom, ada missing pattern, imbalance, fitur temporal ringkas).
- **Sumber:** sintetik meniru pola publik UCI Student Performance / OULAD, dibuat oleh `data/generator.py` (seed 42, reproducible).
- **Komposisi label:** Waspada 741 • At-Risk 496 • Aman 263. Total 1500.
- **Prodi:** Manajemen 385, Teknik Elektro 377, Sistem Informasi 371, Informatika 367.
- **Rata-rata:** IPK 3,06 • kehadiran 76,8%.

Fitur input model (11):

| Fitur | Contoh | Kenapa dipakai |
|---|---|---|
| semester, prodi, sosek, beasiswa, ekstrakurikuler | Sem 5, Informatika | konteks latar belakang |
| ipk | 3,10 | sinyal akademik terkuat |
| kehadiran_pct | 72% | sinyal kedisiplinan |
| jam_belajar_minggu | 6 jam | sinyal usaha |
| tugas_selesai_pct, kuis_avg | 65 / 68 | sinyal konsistensi |
| forum_aktif, partisipasi | 4 / 5,5 | sinyal keterlibatan LMS |

Label target dibuat dari rumus skor + 8% noise agar realistis (tidak 100% bisa ditebak):
`skor = ipk×20 + kehadiran×0,4 + tugas×0,2 + kuis×0,15 + jam_belajar` → Aman jika >145, Waspada jika >120, selebihnya At-Risk.

## 7. Metode Machine Learning (untuk slide 7–9, bagian terpenting)

Pipeline 3 lapis — ini yang bikin nilai plus karena bukan cuma 1 model:

### Lapis 1 — Klasifikasi risiko: RandomForestClassifier (200 trees)
- **Kenapa Random Forest?** Tahan terhadap outlier dan fitur campuran, tidak perlu scaling ketat, memberi probabilitas + feature importance (bisa dijelaskan ke dosen).
- **Setting:** 200 pohon, split 80/20 stratified, `random_state=42`.
- **Hasil asli (tersimpan di `backend/artifacts/meta.json`):** accuracy **0,9167**, F1-macro **0,9035**.
- **Feature importance tertinggi:** IPK (0,26), jam belajar (0,19), kehadiran (0,19), tugas (0,13), kuis (0,09). Artinya model "berpikir" mirip dosen: IPK + kehadiran + usaha.
- **Cara baca metrik untuk PPT:** accuracy = dari 100 mahasiswa, ~92 tebakannya benar. F1-macro = akurasi yang adil walau jumlah Aman lebih sedikit.

### Lapis 2 — Segmentasi: KMeans (k=4)
- **Kenapa KMeans?** Unsupervised — menemukan pola tanpa diberi label, cocok untuk persona.
- **Setting:** k=4, `n_init=10`, input sudah di-StandardScaler.
- **Hasil:** silhouette **0,1644** (rendah tapi wajar untuk data perilaku yang tumpang tindih; cluster dipakai sebagai persona praktis, bukan kebenaran mutlak).
- **Cara presentasi:** "angkaév kecil tidak masalah — yang penting 4 kelompoknya bisa dijelaskan dan dipakai untuk rekomendasi berbeda."

### Lapis 3 — Rekomendasi: Content-Based Filtering (cosine similarity + aturan)
- **Cara kerja:** tiap cluster dipetakan ke katalog 8 materi (modul/video/kuis/aksi). Mahasiswa At-Risk selalu diprioritaskan materi intervensi (konsultasi dosen wali, remedial, studi kelompok).
- **Kenapa bukan collaborative filtering?** Data rating antar-mahasiswa belum ada; content-based bisa jalan sejak hari pertama (cold-start friendly).

### Diagram pipeline (gambar ulang di slide)
```
Data CSV → Cleaning → StandardScaler → [RandomForest → risiko + probabilitas]
                                     → [KMeans → cluster persona]
                                     → [Aturan katalog → 5 rekomendasi]
                                     → Profil LinkedIn + SQLite log
```

## 8. Arsitektur Sistem (untuk slide 10)

```
[Streamlit :8501] --REST JSON--> [FastAPI :8000] --> [model .pkl]
        |                                 |
  UI ala LinkedIn                   SQLite siap.db (log prediksi)
  DILARANG import sklearn           SATU-SATUNYA tempat inference
```

- **Kenapa dipisah?** Frontend boleh ganti (misal jadi React) tanpa menyentuh model. Backend boleh di-scale terpisah. Ini arsitektur "beneran", bukan notebook dibungkus UI.
- **Kontrak API:**

| Method & Endpoint | Fungsi |
|---|---|
| `GET /health` | cek status + versi model + metrik |
| `POST /predict` | kirim 11 fitur → terima risiko, probabilitas, cluster, headline, rekomendasi |
| `GET /recommend/{nama}` | rekomendasi untuk profil tertentu |
| `GET /profiles` | daftar profil + filter search/prodi |
| `GET /dashboard-stats` | agregat dashboard (total, rata-rata, distribusi) |

## 9. Alur Pengguna & Alur Data (untuk slide 11)

**Alur demo 3 menit (hafalkan ini):**
1. Buka Beranda → isi form (contoh jelek: IPK 2,4, hadir 60%) → klik Analisis.
2. Kartu hasil muncul: badge merah **At-Risk 0,95** + cluster **Butuh Dukungan**.
3. Kolom kanan berubah: muncul "Konsultasi Dosen Wali + Remedial Kalkulus".
4. Pindah ke Jelajah Profil → filter prodi → klik expander detail.
5. Tutup di Tentang Model → tunjukkan accuracy 0,92 + diagram feature importance.

**Alur data (untuk pertanyaan dosen):**
Input form → JSON → `POST /predict` → scaler → RF + KMeans → rekomendasi → simpan ke SQLite → tampil di UI. Frontend tidak pernah menyentuh file `.pkl` langsung.

## 10. Tampilan Aplikasi (untuk slide 12–13, screenshot)

1. **Beranda (3 kolom LinkedIn):** kiri kartu operator + pie risiko, tengah form + kartu hasil, kanan rekomendasi + cluster serupa.
2. **Jelajah Profil:** grid kartu + search + filter prodi + expander detail JSON.
3. **Upload Batch:** upload CSV → prediksi massal (maks 50/baris demo) → unduh hasil.
4. **Tentang Model:** metrik + pipeline + bar chart feature importance.

Desain ikut `design.md`: background `#F4F2EE`, kartu putih radius 8px, biru `#0A66C2` hanya untuk CTA, tanpa gradient ungu.

## 11. Cara Menjalankan (untuk lampiran, bukan slide utama)

```bash
git clone https://github.com/mairendraap/MachineLearningUTS.git
cd MachineLearningUTS
pip install -r backend/requirements.txt -r frontend/requirements.txt
python data/generator.py && python backend/train.py   # opsional, artefak sudah ikut repo
uvicorn backend.main:app --reload --port 8000         # terminal 1
streamlit run frontend/app.py                          # terminal 2 → localhost:8501
```

## 12. Keterbatasan & Rencana Lanjut (untuk slide penutup — dosen suka ini)

- Data sintetik, bukan data kampus asli → langkah lanjut: pakai data LMS/PD-Dikti anonim.
- Silhouette KMeans rendah → coba DBSCAN / Gaussian Mixture + elbow analysis.
- SQLite + file `.pkl` → migrasi ke PostgreSQL + MLflow untuk produksi.
- Tambahan: prediksi temporal (LSTM atas aktivitas mingguan) dan chatbot penasihat.

## 13. Glosarium (bekal tanya-jawab)

- **Accuracy:** proporsi tebakan benar. **F1-macro:** rata-rata adil antar kelas tak seimbang.
- **Silhouette:** seberapa rapat anggota dalam satu cluster (0–1, makin tinggi makin terpisah).
- **Random Forest:** voting ratusan decision tree agar tidak overfit.
- **KMeans:** mengelompokkan titik terdekat ke k titik tengah.
- **Content-based filtering:** merekomendasikan barang mirip profil, bukan mirip pengguna lain.
- **StandardScaler:** mengubah fitur ke skala sebanding (mean 0) agar KMeans adil.
- **Stratified split:** bagi data latih/uji dengan proporsi label tetap seimbang.
