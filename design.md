# DESIGN.md — SIAP (Sistem Insight & Rekomendasi Akademik)
> Sumber kebenaran tunggal untuk Frontend + Backend + ML. Semua kode WAJIB ikut dokumen ini.

## 1. Konsep
Aplikasi analisis performa akademik dengan **profil mahasiswa ala LinkedIn** + prediksi ML + rekomendasi.
Bukan dashboard admin kaku. Tiap mahasiswa = profil profesional yang hidup dan dinamis.

Analoginya:
- LinkedIn punya Profil → Koneksi → Feed → Jobs. SIAP punya:
- Profil Mahasiswa → Cluster (komunitas belajar) → Feed Rekomendasi → Prediksi Risiko.

## 2. Data Model (ala LinkedIn)

```python
StudentProfile {
  id: str            # "UTS-2024-001"
  nama: str
  headline: str      # auto: "Mahasiswa Informatika • Semester 5 • Cluster Pejuang Deadline"
  avatar_url: str    # dicebear / ui-avatars
  about: str
  education: [{ kampus, prodi, semester, ipk, status }]
  activity: [{ tanggal, tipe: LMS/kehadiran/tugas, skor }]  # time-series ringkas
  skills: [{ nama: "Python", level: 0-100, endorsement: int }]
  stats: { ipk, kehadiran_pct, jam_belajar_minggu, tugas_selesai_pct, kuis_avg, forum_aktif }
  ml: { risiko: Aman|Waspada|At-Risk, probabilitas: 0-1, cluster_id: 0-3, cluster_label: str }
}
```

Cluster (4 persona, jadi filter dinamis):
- 0 `Deep Learner` — rajin, stabil
- 1 `Pejuang Deadline` — kejar tayang, kuis bagus, tugas telat
- 2 `Butuh Dukungan` — kehadiran + nilai turun
- 3 `Pasif` — jarang aktif LMS

Tabel DB (SQLite `siap.db`):
- `profiles(id, nama, prodi, semester, ipk, kehadiran, jam_belajar, tugas_pct, kuis_avg, forum, sosek, risiko, prob, cluster)`
- `predictions_log(id, profile_id, input_json, risiko, prob, created_at)`

## 3. Arsitektur

```
[ Streamlit :8501 ] -- REST JSON --> [ FastAPI :8000 ] --> [ sklearn .pkl ]
      |                                      |
  LinkedIn-style UI                    SQLite siap.db
  tidak ada logika ML di UI            satu-satunya tempat inference
```

Endpoints (kontrak baku):
- `GET /health` → `{status, model_version}`
- `POST /predict` → input stats → `{risiko, probabilitas, cluster, cluster_label, headline, rekomendasi[]}`
- `GET /recommend/{profile_id}` → list materi
- `GET /profiles` + `GET /profiles/{id}` → data profil
- `GET /dashboard-stats` → agregat untuk chart
- `POST /profiles/batch` → upload CSV batch

Aturan: Frontend TIDAK boleh import sklearn. Semua prediksi via API. Fallback heuristik hanya jika backend mati (tampilkan badge "offline mode").

## 4. Visual Design System (Anti AI-Slop)

Terinspirasi LinkedIn, bukan gradient ungu generik.

**Palette (wajib hex ini):**
- `--bg: #F4F2EE` (abu hangat background)
- `--card: #FFFFFF`
- `--ink: #191919` (teks utama)
- `--muted: #666666`
- `--line: #E8E5DF` (border kartu)
- `--brand: #0A66C2` (biru LinkedIn, CTA saja)
- `--brand-dark: #004182`
- `--ok: #057642` / `--warn: #B54708` / `--danger: #C92A2A`
- DILARANG: gradient purple-blue, glassmorphism, emoji berlebihan, dark-mode default.

**Typography:**
- Font: `Inter, -apple-system, Segoe UI, sans-serif` via Google Fonts.
- H1 22px/700, H2 16px/600, Body 14px/400, Caption 12px muted.
- Angka metrik pakai tabular-nums.

**Layout (responsive):**
- Topbar putih sticky 56px: logo kotak biru "S" + search + nav (Beranda, Jaringan Cluster, Notifikasi).
- Container max 1128px, 3 kolom di desktop `240px | 1fr | 300px`, collapse jadi 1 kolom di <900px.
- Kartu: `bg white, border 1px var(--line), radius 8px, padding 16px, shadow none`. LinkedIn tidak pakai shadow besar.
- Left rail: kartu profil mini (banner abu + avatar 72px overlap + headline + stats).
- Center: composer "Analisis mahasiswa baru…" + kartu hasil prediksi + feed aktivitas.
- Right rail: "Rekomendasi untuk Anda" + "Cluster serupa".

**Komponen:**
- Badge risiko: pill 12px, `Aman=hijau muda, Waspada=kuning muda, At-Risk=merah muda`. Jangan pakai warna solid mencolok.
- Skill bar: track `#E8E5DF` 6px, fill `#0A66C2`.
- Button: Primary filled `#0A66C2` radius 16px (pill), Secondary outline. Tidak ada gradient.
- Chart: Plotly, template `plotly_white`, warna seri `["#0A66C2","#057642","#B54708","#C92A2A"]`, grid tipis.

**Motion (dinamis tapi kalem):**
- Hover kartu: `border-color #0A66C2` saja, tanpa scale/animasi melenting.
- Streamlit `st.tabs`, `st.expander`, `session_state` untuk filter semester/prodi/cluster tanpa reload.
- Skeleton tidak perlu; pakai `st.spinner` + `st.toast` saat hit API.

## 5. Halaman Frontend (Streamlit `app.py`)

Satu app, navigasi via top tabs (biar responsif di HP):
1. **Beranda** — layout 3 kolom LinkedIn. Form input kiri/center, hasil prediksi center, rekomendasi kanan.
2. **Jelajah Profil** — grid kartu profil + filter cluster/prodi + search nama. Klik → drawer detail (expander).
3. **Upload Batch** — upload CSV → POST batch → tabel hasil + download.
4. **Tentang Model** — metrik (accuracy/F1/silhouette), feature importance, penjelasan.

Responsive: pakai `st.columns([1,2,1])` di desktop, otomatis stack di mobile. CSS media query di `assets/style.css` untuk topbar & hide right-rail <900px.

## 6. ML Pipeline

Fitur input (11): `semester, ipk, kehadiran_pct, jam_belajar, tugas_pct, kuis_avg, forum_aktif, partisipasi, sosek, beasiswa, ekstrakurikuler`.
- Preprocess: StandardScaler + OneHot prodi (atau label sederhana).
- Model A: `RandomForestClassifier(n=200)` → risiko 3 kelas. Target dibuat dari aturan + noise agar realistis.
- Model B: `KMeans(k=4, n_init=10)` → cluster persona di atas.
- Model C: cosine similarity skill-gap → mapping ke katalog 12 materi (id, judul, tipe, durasi, level, link).
- Artefak: `backend/artifacts/{rf.pkl, kmeans.pkl, scaler.pkl, meta.json}`. Versi `v1.0.0`.

## 7. Definisi Selesai
- `design.md` ini dipatuhi (warna, radius, endpoint sama).
- Backend `uvicorn main:app` hijau di :8000, frontend `streamlit run app.py` di :8501.
- Demo 3 menit jalan offline dengan data sintetik tanpa internet kecuali font/avatar.
