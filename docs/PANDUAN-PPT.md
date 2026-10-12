# PANDUAN PPT — 12 Slide (tinggal isi screenshot)

> Sumber angka/teks: `docs/DOKUMENTASI.md`. Jangan mengarang angka — semua metrik ada di `backend/artifacts/meta.json`.

1. **Cover** — Judul + nama + NIM + foto dashboard. Slogan: "Profil ala LinkedIn, dibaca AI."
2. **Latar belakang** — DOKUMENTASI §2. Masalah: tahu bermasalah setelah nilai akhir.
3. **Rumusan masalah** — 3 pertanyaan (§3). Satu slide, tiga bullet.
4. **Konsep solusi** — Tabel LinkedIn vs SIAP (§5) + 4 persona. Ini slide pembeda.
5. **Dataset** — 1500 baris, distribusi label (Waspada 741 / At-Risk 496 / Aman 263), rata-rata IPK 3,06.
6. **Fitur** — Tabel 11 fitur (§6). Tekankan IPK + kehadiran + jam belajar.
7. **Model 1: RandomForest** — accuracy 0,9167, F1 0,9035, top-3 importance (IPK, jam belajar, kehadiran).
8. **Model 2+3: KMeans & Rekomendasi** — k=4, silhouette 0,1644 (jujur + jelaskan), katalog 8 materi.
9. **Arsitektur** — Diagram Streamlit → FastAPI → .pkl (§8) + tabel 5 endpoint.
10. **Demo** — Screenshot Beranda sebelum/sesudah prediksi + skenario 3 menit (§9). Siapkan video cadangan jika wifi mati.
11. **Hasil & keterbatasan** — Apa yang jalan + 4 rencana lanjut (§12). Dosen menilai kejujuran.
12. **Penutup** — Kesimpulan 3 baris + terima kasih + link GitHub + siap tanya-jawab (bekal: Glosarium §13).

**Checklist sebelum presentasi:**
- [ ] Backend `:8000/health` hijau, frontend `:8501` terbuka.
- [ ] Siapkan 2 profil contoh: bagus (IPK 3,6 → Aman) dan jelek (IPK 2,4 → At-Risk).
- [ ] Screenshot tiap tab disimpan di `docs/img/` bila demo live gagal.
- [ ] Hapus token GitHub yang pernah dipaste di chat (revoke).
