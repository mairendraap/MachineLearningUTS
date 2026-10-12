"""Generator dataset sintetik akademik — kompleks, tabular + temporal ringkas."""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 1500

prodi = RNG.choice(["Informatika", "Sistem Informasi", "Teknik Elektro", "Manajemen"], N)
semester = RNG.integers(1, 9, N)
beasiswa = RNG.choice([0, 1], N, p=[0.7, 0.3])
sosek = RNG.choice(["rendah", "menengah", "tinggi"], N, p=[0.3, 0.5, 0.2])
ekstra = RNG.integers(0, 4, N)

# base laten: rajin vs pasif
rajin = RNG.normal(0.5, 0.2, N).clip(0, 1)
ipk = (2.2 + rajin * 1.7 + RNG.normal(0, 0.15, N)).clip(2.0, 4.0).round(2)
kehadiran = (55 + rajin * 43 + RNG.normal(0, 5, N)).clip(40, 100).round(1)
jam_belajar = (rajin * 18 + RNG.normal(0, 2, N)).clip(0, 25).round(1)
tugas_pct = (50 + rajin * 50 + RNG.normal(0, 6, N)).clip(20, 100).round(1)
kuis_avg = (45 + rajin * 55 + RNG.normal(0, 7, N)).clip(20, 100).round(1)
forum = (rajin * 25 + RNG.normal(0, 4, N)).clip(0, 40).astype(int)
partisipasi = (rajin * 10 + RNG.normal(0, 1.5, N)).clip(0, 10).round(1)

# label risiko dari aturan + noise (biar tidak trivial)
skor = (ipk * 20 + kehadiran * 0.4 + tugas_pct * 0.2 + kuis_avg * 0.15 + jam_belajar)
risiko = np.where(skor > 145, "Aman", np.where(skor > 120, "Waspada", "At-Risk"))
# 8% label flip untuk realisme
flip = RNG.random(N) < 0.08
risiko[flip] = RNG.choice(["Aman", "Waspada", "At-Risk"], flip.sum())

df = pd.DataFrame({
    "id": [f"UTS-2024-{i:04d}" for i in range(1, N + 1)],
    "nama": [f"Mahasiswa {i}" for i in range(1, N + 1)],
    "prodi": prodi, "semester": semester, "ipk": ipk,
    "kehadiran_pct": kehadiran, "jam_belajar_minggu": jam_belajar,
    "tugas_selesai_pct": tugas_pct, "kuis_avg": kuis_avg,
    "forum_aktif": forum, "partisipasi": partisipasi,
    "sosek": sosek, "beasiswa": beasiswa, "ekstrakurikuler": ekstra,
    "risiko": risiko,
})
df.to_csv("data/dataset_sintetik.csv", index=False)
print(f"OK: {len(df)} baris -> data/dataset_sintetik.csv")
print(df["risiko"].value_counts())
