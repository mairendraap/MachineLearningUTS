"""SIAP Frontend — Streamlit LinkedIn-style. Logika ML hanya via FastAPI (lihat design.md)."""
import json
from pathlib import Path
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

API = "http://localhost:8000"
st.set_page_config(page_title="SIAP — Insight Akademik", page_icon="🎓", layout="wide")

CSS = (Path(__file__).parent / "assets" / "style.css").read_text()
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
st.markdown(
    """<div class="siap-topbar"><div class="siap-logo">S</div>
    <div class="siap-search">🔍&nbsp; Cari mahasiswa, materi, cluster…</div>
    <div class="siap-nav"><span><b>Beranda</b></span><span>Jaringan Cluster</span><span>Notifikasi</span></div></div>""",
    unsafe_allow_html=True,
)

def api_get(path, **kw):
    try:
        r = requests.get(f"{API}{path}", timeout=5, **kw)
        r.raise_for_status()
        return r.json(), True
    except Exception:
        return None, False

def api_post(path, payload):
    try:
        r = requests.post(f"{API}{path}", json=payload, timeout=10)
        r.raise_for_status()
        return r.json(), True
    except Exception as e:
        return {"error": str(e)}, False

def pill(risiko: str) -> str:
    cls = {"Aman": "pill-ok", "Waspada": "pill-warn", "At-Risk": "pill-danger"}.get(risiko, "pill-warn")
    return f'<span class="siap-pill {cls}">{risiko}</span>'

def avatar(nama: str) -> str:
    return f"https://ui-avatars.com/api/?name={nama.replace(' ','+')}&background=0A66C2&color=fff&size=128"

tab1, tab2, tab3, tab4 = st.tabs(["🏠 Beranda", "🧑‍🎓 Jelajah Profil", "📤 Upload Batch", "📊 Tentang Model"])

# ============ TAB 1: BERANDA ============
with tab1:
    col_l, col_c, col_r = st.columns([1, 2, 1])

    with col_l:
        st.markdown(f"""<div class="siap-card"><div class="siap-banner"></div>
        <img class="siap-avatar" src="{avatar('SIAP User')}"/>
        <div class="siap-name">Operator Akademik</div>
        <div class="siap-headline">Fakultas Teknologi • UTS</div>
        <hr/><div class="siap-cap">Total mahasiswa</div>
        <div class="siap-metric" id="m-total">—</div></div>""", unsafe_allow_html=True)
        stats, ok = api_get("/dashboard-stats")
        if ok:
            st.markdown(f"""<div class="siap-card"><div class="siap-cap">Rata-rata IPK</div>
            <div class="siap-metric">{stats['avg_ipk']}</div>
            <div class="siap-cap">Kehadiran {stats['avg_hadir']}% • {stats['total']} profil</div></div>""",
                        unsafe_allow_html=True)
            fig = px.pie(names=list(stats["by_risiko"].keys()), values=list(stats["by_risiko"].values()),
                         color_discrete_sequence=["#057642", "#B54708", "#C92A2A"], hole=0.5)
            fig.update_layout(margin=dict(t=10, b=10), height=220, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

    with col_c:
        st.markdown('<div class="siap-card"><b>✍️ Analisis mahasiswa baru…</b><div class="siap-cap">Isi form, klik Analisis. Hasil muncul dinamis di bawah.</div></div>',
                    unsafe_allow_html=True)
        with st.form("prediksi"):
            c1, c2 = st.columns(2)
            nama = c1.text_input("Nama", "Ayu Lestari")
            prodi = c1.selectbox("Prodi", ["Informatika", "Sistem Informasi", "Teknik Elektro", "Manajemen"])
            semester = c1.slider("Semester", 1, 8, 5)
            sosek = c1.selectbox("Sosial-ekonomi", ["rendah", "menengah", "tinggi"], index=1)
            ipk = c2.slider("IPK", 2.0, 4.0, 3.1, 0.05)
            hadir = c2.slider("Kehadiran %", 40, 100, 72)
            jam = c2.slider("Jam belajar/minggu", 0, 25, 6)
            c3, c4 = st.columns(2)
            tugas = c3.slider("Tugas selesai %", 20, 100, 65)
            kuis = c3.slider("Rata-rata kuis", 20, 100, 68)
            forum = c4.number_input("Aktivitas forum", 0, 40, 4)
            part = c4.slider("Partisipasi (0-10)", 0.0, 10.0, 5.5)
            submit = st.form_submit_button("🔍 Analisis Sekarang", use_container_width=True)

        if submit:
            payload = dict(nama=nama, prodi=prodi, semester=semester, ipk=ipk,
                           kehadiran_pct=hadir, jam_belajar_minggu=jam, tugas_selesai_pct=tugas,
                           kuis_avg=kuis, forum_aktif=int(forum), partisipasi=float(part),
                           sosek=sosek, beasiswa=0, ekstrakurikuler=1)
            with st.spinner("Model memprediksi…"):
                res, ok = api_post("/predict", payload)
            if not ok:
                st.warning("Backend mati — tampil mode offline (heuristik). Jalankan `uvicorn main:app` di backend/.")
                skor = ipk * 20 + hadir * 0.4 + tugas * 0.2
                risiko = "Aman" if skor > 155 else ("Waspada" if skor > 130 else "At-Risk")
                res = {"nama": nama, "headline": f"Mahasiswa {prodi} • Semester {semester} • —",
                       "risiko": risiko, "probabilitas": 0.7, "cluster_label": "Butuh Dukungan" if risiko == "At-Risk" else "Pejuang Deadline",
                       "rekomendasi": []}
            st.session_state["hasil"] = res
            st.toast(f"Hasil: {res['risiko']} ({res['probabilitas']})", icon="🎯")

        h = st.session_state.get("hasil")
        if h:
            st.markdown(f"""<div class="siap-card"><div style="display:flex;gap:12px;align-items:center">
            <img class="siap-avatar" style="margin:0;width:56px;height:56px" src="{avatar(h['nama'])}"/>
            <div><div class="siap-name">{h['nama']}</div>
            <div class="siap-headline">{h.get('headline','')}</div></div>
            <div style="margin-left:auto">{pill(h['risiko'])}</div></div>
            <hr/><div class="siap-cap">Probabilitas • Cluster</div>
            <div class="siap-metric">{h['probabilitas']} • {h.get('cluster_label','—')}</div></div>""",
                        unsafe_allow_html=True)
            if h.get("rekomendasi"):
                st.markdown('<div class="siap-card"><b>📚 Rekomendasi untuk profil ini</b></div>', unsafe_allow_html=True)
                for m in h["rekomendasi"]:
                    st.markdown(f"""<div class="siap-card rec-row"><div class="rec-badge">{m['id']}</div>
                    <div><b>{m['judul']}</b><br/><span class="siap-cap">{m['tipe']} • {m['durasi']} • {m['level']}</span></div></div>""",
                                unsafe_allow_html=True)

    with col_r:
        st.markdown("""<div class="siap-card"><b>✨ Rekomendasi untuk Anda</b>
        <div class="siap-cap">Berdasarkan cluster & risiko, diperbarui dinamis tiap prediksi.</div></div>""",
                    unsafe_allow_html=True)
        h = st.session_state.get("hasil", {})
        recs = h.get("rekomendasi", [
            {"id": "M1", "judul": "Manajemen Waktu & Pomodoro", "tipe": "Modul", "durasi": "45 mnt", "level": "Dasar"},
            {"id": "M3", "judul": "Python untuk Analisis Data", "tipe": "Praktik", "durasi": "90 mnt", "level": "Menengah"},
            {"id": "M6", "judul": "Latihan Soal Kuis Adaptif", "tipe": "Kuis", "durasi": "40 mnt", "level": "Menengah"},
        ])
        for m in recs[:4]:
            st.markdown(f"""<div class="siap-card rec-row"><div class="rec-badge">▶</div>
            <div><b>{m['judul']}</b><br/><span class="siap-cap">{m['tipe']} • {m['durasi']}</span></div></div>""",
                        unsafe_allow_html=True)
        st.markdown("""<div class="siap-card"><b>👥 Cluster serupa</b><div class="siap-cap">
        Deep Learner • Pejuang Deadline • Butuh Dukungan • Pasif — lihat tab Jelajah Profil.</div></div>""",
                    unsafe_allow_html=True)

# ============ TAB 2: JELAJAH ============
with tab2:
    f1, f2, f3 = st.columns([2, 1, 1])
    q = f1.text_input("🔍 Cari nama / NIM", "")
    prodi_f = f2.selectbox("Prodi", ["Semua", "Informatika", "Sistem Informasi", "Teknik Elektro", "Manajemen"])
    lim = f3.slider("Tampilkan", 6, 48, 12)
    data, ok = api_get("/profiles", params={"q": q, "prodi": prodi_f, "limit": lim})
    if not ok:
        st.error("Backend belum jalan. Jalankan backend dulu: `uvicorn main:app --reload --port 8000`")
    elif not data:
        st.info("Tidak ada hasil.")
    else:
        cols = st.columns(3)
        for i, p in enumerate(data):
            with cols[i % 3]:
                st.markdown(f"""<div class="siap-card"><div class="siap-banner"></div>
                <img class="siap-avatar" src="{avatar(p['nama'])}"/>
                <div class="siap-name">{p['nama']}</div>
                <div class="siap-headline">{p['id']} • {p['prodi']} • Sem {p['semester']}</div>
                <div style="margin:8px 0">{pill(p['risiko'])}</div>
                <div class="siap-cap">IPK {p['ipk']} • Hadir {p['kehadiran_pct']}% • Tugas {p['tugas_selesai_pct']}%</div>
                <div class="skill-track"><div class="skill-fill" style="width:{min(100, p['kuis_avg'])}%"></div></div>
                <div class="siap-cap">Kuis avg {p['kuis_avg']}</div></div>""", unsafe_allow_html=True)
                with st.expander("Lihat detail"):
                    st.json({k: p[k] for k in ("id", "prodi", "semester", "ipk", "kehadiran_pct",
                             "jam_belajar_minggu", "tugas_selesai_pct", "kuis_avg", "risiko")})

# ============ TAB 3: BATCH ============
with tab3:
    st.markdown('<div class="siap-card"><b>📤 Prediksi massal via CSV</b><div class="siap-cap">Kolom wajib: nama, prodi, semester, ipk, kehadiran_pct, jam_belajar_minggu, tugas_selesai_pct, kuis_avg, forum_aktif, partisipasi, sosek. <a href="#" >Unduh contoh</a> dari data/dataset_sintetik.csv</div></div>',
                unsafe_allow_html=True)
    f = st.file_uploader("Upload CSV", type="csv")
    if f:
        df = pd.read_csv(f)
        st.dataframe(df.head(10), use_container_width=True)
        if st.button("🚀 Prediksi semua", use_container_width=True):
            hasil = []
            prog = st.progress(0)
            for i, row in df.head(50).iterrows():
                res, _ = api_post("/predict", row.fillna(0).to_dict())
                if "risiko" in res:
                    hasil.append({"nama": row.get("nama"), "risiko": res["risiko"],
                                  "prob": res["probabilitas"], "cluster": res["cluster_label"]})
                prog.progress((i + 1) / min(50, len(df)))
            out = pd.DataFrame(hasil)
            st.success(f"Selesai: {len(out)} baris diprediksi.")
            st.dataframe(out, use_container_width=True)
            st.download_button("⬇️ Unduh hasil", out.to_csv(index=False), "hasil_prediksi.csv")

# ============ TAB 4: MODEL ============
with tab4:
    health, ok = api_get("/health")
    meta_path = Path(__file__).resolve().parent.parent / "backend" / "artifacts" / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else (health or {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", meta.get("accuracy", "—"))
    c2.metric("F1 Macro", meta.get("f1_macro", "—"))
    c3.metric("Silhouette (KMeans)", meta.get("silhouette", "—"))
    st.markdown("""<div class="siap-card"><b>🧠 Pipeline</b><div class="siap-cap">
    RandomForest (200 trees) untuk klasifikasi risiko 3 kelas → KMeans (k=4) untuk persona →
    Cosine-similarity katalog untuk rekomendasi. Preprocessing StandardScaler. Split 80/20 stratified.</div></div>""",
                unsafe_allow_html=True)
    if meta.get("importance"):
        imp = pd.DataFrame(list(meta["importance"].items()), columns=["fitur", "penting"]).sort_values("penting")
        st.plotly_chart(px.bar(imp, x="penting", y="fitur", orientation="h",
                               color_discrete_sequence=["#0A66C2"], template="plotly_white"), use_container_width=True)
    st.caption(f"Model version: {meta.get('version','?')} • Backend: {'online' if ok else 'offline'} • Kontrak API sesuai design.md")
