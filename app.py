from datetime import datetime

import pandas as pd
import streamlit as st

from simulator import (
    build_distribution_from_history,
    build_distribution_from_table,
    build_monte_carlo_interval_table,
    detect_dataset_columns,
    export_results_to_excel,
    load_demand_distribution,
    normalize_distribution,
    read_uploaded_dataset,
    run_monte_carlo,
    simulate_one_run,
    suggest_inventory_parameters,
    validate_distribution,
)

st.set_page_config(
    page_title="Simulasi Persediaan Monte Carlo",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Gaya elegan & minimalis
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }
        h1 {
            font-weight: 700 !important;
            letter-spacing: -0.025em !important;
            color: #0F172A !important;
            margin-bottom: 0.2rem !important;
        }
        h2, h3 {
            font-weight: 600 !important;
            letter-spacing: -0.015em !important;
            color: #0F172A !important;
        }
        .subtitle-text {
            color: #64748B;
            font-size: 0.98rem;
            margin-bottom: 1.5rem;
        }
        .step-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 1rem 1.15rem;
            height: 100%;
        }
        .step-num {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #64748B;
            margin-bottom: 0.25rem;
        }
        .step-title {
            font-size: 0.95rem;
            font-weight: 600;
            color: #0F172A;
            margin-bottom: 0.25rem;
        }
        .step-desc {
            font-size: 0.84rem;
            color: #475569;
            line-height: 1.45;
            margin: 0;
        }
        .section-label {
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: #64748B;
            margin-top: 1.6rem;
            margin-bottom: 0.15rem;
        }
        .section-heading {
            font-size: 1.25rem;
            font-weight: 600;
            color: #0F172A;
            margin-bottom: 0.65rem;
        }
        [data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 0.9rem 1rem;
        }
        [data-testid="stMetricLabel"] {
            color: #64748B !important;
            font-size: 0.82rem !important;
        }
        [data-testid="stMetricValue"] {
            color: #0F172A !important;
            font-weight: 600 !important;
            font-size: 1.45rem !important;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid #E2E8F0;
        }
        .stButton > button[kind="primary"] {
            background-color: #0F172A;
            color: #FFFFFF;
            border: 1px solid #0F172A;
            border-radius: 8px;
            font-weight: 500;
            padding: 0.55rem 1rem;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #1E293B;
            border-color: #1E293B;
        }
        .stButton > button {
            border-radius: 8px;
        }
        .status-box {
            border-radius: 10px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
            border: 1px solid;
        }
        .status-safe {
            background-color: #F0FDF4;
            border-color: #BBF7D0;
            color: #166534;
        }
        .status-warn {
            background-color: #FFFBEB;
            border-color: #FDE68A;
            color: #92400E;
        }
        .status-risk {
            background-color: #FEF2F2;
            border-color: #FECACA;
            color: #991B1B;
        }
        .status-title {
            font-weight: 600;
            font-size: 0.95rem;
            margin-bottom: 0.25rem;
        }
        .status-body {
            font-size: 0.88rem;
            line-height: 1.5;
            margin: 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def section_header(step_label, title):
    st.markdown(
        f'<div class="section-label">{step_label}</div>'
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True,
    )


# Terapkan parameter rekomendasi sebelum widget sidebar dibuat
if "pending_params" in st.session_state:
    params_to_apply = st.session_state.pop("pending_params")
    for k, v in params_to_apply.items():
        st.session_state[k] = v

# Inisialisasi nilai default parameter sidebar
DEFAULT_PARAMS = {
    "initial_stock": 100,
    "reorder_point": 30,
    "order_quantity": 80,
    "lead_time": 3,
    "days": 30,
    "replications": 500,
    "seed": 42,
}
for key, default_val in DEFAULT_PARAMS.items():
    if key not in st.session_state:
        st.session_state[key] = default_val

if "sim_history" not in st.session_state:
    st.session_state["sim_history"] = []
if "active_view" not in st.session_state:
    st.session_state["active_view"] = "simulasi"


def render_simulation_result(res, key_prefix="main"):
    """Tampilkan ringkasan, grafik harian, tabel pengulangan, dan tombol unduh."""
    summary = res["summary"]
    sample_results = res["sample_results"]
    detail = res["detail"]
    used_dist = res["distribution"]
    used_params = res["params"]

    st.caption(
        f"Menampilkan: **Skenario #{res.get('scenario_no', 1)}** "
        f"({res.get('timestamp', '-')}) — {res.get('source_label', 'Data Contoh / Manual')} · "
        f"Stok awal {used_params['initial_stock']:,}, ROP {used_params['reorder_point']:,}, "
        f"Pesan {used_params['order_quantity']:,}, Lead time {used_params['lead_time']} hari."
    )

    stockout_pct = summary["stockout_probability"] * 100
    if stockout_pct <= 5:
        status_cls = "status-safe"
        status_title = "Status: Rencana Stok Sangat Aman"
        status_body = (
            f"Dari {used_params['replications']:,} pengulangan selama {used_params['days']} hari, "
            f"kemungkinan kehabisan stok hanya {stockout_pct:.1f}%."
        )
    elif stockout_pct <= 20:
        status_cls = "status-warn"
        status_title = f"Status: Cukup Aman dengan Risiko Sedang ({stockout_pct:.1f}%)"
        status_body = (
            "Masih terdapat peluang kehabisan stok pada sebagian skenario. "
            "Pertimbangkan menaikkan sedikit batas pesan ulang (ROP) atau jumlah pesanan."
        )
    else:
        status_cls = "status-risk"
        status_title = f"Status: Risiko Kehabisan Stok Tinggi ({stockout_pct:.1f}%)"
        status_body = (
            "Stok sering habis sebelum pesanan baru tiba. "
            "Disarankan menaikkan stok awal, batas pesan ulang (ROP), atau gunakan tombol Terapkan Saran."
        )

    st.markdown(
        f'<div class="status-box {status_cls}">'
        f'<div class="status-title">{status_title}</div>'
        f'<p class="status-body">{status_body}</p>'
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Rata-rata permintaan",
        f"{summary['mean_demand']:.2f} unit/hari",
        help="Estimasi rata-rata permintaan pembeli per hari.",
    )
    c2.metric(
        "Rata-rata terjual",
        f"{summary['mean_sales']:.2f} unit/hari",
        help="Estimasi rata-rata penjualan yang berhasil dipenuhi per hari.",
    )
    c3.metric(
        "Tidak terpenuhi",
        f"{summary['mean_lost_units']:.2f} unit",
        help=f"Rata-rata permintaan yang tidak dapat dilayani selama {used_params['days']} hari.",
    )
    c4.metric(
        "Risiko kehabisan stok",
        f"{stockout_pct:.1f}%",
        help="Persentase percobaan yang mengalami kehabisan stok minimal satu hari.",
    )

    extra1, extra2, extra3 = st.columns(3)
    extra1.metric("Rata-rata stok akhir", f"{summary['mean_ending_stock']:.1f} unit")
    extra2.metric("Rata-rata jumlah pesanan", f"{summary['mean_number_of_orders']:.1f} kali")
    extra3.metric("Rata-rata hari stok habis", f"{summary['mean_stockout_days']:.1f} hari")

    t_daily, t_rep, t_export = st.tabs(
        ["Grafik & Simulasi Harian", "Semua Pengulangan", "Unduh Laporan"]
    )

    with t_daily:
        st.subheader(f"Pergerakan Stok Selama {used_params['days']} Hari")
        chart_daily = detail.set_index("day")[["ending_stock", "demand"]].rename(
            columns={"ending_stock": "Stok Akhir Hari", "demand": "Permintaan Harian"}
        )
        st.line_chart(chart_daily, color=["#0F172A", "#94A3B8"])
        st.caption(
            "Garis gelap menunjukkan stok akhir tiap hari; garis abu-abu menunjukkan permintaan harian."
        )

        display_columns = {
            "day": "Hari ke-",
            "starting_stock": "Stok Awal",
            "arrivals": "Barang Masuk",
            "demand": "Permintaan",
            "sales": "Terjual",
            "lost_units": "Tidak Terpenuhi",
            "ending_stock": "Stok Akhir",
            "order_placed": "Pesanan Dibuat",
        }
        st.dataframe(
            detail[list(display_columns.keys())].rename(columns=display_columns),
            use_container_width=True,
            hide_index=True,
        )

    with t_rep:
        st.subheader(f"Ringkasan {used_params['replications']:,} Pengulangan Monte Carlo")
        st.caption("Menampilkan 100 pengulangan pertama. Seluruh data tersedia pada file unduhan.")
        rep_columns = {
            "replication": "Percobaan ke-",
            "total_demand": "Total Permintaan",
            "total_sales": "Total Terjual",
            "total_lost_units": "Total Tidak Terpenuhi",
            "ending_stock": "Stok Akhir",
            "number_of_orders": "Jumlah Pemesanan",
            "stockout_days": "Hari Stok Habis",
            "experienced_stockout": "Pernah Stok Habis",
        }
        st.dataframe(
            sample_results.head(100).rename(columns=rep_columns),
            use_container_width=True,
            hide_index=True,
        )

    with t_export:
        st.subheader("Unduh Hasil Simulasi")
        st.caption("Tersedia dalam satu file Excel lengkap (4 sheet) maupun file CSV terpisah.")

        try:
            excel_bytes = export_results_to_excel(summary, detail, sample_results, used_dist)
            st.download_button(
                "Unduh Laporan Lengkap (Excel .xlsx)",
                data=excel_bytes,
                file_name=f"laporan_skenario_{res.get('scenario_no', 1)}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True,
                key=f"{key_prefix}_dl_xlsx",
            )
        except ImportError:
            st.warning("Untuk unduh format Excel (.xlsx), pasang library openpyxl: `pip install openpyxl`")

        d1, d2, d3 = st.columns(3)
        d1.download_button(
            "Unduh Ringkasan (.csv)",
            data=pd.DataFrame([summary]).to_csv(index=False).encode("utf-8"),
            file_name=f"ringkasan_skenario_{res.get('scenario_no', 1)}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"{key_prefix}_dl_summary",
        )
        d2.download_button(
            "Unduh Simulasi Harian (.csv)",
            data=detail.to_csv(index=False).encode("utf-8"),
            file_name=f"detail_harian_skenario_{res.get('scenario_no', 1)}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"{key_prefix}_dl_detail",
        )
        d3.download_button(
            "Unduh Semua Pengulangan (.csv)",
            data=sample_results.to_csv(index=False).encode("utf-8"),
            file_name=f"pengulangan_skenario_{res.get('scenario_no', 1)}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"{key_prefix}_dl_rep",
        )

# Template contoh untuk diunduh user
TEMPLATE_HISTORY_CSV = (
    "tanggal,nama_barang,permintaan\n"
    "2026-01-01,Kopi Susu,12\n"
    "2026-01-02,Kopi Susu,15\n"
    "2026-01-03,Kopi Susu,10\n"
    "2026-01-04,Kopi Susu,18\n"
    "2026-01-05,Kopi Susu,15\n"
    "2026-01-06,Kopi Susu,12\n"
    "2026-01-07,Kopi Susu,20\n"
    "2026-01-08,Kopi Susu,14\n"
    "2026-01-09,Kopi Susu,15\n"
    "2026-01-10,Kopi Susu,12\n"
)

TEMPLATE_TABLE_CSV = (
    "permintaan,probabilitas\n"
    "0,0.10\n"
    "2,0.20\n"
    "4,0.40\n"
    "6,0.20\n"
    "8,0.10\n"
)

# --- HEADER & NAVIGASI ---
st.title("Simulasi Persediaan Barang")
st.markdown(
    '<div class="subtitle-text">'
    "Perencanaan stok menggunakan metode Monte Carlo berbasis dataset CSV dan Excel."
    "</div>",
    unsafe_allow_html=True,
)

history_total = len(st.session_state["sim_history"])
nav1, nav2, _ = st.columns([1.2, 1.2, 2.6])
with nav1:
    if st.button(
        "Simulasi Monte Carlo",
        type="primary" if st.session_state["active_view"] == "simulasi" else "secondary",
        use_container_width=True,
        key="nav_simulasi",
    ):
        st.session_state["active_view"] = "simulasi"
        st.rerun()
with nav2:
    if st.button(
        f"Riwayat Simulasi ({history_total})",
        type="primary" if st.session_state["active_view"] == "riwayat" else "secondary",
        use_container_width=True,
        key="nav_riwayat",
    ):
        st.session_state["active_view"] = "riwayat"
        st.rerun()

st.divider()

# --- SIDEBAR PARAMETER ---
with st.sidebar:
    st.markdown("### Pengaturan Rencana Stok")
    st.caption("Sesuaikan angka di bawah dengan kondisi persediaan Anda.")

    initial_stock = st.number_input(
        "Stok awal (unit)",
        min_value=0,
        max_value=1000000,
        step=10,
        key="initial_stock",
        help="Jumlah barang yang tersedia di gudang pada hari pertama.",
    )
    reorder_point = st.number_input(
        "Batas pesan ulang / ROP (unit)",
        min_value=0,
        max_value=1000000,
        step=5,
        key="reorder_point",
        help="Jika posisi stok menyentuh angka ini atau lebih rendah, sistem otomatis membuat pesanan baru.",
    )
    order_quantity = st.number_input(
        "Jumlah setiap kali pesan (unit)",
        min_value=1,
        max_value=1000000,
        step=10,
        key="order_quantity",
        help="Jumlah unit barang yang dibeli setiap kali melakukan pemesanan.",
    )
    lead_time = st.number_input(
        "Lama pengiriman / Lead time (hari)",
        min_value=1,
        max_value=60,
        step=1,
        key="lead_time",
        help="Waktu tunggu sejak barang dipesan hingga tiba di gudang.",
    )
    days = st.number_input(
        "Lama simulasi (hari)",
        min_value=1,
        max_value=365,
        step=1,
        key="days",
        help="Durasi periode yang disimulasikan, misalnya 30 hari untuk satu bulan.",
    )

    with st.expander("Pengaturan lanjutan"):
        replications = st.number_input(
            "Jumlah pengulangan Monte Carlo",
            min_value=10,
            max_value=10000,
            step=100,
            key="replications",
            help="Semakin banyak pengulangan, hasil estimasi rata-rata semakin stabil (disarankan 500–1000).",
        )
        seed = st.number_input(
            "Angka acak tetap (Seed)",
            min_value=0,
            max_value=999999,
            step=1,
            key="seed",
            help="Menjaga hasil tetap konsisten saat dijalankan ulang dengan pengaturan yang sama.",
        )

# --- HALAMAN RIWAYAT SIMULASI ---
if st.session_state["active_view"] == "riwayat":
    section_header("Riwayat Sesi", "Perbandingan & Riwayat Simulasi")
    st.caption(
        "Seluruh simulasi yang dijalankan pada sesi ini tersimpan sementara tanpa database. "
        "Riwayat otomatis hilang saat halaman di-refresh atau aplikasi ditutup."
    )

    history_list = st.session_state["sim_history"]
    if not history_list:
        st.info(
            "Belum ada riwayat simulasi pada sesi ini. Jalankan simulasi terlebih dahulu pada menu Simulasi Monte Carlo."
        )
        if st.button("Buka Halaman Simulasi", type="primary"):
            st.session_state["active_view"] = "simulasi"
            st.rerun()
        st.stop()

    history_rows = []
    for item in reversed(history_list):
        p = item["params"]
        s = item["summary"]
        history_rows.append(
            {
                "Skenario": f"#{item['scenario_no']}",
                "Waktu": item["timestamp"],
                "Sumber Data": item["source_label"],
                "Stok Awal": p["initial_stock"],
                "ROP": p["reorder_point"],
                "Jml Pesan": p["order_quantity"],
                "Lead Time": p["lead_time"],
                "Durasi (Hari)": p["days"],
                "Rata-rata Permintaan": round(s["mean_demand"], 2),
                "Tidak Terpenuhi": round(s["mean_lost_units"], 2),
                "Risiko Habis Stok (%)": round(s["stockout_probability"] * 100, 1),
                "Status": item["status_short"],
            }
        )
    history_df = pd.DataFrame(history_rows)
    st.dataframe(history_df, use_container_width=True, hide_index=True)

    act1, act2 = st.columns([3, 1])
    with act1:
        st.download_button(
            "Unduh Tabel Riwayat Perbandingan (.csv)",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="riwayat_perbandingan_simulasi.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with act2:
        if st.button("Hapus Semua Riwayat", use_container_width=True):
            st.session_state["sim_history"] = []
            st.session_state.pop("sim_result", None)
            st.rerun()

    if len(history_list) >= 2:
        st.subheader("Perbandingan Risiko Kehabisan Stok (%)")
        compare_chart = pd.DataFrame(
            {
                "Skenario": [f"Skenario #{item['scenario_no']}" for item in history_list],
                "Risiko Habis Stok (%)": [
                    round(item["summary"]["stockout_probability"] * 100, 1)
                    for item in history_list
                ],
            }
        ).set_index("Skenario")
        st.bar_chart(compare_chart, color="#334155")

    st.divider()
    st.subheader("Lihat Detail Skenario")
    reversed_items = list(reversed(history_list))
    scenario_labels = [
        f"Skenario #{item['scenario_no']} ({item['timestamp']}) — {item['source_label']} | "
        f"Stok {item['params']['initial_stock']}, ROP {item['params']['reorder_point']}, "
        f"Pesan {item['params']['order_quantity']} | Risiko {item['summary']['stockout_probability'] * 100:.1f}%"
        for item in reversed_items
    ]
    selected_idx = st.selectbox(
        "Pilih skenario yang ingin dibuka:",
        options=range(len(scenario_labels)),
        format_func=lambda idx: scenario_labels[idx],
    )
    render_simulation_result(reversed_items[selected_idx], key_prefix=f"hist_{selected_idx}")
    st.stop()

g1, g2, g3 = st.columns(3)
with g1:
    st.markdown(
        '<div class="step-card">'
        '<div class="step-num">Langkah 01</div>'
        '<div class="step-title">Masukkan Data</div>'
        '<p class="step-desc">Unggah file CSV/Excel penjualan atau gunakan data contoh yang tersedia.</p>'
        "</div>",
        unsafe_allow_html=True,
    )
with g2:
    st.markdown(
        '<div class="step-card">'
        '<div class="step-num">Langkah 02</div>'
        '<div class="step-title">Atur Rencana Stok</div>'
        '<p class="step-desc">Sesuaikan stok awal, batas pesan ulang, dan lama pengiriman pada panel kiri.</p>'
        "</div>",
        unsafe_allow_html=True,
    )
with g3:
    st.markdown(
        '<div class="step-card">'
        '<div class="step-num">Langkah 03</div>'
        '<div class="step-title">Analisis Hasil</div>'
        '<p class="step-desc">Jalankan simulasi untuk melihat risiko kehabisan stok, grafik, dan laporan.</p>'
        "</div>",
        unsafe_allow_html=True,
    )

# --- LANGKAH 1: SUMBER DATA ---
section_header("Langkah 01", "Masukkan Data Permintaan")

data_source = st.radio(
    "Pilih sumber data:",
    options=["Upload File CSV / Excel", "Data Contoh / Input Manual"],
    horizontal=True,
    label_visibility="collapsed",
)

base_distribution = None
source_signature = None
source_label = "Data Contoh / Manual"

if data_source == "Upload File CSV / Excel":
    st.caption(
        "Mendukung file .csv, .xlsx, dan .xls. Anda dapat mengunggah catatan penjualan harian "
        "(probabilitas dihitung otomatis) atau tabel distribusi yang sudah jadi."
    )

    with st.expander("Belum punya dataset? Unduh contoh template"):
        t1, t2 = st.columns(2)
        with t1:
            st.caption("Format A — Riwayat penjualan harian (paling umum)")
            st.download_button(
                "Unduh Template Riwayat Penjualan (.csv)",
                data=TEMPLATE_HISTORY_CSV.encode("utf-8"),
                file_name="contoh_riwayat_penjualan.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with t2:
            st.caption("Format B — Tabel permintaan & probabilitas")
            st.download_button(
                "Unduh Template Tabel Distribusi (.csv)",
                data=TEMPLATE_TABLE_CSV.encode("utf-8"),
                file_name="contoh_distribusi_permintaan.csv",
                mime="text/csv",
                use_container_width=True,
            )

    uploaded_file = st.file_uploader(
        "Pilih atau tarik file CSV / Excel ke area ini",
        type=["csv", "xlsx", "xls"],
        help="Mendukung pemisah koma maupun titik koma, serta format Excel (.xlsx / .xls).",
    )

    if uploaded_file is None:
        st.info(
            "Unggah file CSV atau Excel di atas untuk memulai, atau pilih opsi Data Contoh / Input Manual."
        )
        st.stop()

    try:
        raw_df, sheets = read_uploaded_dataset(uploaded_file, uploaded_file.name)
        selected_sheet = None
        if len(sheets) > 1:
            selected_sheet = st.selectbox("Pilih Sheet Excel:", sheets)
            raw_df, _ = read_uploaded_dataset(uploaded_file, uploaded_file.name, sheet_name=selected_sheet)
    except ValueError as exc:
        st.error(f"Gagal membaca file: {exc}")
        st.stop()

    st.caption(
        f"File aktif: **{uploaded_file.name}** — {len(raw_df):,} baris, {len(raw_df.columns)} kolom."
    )

    with st.expander("Lihat pratinjau data (10 baris pertama)", expanded=False):
        st.dataframe(raw_df.head(10), use_container_width=True, hide_index=True)

    # Filter opsional jika dataset berisi beberapa produk/kategori yang berulang
    filtered_df = raw_df.copy()
    filter_desc = "all"
    source_label = uploaded_file.name
    categorical_cols = [
        col
        for col in raw_df.columns
        if (raw_df[col].dtype == "object" or pd.api.types.is_string_dtype(raw_df[col]))
        and not any(k in str(col).lower() for k in ("tgl", "tanggal", "date", "waktu", "time"))
        and 2 <= raw_df[col].dropna().nunique() <= 50
        and raw_df[col].dropna().nunique() < max(len(raw_df) * 0.8, 2)
    ]
    if categorical_cols:
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            filter_col = st.selectbox(
                "Filter berdasarkan barang/kategori (opsional):",
                options=["Semua data"] + categorical_cols,
                help="Gunakan jika satu file memuat beberapa produk dan Anda ingin mensimulasikan satu produk saja.",
            )
        if filter_col != "Semua data":
            unique_vals = sorted(raw_df[filter_col].dropna().astype(str).unique().tolist())
            with f_col2:
                chosen_val = st.selectbox(f"Pilih nilai pada kolom '{filter_col}':", options=unique_vals)
            filtered_df = raw_df[raw_df[filter_col].astype(str) == chosen_val].copy()
            filter_desc = f"{filter_col}={chosen_val}"
            source_label = f"{uploaded_file.name} ({chosen_val})"
            st.caption(f"Data terfilter: {len(filtered_df):,} baris untuk **{chosen_val}**.")

    detected = detect_dataset_columns(filtered_df)
    mode_options = [
        "Data Riwayat Harian (hitung probabilitas otomatis)",
        "Tabel Distribusi (sudah ada kolom Probabilitas / Frekuensi)",
    ]
    default_mode_idx = 1 if detected["suggested_mode"] == "table" else 0

    dataset_mode = st.radio(
        "Jenis struktur dataset:",
        options=mode_options,
        index=default_mode_idx,
        horizontal=True,
    )

    all_columns = list(filtered_df.columns)
    default_demand_idx = (
        all_columns.index(detected["demand_col"]) if detected["demand_col"] in all_columns else 0
    )

    if dataset_mode == mode_options[0]:
        demand_col = st.selectbox(
            "Kolom jumlah permintaan / penjualan:",
            options=all_columns,
            index=default_demand_idx,
        )
        try:
            dist_from_history, stats = build_distribution_from_history(filtered_df, demand_col)
            base_distribution = dist_from_history[["demand", "probability"]].copy()
            source_signature = (
                f"upload_history:{uploaded_file.name}:{uploaded_file.size}:{selected_sheet}:{filter_desc}:{demand_col}"
            )
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Data valid", f"{stats['valid_rows']:,} baris")
        m2.metric("Permintaan minimum", f"{stats['min_demand']:,} unit")
        m3.metric("Rata-rata harian", f"{stats['mean_demand']:.2f} unit")
        m4.metric("Permintaan maksimum", f"{stats['max_demand']:,} unit")

        notes = []
        if stats["dropped_rows"] > 0:
            notes.append(f"{stats['dropped_rows']} baris kosong/non-angka dilewati")
        if stats["rounded_count"] > 0:
            notes.append(f"{stats['rounded_count']} angka desimal dibulatkan ke bilangan bulat terdekat")
        if notes:
            st.caption("Catatan pembersihan data: " + "; ".join(notes) + ".")

    else:
        col_a, col_b = st.columns(2)
        with col_a:
            demand_col = st.selectbox(
                "Kolom jumlah permintaan (unit):",
                options=all_columns,
                index=default_demand_idx,
            )
        weight_candidates = [c for c in all_columns if c != demand_col]
        if not weight_candidates:
            st.error("Mode Tabel Distribusi memerlukan minimal 2 kolom.")
            st.stop()

        default_weight_idx = (
            weight_candidates.index(detected["weight_col"])
            if detected["weight_col"] in weight_candidates
            else 0
        )
        with col_b:
            weight_col = st.selectbox(
                "Kolom probabilitas / frekuensi:",
                options=weight_candidates,
                index=default_weight_idx,
            )

        auto_norm = st.checkbox(
            "Normalisasi otomatis agar total menjadi 1.00 (100%)",
            value=True,
            help="Aktifkan jika kolom berisi frekuensi (jumlah hari) atau angka persentase.",
        )

        try:
            base_distribution, info = build_distribution_from_table(
                filtered_df, demand_col, weight_col, auto_normalize=auto_norm
            )
            source_signature = (
                f"upload_table:{uploaded_file.name}:{uploaded_file.size}:{selected_sheet}:{filter_desc}:{demand_col}:{weight_col}:{auto_norm}"
            )
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        if info["was_normalized"]:
            st.caption(
                f"Total kolom **{weight_col}** semula {info['original_sum']:.2f} dan telah disesuaikan otomatis ke 1.00."
            )

else:
    base_distribution = load_demand_distribution("data/demand_distribution.csv")
    source_signature = "default_sample"
    st.caption(
        "Menggunakan data contoh bawaan (`data/demand_distribution.csv`). "
        "Anda dapat langsung menjalankan simulasi atau mengubah nilainya pada tab Edit Manual."
    )

# Sinkronkan tabel kerja jika sumber data berubah
if st.session_state.get("dist_source_signature") != source_signature:
    st.session_state["dist_source_signature"] = source_signature
    st.session_state["working_distribution"] = base_distribution.copy()
    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1

# --- LANGKAH 2: TABEL DISTRIBUSI MONTE CARLO ---
section_header("Langkah 02", "Distribusi Probabilitas Monte Carlo")
st.caption(
    "Pola permintaan harian beserta probabilitas kumulatif dan interval angka acak yang digunakan dalam simulasi."
)

tab_view, tab_edit = st.tabs(["Ringkasan & Interval Angka Acak", "Edit Manual"])

with tab_edit:
    st.caption("Ubah, tambah, atau hapus baris sesuai kebutuhan. Total probabilitas harus bernilai 1.00.")
    editor_key = f"distribution_editor_{st.session_state.get('editor_version', 0)}"
    edited_df = st.data_editor(
        st.session_state["working_distribution"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "demand": st.column_config.NumberColumn("Permintaan (unit)", min_value=0, step=1),
            "probability": st.column_config.NumberColumn(
                "Probabilitas", min_value=0.0, max_value=1.0, step=0.01, format="%.4f"
            ),
        },
        key=editor_key,
    )

    if st.button("Rapikan Total Probabilitas ke 1.00"):
        try:
            st.session_state["working_distribution"] = normalize_distribution(edited_df)
            st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

try:
    distribution = validate_distribution(edited_df)
except ValueError as exc:
    st.error(
        f"Distribusi belum valid: {exc} "
        "Buka tab Edit Manual lalu klik tombol 'Rapikan Total Probabilitas ke 1.00'."
    )
    st.stop()

with tab_view:
    interval_table = build_monte_carlo_interval_table(distribution)
    col_tbl, col_chart = st.columns([3, 2])
    with col_tbl:
        st.dataframe(interval_table, use_container_width=True, hide_index=True)
    with col_chart:
        st.caption("Sebaran Peluang Permintaan (%)")
        chart_df = interval_table.set_index("Permintaan (unit)")[["Persentase (%)"]]
        st.bar_chart(chart_df, color="#334155")

# Rekomendasi parameter otomatis
suggestion = suggest_inventory_parameters(distribution, lead_time=int(lead_time))
with st.container(border=True):
    s_col1, s_col2 = st.columns([3, 1])
    with s_col1:
        st.markdown(
            f"**Rekomendasi Parameter Berdasarkan Data**  \n"
            f"Rata-rata permintaan **{suggestion['mean_demand']:.1f} unit/hari** "
            f"(maksimum **{suggestion['max_demand']:,} unit/hari**). "
            f"Saran: Stok awal **{suggestion['initial_stock']:,} unit**, "
            f"ROP **{suggestion['reorder_point']:,} unit**, "
            f"Jumlah pesan **{suggestion['order_quantity']:,} unit**."
        )
    with s_col2:
        st.write("")
        if st.button("Terapkan Saran", use_container_width=True):
            st.session_state["pending_params"] = {
                "initial_stock": suggestion["initial_stock"],
                "reorder_point": suggestion["reorder_point"],
                "order_quantity": suggestion["order_quantity"],
            }
            st.rerun()

if reorder_point >= initial_stock + order_quantity:
    st.warning(
        "Catatan: Batas pesan ulang (ROP) cukup tinggi dibandingkan stok awal dan jumlah pesanan, "
        "sehingga pemesanan akan dilakukan lebih sering."
    )

# --- LANGKAH 3: JALANKAN SIMULASI ---
section_header("Langkah 03", "Jalankan Simulasi & Analisis Hasil")

if st.button("Jalankan Simulasi Monte Carlo", type="primary", use_container_width=True):
    with st.spinner("Menjalankan simulasi Monte Carlo..."):
        summary, sample_results = run_monte_carlo(
            distribution=distribution,
            initial_stock=int(initial_stock),
            reorder_point=int(reorder_point),
            order_quantity=int(order_quantity),
            lead_time=int(lead_time),
            days=int(days),
            replications=int(replications),
            seed=int(seed),
        )
        detail = simulate_one_run(
            distribution=distribution,
            initial_stock=int(initial_stock),
            reorder_point=int(reorder_point),
            order_quantity=int(order_quantity),
            lead_time=int(lead_time),
            days=int(days),
            seed=int(seed),
        )

    stockout_pct_run = summary["stockout_probability"] * 100
    if stockout_pct_run <= 5:
        status_short = "Sangat Aman"
    elif stockout_pct_run <= 20:
        status_short = "Cukup Aman"
    else:
        status_short = "Risiko Tinggi"

    scenario_no = len(st.session_state["sim_history"]) + 1
    run_record = {
        "scenario_no": scenario_no,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "source_label": source_label,
        "status_short": status_short,
        "summary": summary,
        "sample_results": sample_results,
        "detail": detail,
        "distribution": distribution.copy(),
        "params": {
            "initial_stock": int(initial_stock),
            "reorder_point": int(reorder_point),
            "order_quantity": int(order_quantity),
            "lead_time": int(lead_time),
            "days": int(days),
            "replications": int(replications),
            "seed": int(seed),
        },
    }
    st.session_state["sim_result"] = run_record
    st.session_state["sim_history"].append(run_record)
    st.rerun()

if "sim_result" in st.session_state:
    render_simulation_result(st.session_state["sim_result"], key_prefix="main")

with st.expander("Penjelasan istilah & cara kerja model"):
    st.markdown(
        """
        - **Simulasi Monte Carlo**: metode pengujian skenario permintaan secara acak berulang kali berdasarkan distribusi data untuk mengestimasi performa stok.
        - **ROP (Reorder Point / Batas Pesan Ulang)**: batas minimum posisi persediaan yang memicu pemesanan baru.
        - **Lead Time**: jumlah hari tunggu sejak pesanan dibuat hingga barang masuk ke gudang.
        - **Lost Sales (Tidak Terpenuhi)**: kekurangan stok dicatat sebagai permintaan tidak terpenuhi dan tidak menjadi utang pesanan hari berikutnya.
        - Hasil simulasi merupakan estimasi statistik berdasarkan asumsi dan data yang dimasukkan.
        """
    )

st.divider()
st.caption("Aplikasi Simulasi Persediaan Barang — Metode Monte Carlo")
