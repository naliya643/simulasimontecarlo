"""Logika simulasi persediaan Monte Carlo.

Model:
- Permintaan harian dipilih dari distribusi diskret.
- Lost sales: permintaan yang tidak dapat dipenuhi tidak menjadi backlog.
- Pemesanan dipicu jika inventory position <= reorder point.
- Pesanan tiba pada awal hari t + lead_time.
"""

from pathlib import Path
import csv
import io
import numpy as np
import pandas as pd


def load_demand_distribution(path):
    """Muat distribusi permintaan dari CSV."""
    return pd.read_csv(path)


def _extract_bytes(file_obj):
    """Ambil bytes mentah dari UploadedFile, BytesIO, bytes, atau path."""
    if isinstance(file_obj, (bytes, bytearray)):
        return bytes(file_obj)
    if isinstance(file_obj, (str, Path)):
        return Path(file_obj).read_bytes()
    if hasattr(file_obj, "getvalue"):
        return file_obj.getvalue()
    pos = file_obj.tell() if hasattr(file_obj, "tell") else None
    data = file_obj.read()
    if pos is not None and hasattr(file_obj, "seek"):
        file_obj.seek(pos)
    return data if isinstance(data, bytes) else str(data).encode("utf-8")


def _detect_csv_separator(sample_text):
    """Deteksi pemisah kolom CSV (koma, titik koma, atau tab)."""
    lines = [line for line in sample_text.splitlines() if line.strip()][:10]
    sample = "\n".join(lines)
    if not sample:
        return ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        counts = {sep: sample.count(sep) for sep in [";", ",", "\t", "|"]}
        best_sep = max(counts, key=counts.get)
        return best_sep if counts[best_sep] > 0 else ","


def read_uploaded_dataset(file_obj, file_name, sheet_name=None):
    """Baca dataset dari file CSV atau Excel (.xlsx/.xls).

    Mengembalikan tuple (dataframe, daftar_nama_sheet).
    """
    raw_bytes = _extract_bytes(file_obj)
    ext = Path(str(file_name)).suffix.lower()

    if ext in {".csv", ".txt"}:
        text = None
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
            try:
                text = raw_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            raise ValueError("File CSV tidak dapat dibaca. Pastikan encoding file valid (UTF-8).")

        sep = _detect_csv_separator(text)
        df = pd.read_csv(io.StringIO(text), sep=sep)
        df = df.dropna(how="all")
        df.columns = [str(col).strip() for col in df.columns]
        if df.empty:
            raise ValueError("File CSV kosong atau tidak memiliki baris data.")
        return df, []

    if ext in {".xlsx", ".xls"}:
        try:
            excel = pd.ExcelFile(io.BytesIO(raw_bytes))
        except ImportError as exc:
            raise ValueError(
                "Library pembaca Excel belum terpasang. Jalankan perintah: pip install openpyxl"
            ) from exc
        except Exception as exc:
            raise ValueError(f"File Excel tidak dapat dibaca: {exc}") from exc

        sheets = list(excel.sheet_names)
        if not sheets:
            raise ValueError("File Excel tidak memiliki sheet.")
        selected_sheet = sheet_name if sheet_name in sheets else sheets[0]
        df = excel.parse(selected_sheet)
        df = df.dropna(how="all")
        df.columns = [str(col).strip() for col in df.columns]
        if df.empty:
            raise ValueError(f"Sheet '{selected_sheet}' kosong atau tidak memiliki data.")
        return df, sheets

    raise ValueError("Format file tidak didukung. Gunakan file .csv, .xlsx, atau .xls.")


def parse_numeric_series(series):
    """Konversi Series ke angka, mendukung desimal koma (0,25) dan persen (25%)."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    def _clean_value(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        if not s:
            return np.nan

        is_percent = s.endswith("%")
        if is_percent:
            s = s[:-1].strip()

        s = s.replace(" ", "")
        if "," in s and "." in s:
            if s.rfind(",") > s.rfind("."):
                # Format Indonesia/Eropa: 1.250,50
                s = s.replace(".", "").replace(",", ".")
            else:
                # Format US: 1,250.50
                s = s.replace(",", "")
        elif "," in s:
            s = s.replace(",", ".")

        try:
            num = float(s)
            return num / 100.0 if is_percent else num
        except ValueError:
            return np.nan

    return pd.Series([_clean_value(v) for v in series], index=series.index, dtype="float64")


def detect_dataset_columns(df):
    """Tebak kolom permintaan, kolom probabilitas/frekuensi, dan jenis data."""
    columns = list(df.columns)
    lower_map = {col: str(col).lower() for col in columns}

    numeric_cols = []
    for col in columns:
        parsed = parse_numeric_series(df[col])
        if parsed.notna().mean() >= 0.6 and parsed.notna().sum() > 0:
            numeric_cols.append(col)

    weight_keywords = ("prob", "peluang", "frek", "freq", "persen", "percent", "%", "bobot", "weight", "hari")
    demand_keywords = ("permintaan", "demand", "penjualan", "sales", "qty", "jumlah", "terjual", "unit", "order")

    weight_col = None
    for col in numeric_cols:
        if any(k in lower_map[col] for k in weight_keywords):
            weight_col = col
            break

    demand_col = None
    for col in numeric_cols:
        if col == weight_col:
            continue
        if any(k in lower_map[col] for k in demand_keywords):
            demand_col = col
            break

    if demand_col is None and numeric_cols:
        remaining = [c for c in numeric_cols if c != weight_col]
        demand_col = remaining[0] if remaining else numeric_cols[0]

    if weight_col is None and len(numeric_cols) >= 2:
        for col in numeric_cols:
            if col != demand_col:
                weight_col = col
                break

    # Jika ada kolom bernama probabilitas/frekuensi & barisnya tidak banyak duplikat, kemungkinan tabel distribusi
    looks_like_table = False
    if weight_col is not None and any(k in lower_map[weight_col] for k in ("prob", "peluang", "frek", "freq", "persen", "percent", "%", "bobot", "weight")):
        looks_like_table = True

    return {
        "numeric_columns": numeric_cols,
        "demand_col": demand_col or (columns[0] if columns else None),
        "weight_col": weight_col,
        "suggested_mode": "table" if looks_like_table else "history",
    }


def build_distribution_from_history(df, demand_column):
    """Hitung distribusi probabilitas Monte Carlo otomatis dari data riwayat permintaan."""
    if demand_column not in df.columns:
        raise ValueError(f"Kolom '{demand_column}' tidak ditemukan di dataset.")

    raw_series = parse_numeric_series(df[demand_column])
    total_rows = int(len(raw_series))
    valid_series = raw_series.dropna()

    if valid_series.empty:
        raise ValueError(
            f"Kolom '{demand_column}' tidak berisi data angka yang valid. Pilih kolom angka yang sesuai."
        )
    if (valid_series < 0).any():
        raise ValueError(
            f"Kolom '{demand_column}' mengandung nilai negatif. Permintaan barang harus 0 atau lebih."
        )

    rounded_mask = ~np.isclose(valid_series, np.round(valid_series))
    rounded_count = int(rounded_mask.sum())
    clean_int = np.round(valid_series).astype(int)

    counts = clean_int.value_counts().sort_index()
    total_valid = int(counts.sum())
    probs = counts.to_numpy(dtype=float) / total_valid
    # Pastikan jumlah probabilitas tepat 1.0
    probs = probs / probs.sum()

    dist = pd.DataFrame(
        {
            "demand": counts.index.astype(int),
            "frequency": counts.to_numpy(dtype=int),
            "probability": probs,
        }
    )

    stats = {
        "total_rows": total_rows,
        "valid_rows": total_valid,
        "dropped_rows": total_rows - total_valid,
        "rounded_count": rounded_count,
        "unique_demands": int(len(dist)),
        "min_demand": int(clean_int.min()),
        "max_demand": int(clean_int.max()),
        "mean_demand": float(clean_int.mean()),
    }
    return dist, stats


def build_distribution_from_table(df, demand_column, weight_column, auto_normalize=True):
    """Bangun distribusi valid dari tabel permintaan + probabilitas/frekuensi."""
    if demand_column not in df.columns or weight_column not in df.columns:
        raise ValueError("Kolom permintaan dan kolom probabilitas/frekuensi harus dipilih.")
    if demand_column == weight_column:
        raise ValueError("Kolom permintaan dan kolom probabilitas/frekuensi tidak boleh sama.")

    demands = parse_numeric_series(df[demand_column])
    weights = parse_numeric_series(df[weight_column])

    temp = pd.DataFrame({"demand": demands, "weight": weights}).dropna()
    if temp.empty:
        raise ValueError("Tidak ada baris angka yang valid pada kolom yang dipilih.")
    if (temp["demand"] < 0).any():
        raise ValueError("Nilai permintaan tidak boleh negatif.")
    if (temp["weight"] < 0).any():
        raise ValueError("Nilai probabilitas atau frekuensi tidak boleh negatif.")
    if temp["weight"].sum() <= 0:
        raise ValueError("Total probabilitas/frekuensi harus lebih besar dari nol.")

    rounded_count = int((~np.isclose(temp["demand"], np.round(temp["demand"]))).sum())
    temp["demand"] = np.round(temp["demand"]).astype(int)
    grouped = temp.groupby("demand", as_index=False)["weight"].sum().sort_values("demand").reset_index(drop=True)

    original_sum = float(grouped["weight"].sum())
    was_normalized = False
    if not np.isclose(original_sum, 1.0, atol=1e-6):
        if auto_normalize:
            grouped["probability"] = grouped["weight"] / original_sum
            was_normalized = True
        else:
            raise ValueError(
                f"Total probabilitas saat ini {original_sum:.4f} (harus 1.00). "
                "Aktifkan normalisasi otomatis jika kolom berisi frekuensi atau persen."
            )
    else:
        grouped["probability"] = grouped["weight"] / original_sum

    dist = grouped[["demand", "probability"]].copy()
    info = {
        "valid_rows": int(len(temp)),
        "dropped_rows": int(len(df) - len(temp)),
        "rounded_count": rounded_count,
        "original_sum": original_sum,
        "was_normalized": was_normalized,
        "mean_demand": float((dist["demand"] * dist["probability"]).sum()),
    }
    return dist, info


def normalize_distribution(distribution):
    """Normalisasi kolom probability agar totalnya tepat 1.00."""
    result = distribution[["demand", "probability"]].copy()
    result["demand"] = parse_numeric_series(result["demand"])
    result["probability"] = parse_numeric_series(result["probability"])
    result = result.dropna()
    if result.empty:
        raise ValueError("Tabel distribusi kosong atau belum diisi angka.")
    if (result["demand"] < 0).any() or (result["probability"] < 0).any():
        raise ValueError("Nilai permintaan dan probabilitas tidak boleh negatif.")
    total = float(result["probability"].sum())
    if total <= 0:
        raise ValueError("Total probabilitas harus lebih besar dari nol.")
    result["demand"] = np.round(result["demand"]).astype(int)
    result["probability"] = result["probability"] / total
    result = result.groupby("demand", as_index=False)["probability"].sum().sort_values("demand").reset_index(drop=True)
    return result


def build_monte_carlo_interval_table(distribution):
    """Buat tabel Monte Carlo lengkap dengan kumulatif & interval angka acak (0.00 - 1.00)."""
    dist = validate_distribution(distribution)
    cum_probs = dist["probability"].cumsum().to_numpy()
    starts = np.concatenate(([0.0], cum_probs[:-1]))
    ends = np.minimum(cum_probs, 1.0)
    ends[-1] = 1.0

    intervals = [f"{s:.4f} – {e:.4f}" for s, e in zip(starts, ends)]
    return pd.DataFrame(
        {
            "Permintaan (unit)": dist["demand"].astype(int),
            "Probabilitas": dist["probability"].round(4),
            "Persentase (%)": (dist["probability"] * 100).round(2),
            "Probabilitas Kumulatif": np.round(ends, 4),
            "Interval Angka Acak": intervals,
        }
    )


def suggest_inventory_parameters(distribution, lead_time=3):
    """Hitung rekomendasi parameter stok awal, ROP, dan jumlah pesanan berdasarkan data."""
    dist = validate_distribution(distribution)
    mean_demand = float((dist["demand"] * dist["probability"]).sum())
    max_demand = int(dist["demand"].max())
    lead = max(1, int(lead_time))

    # Jika rata-rata permintaan sangat kecil (misal 0), tetap beri angka minimal masuk akal
    effective_mean = max(mean_demand, 1.0)
    rop = max(1, int(round(effective_mean * lead * 1.25)))
    order_qty = max(1, int(round(effective_mean * max(lead * 2, 7))))
    initial_stock = max(rop + int(round(effective_mean * 2)), order_qty, max_demand * 2)

    return {
        "mean_demand": mean_demand,
        "max_demand": max_demand,
        "initial_stock": int(initial_stock),
        "reorder_point": int(rop),
        "order_quantity": int(order_qty),
    }


def export_results_to_excel(summary, detail, sample_results, distribution):
    """Ekspor seluruh hasil simulasi ke file Excel (.xlsx) dengan beberapa sheet."""
    output = io.BytesIO()
    summary_rows = [
        ("Jumlah pengulangan Monte Carlo", summary["replications"]),
        ("Lama simulasi per pengulangan (hari)", summary["days_per_replication"]),
        ("Rata-rata permintaan (unit/hari)", round(summary["mean_demand"], 2)),
        ("Rata-rata penjualan terpenuhi (unit/hari)", round(summary["mean_sales"], 2)),
        ("Rata-rata permintaan tidak terpenuhi (unit/simulasi)", round(summary["mean_lost_units"], 2)),
        ("Risiko kehabisan stok (%)", round(summary["stockout_probability"] * 100, 2)),
        ("Rata-rata stok akhir (unit)", round(summary["mean_ending_stock"], 2)),
        ("Rata-rata jumlah pemesanan (kali/simulasi)", round(summary["mean_number_of_orders"], 2)),
        ("Rata-rata hari kehabisan stok (hari/simulasi)", round(summary["mean_stockout_days"], 2)),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Indikator", "Nilai"])
    interval_df = build_monte_carlo_interval_table(distribution)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Ringkasan", index=False)
        interval_df.to_excel(writer, sheet_name="Distribusi_Monte_Carlo", index=False)
        detail.to_excel(writer, sheet_name="Contoh_Simulasi_Harian", index=False)
        sample_results.to_excel(writer, sheet_name="Hasil_Semua_Pengulangan", index=False)

    return output.getvalue()


def validate_distribution(distribution):
    """Validasi dan kembalikan distribusi dengan format standar."""
    required = {"demand", "probability"}
    if not required.issubset(distribution.columns):
        raise ValueError("Distribusi wajib memiliki kolom 'demand' dan 'probability'.")

    result = distribution[["demand", "probability"]].copy()
    if result.empty:
        raise ValueError("Distribusi permintaan tidak boleh kosong.")

    result["demand"] = pd.to_numeric(result["demand"], errors="coerce")
    result["probability"] = pd.to_numeric(result["probability"], errors="coerce")

    if result.isna().any().any():
        raise ValueError("Nilai permintaan dan probabilitas harus berupa angka.")
    if (result["demand"] < 0).any():
        raise ValueError("Permintaan tidak boleh negatif.")
    if not np.all(np.isclose(result["demand"], np.round(result["demand"]))):
        raise ValueError("Permintaan harus berupa bilangan bulat.")
    if (result["probability"] < 0).any():
        raise ValueError("Probabilitas tidak boleh negatif.")
    if result["probability"].sum() <= 0:
        raise ValueError("Total probabilitas harus lebih besar dari nol.")
    if not np.isclose(result["probability"].sum(), 1.0, atol=1e-6):
        raise ValueError(
            f"Total probabilitas harus 1.00, saat ini {result['probability'].sum():.4f}."
        )

    result["demand"] = result["demand"].astype(int)
    result = result.groupby("demand", as_index=False)["probability"].sum()
    result = result.sort_values("demand").reset_index(drop=True)
    if not np.isclose(result["probability"].sum(), 1.0, atol=1e-6):
        raise ValueError("Total probabilitas setelah validasi harus 1.00.")
    return result


def simulate_one_run(
    distribution,
    initial_stock=100,
    reorder_point=30,
    order_quantity=80,
    lead_time=3,
    days=30,
    seed=42,
):
    """Jalankan satu replikasi simulasi dan kembalikan detail per hari."""
    distribution = validate_distribution(distribution)

    if initial_stock < 0:
        raise ValueError("Stok awal tidak boleh negatif.")
    if reorder_point < 0:
        raise ValueError("ROP tidak boleh negatif.")
    if order_quantity <= 0:
        raise ValueError("Jumlah pemesanan harus lebih besar dari nol.")
    if lead_time < 1:
        raise ValueError("Lead time minimal 1 hari.")
    if days < 1:
        raise ValueError("Durasi simulasi minimal 1 hari.")

    rng = np.random.default_rng(seed)
    demand_values = distribution["demand"].to_numpy()
    probabilities = distribution["probability"].to_numpy(dtype=float)
    probabilities = probabilities / probabilities.sum()

    on_hand = int(initial_stock)
    # Daftar pesanan yang akan tiba: (hari_kedatangan, jumlah_unit).
    pipeline = []
    records = []

    for day in range(1, int(days) + 1):
        # Barang yang dijadwalkan tiba pada hari ini masuk di awal hari.
        arrivals = sum(qty for arrival_day, qty in pipeline if arrival_day == day)
        pipeline = [(arrival_day, qty) for arrival_day, qty in pipeline if arrival_day != day]
        on_hand += arrivals
        starting_stock = on_hand

        demand = int(rng.choice(demand_values, p=probabilities))
        sales = min(on_hand, demand)
        lost_units = demand - sales
        on_hand -= sales

        outstanding = sum(qty for _, qty in pipeline)
        inventory_position = on_hand + outstanding
        order_placed = 0

        # Pesanan yang dibuat hari ini tiba pada awal hari day + lead_time.
        if inventory_position <= reorder_point:
            order_placed = int(order_quantity)
            pipeline.append((day + int(lead_time), order_placed))

        records.append(
            {
                "day": day,
                "starting_stock": starting_stock,
                "arrivals": int(arrivals),
                "demand": demand,
                "sales": int(sales),
                "lost_units": int(lost_units),
                "ending_stock": int(on_hand),
                "inventory_position": int(inventory_position),
                "order_placed": order_placed,
                "outstanding_orders_end_of_day": int(sum(qty for _, qty in pipeline)),
            }
        )

    return pd.DataFrame(records)


def run_monte_carlo(
    distribution,
    initial_stock=100,
    reorder_point=30,
    order_quantity=80,
    lead_time=3,
    days=30,
    replications=500,
    seed=42,
):
    """Jalankan banyak replikasi dan kembalikan ringkasan serta hasil per replikasi."""
    distribution = validate_distribution(distribution)
    if replications < 1:
        raise ValueError("Jumlah replikasi minimal 1.")

    seed_sequence = np.random.SeedSequence(int(seed))
    child_seeds = seed_sequence.spawn(int(replications))
    rows = []

    for replication, child_seed in enumerate(child_seeds, start=1):
        run_seed = int(child_seed.generate_state(1)[0])
        detail = simulate_one_run(
            distribution=distribution,
            initial_stock=initial_stock,
            reorder_point=reorder_point,
            order_quantity=order_quantity,
            lead_time=lead_time,
            days=days,
            seed=run_seed,
        )
        total_demand = int(detail["demand"].sum())
        total_sales = int(detail["sales"].sum())
        lost_units = int(detail["lost_units"].sum())
        rows.append(
            {
                "replication": replication,
                "total_demand": total_demand,
                "total_sales": total_sales,
                "total_lost_units": lost_units,
                "ending_stock": int(detail.iloc[-1]["ending_stock"]),
                "number_of_orders": int((detail["order_placed"] > 0).sum()),
                "stockout_days": int((detail["lost_units"] > 0).sum()),
                "experienced_stockout": bool((detail["lost_units"] > 0).any()),
            }
        )

    results = pd.DataFrame(rows)
    summary = {
        "replications": int(replications),
        "days_per_replication": int(days),
        "mean_demand": float(results["total_demand"].mean() / days),
        "mean_sales": float(results["total_sales"].mean() / days),
        "mean_lost_units": float(results["total_lost_units"].mean()),
        "stockout_probability": float(results["experienced_stockout"].mean()),
        "mean_ending_stock": float(results["ending_stock"].mean()),
        "mean_number_of_orders": float(results["number_of_orders"].mean()),
        "mean_stockout_days": float(results["stockout_days"].mean()),
    }
    return summary, results
