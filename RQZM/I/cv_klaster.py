# =========================================================
# Evaluasi Klaster: Coefficient of Variation (CV)
# Rustiadi's Quantitative Zoning Method (RQZM)
#
# CV = (std / |mean|) * 100%
# Dihitung menggunakan nilai ASLI (sebelum standarisasi)
# agar interpretasi CV tetap stabil dan bermakna.
# =========================================================

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import statistics as stt
import warnings
warnings.filterwarnings('ignore')


# =========================================================
# KONFIGURASI
# =========================================================
INPUT_SHP = r"RQZM/1/data/jbmur.shp"
OUTPUT_CSV = r"RQZM/1/data/cv_klaster.csv"
OUTPUT_SUMMARY_CSV = r"RQZM/1/data/cv_summary.csv"

N_CLUSTERS = 3
RANDOM_STATE = 42

zi_columns = [
    "Z1i", "Z2i", "Z3i", "Z4i", "Z5i", "Z6i",
    "Z7i", "Z8i", "Z9i", "Z10i", "Z11i",
]
bobot_list = [0.5, 1, 2, 4]


# =========================================================
# 1. Membaca shapefile dan menyimpan nilai Zi asli
# =========================================================
jbmur = gpd.read_file(INPUT_SHP)

# Simpan nilai asli SEBELUM standarisasi untuk perhitungan CV
jbmur_asli = jbmur[zi_columns].copy()


# =========================================================
# 2. Membangun koordinat berbobot (sama seperti script.py)
# =========================================================
jbmur["centroid_x"] = jbmur.geometry.centroid.x
jbmur["centroid_y"] = jbmur.geometry.centroid.y

minX = jbmur.centroid_x.min()
stdevX = stt.stdev(jbmur.centroid_x)
minY = jbmur.centroid_y.min()
stdevY = stt.stdev(jbmur.centroid_y)

conts_2, conts_05 = 2, 0.5
jbmur["X_aksen"] = np.power((np.power((jbmur.centroid_x - minX), conts_2) / (stdevX * stdevY)), conts_05)
jbmur["Y_aksen"] = np.power((np.power((jbmur.centroid_y - minY), conts_2) / (stdevX * stdevY)), conts_05)

for b in bobot_list:
    suffix = str(b).replace('.', '')
    jbmur[f"X_bobot{suffix}"] = np.power(b, conts_05) * jbmur.X_aksen
    jbmur[f"Y_bobot{suffix}"] = np.power(b, conts_05) * jbmur.Y_aksen


# =========================================================
# 3. Standarisasi Zi untuk clustering (bukan untuk CV)
# =========================================================
jbmur[zi_columns] = StandardScaler().fit_transform(jbmur[zi_columns])


# =========================================================
# 4. Fungsi menghitung CV per klaster per variabel
# =========================================================
def hitung_cv_klaster(jbmur_asli_vals, labels, zi_cols, n_clusters):
    """
    Menghitung CV menggunakan nilai ASLI (sebelum standarisasi).
    Mengembalikan:
        df_cv      : DataFrame CV per variabel per klaster (%)
        mean_cv    : Rata-rata CV keseluruhan (%)
        max_cv     : CV maksimum (indikator variabel paling tidak homogen)
    """
    df_temp = jbmur_asli_vals.copy()
    df_temp['cluster'] = labels

    records = []
    for c in range(n_clusters):
        subset = df_temp[df_temp['cluster'] == c][zi_cols]
        for col in zi_cols:
            mean_val = subset[col].mean()
            std_val  = subset[col].std()
            # CV tidak bermakna jika mean sangat dekat nol
            if abs(mean_val) > 1e-9:
                cv = (std_val / abs(mean_val)) * 100
            else:
                cv = np.nan
            records.append({
                "klaster": c,
                "variabel": col,
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "cv_persen": round(cv, 2) if not np.isnan(cv) else np.nan,
            })

    df_cv_long = pd.DataFrame(records)

    # Pivot menjadi tabel lebar: baris = variabel, kolom = klaster
    df_cv_wide = df_cv_long.pivot(index="variabel", columns="klaster", values="cv_persen")
    df_cv_wide.columns = [f"CV_Klaster{c} (%)" for c in df_cv_wide.columns]
    df_cv_wide = df_cv_wide.reindex(zi_cols)  # urutan variabel asli

    mean_cv = df_cv_long["cv_persen"].mean(skipna=True)
    max_cv  = df_cv_long["cv_persen"].max(skipna=True)

    return df_cv_wide, df_cv_long, mean_cv, max_cv


# =========================================================
# 5. Loop evaluasi untuk setiap bobot spasial
# =========================================================
all_cv_long  = []
summary_rows = []

print("=" * 65)
print("  EVALUASI COEFFICIENT OF VARIATION (CV) PER KLASTER")
print("  (CV dihitung dari nilai variabel Zi ASLI / sebelum standarisasi)")
print("=" * 65)

for b in bobot_list:
    suffix   = str(b).replace('.', '')
    features = zi_columns + [f"X_bobot{suffix}", f"Y_bobot{suffix}"]
    X        = jbmur[features].values

    km     = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)

    df_cv_wide, df_cv_long, mean_cv, max_cv = hitung_cv_klaster(
        jbmur_asli, labels, zi_columns, N_CLUSTERS
    )

    # Ukuran klaster
    cluster_sizes = pd.Series(labels).value_counts().sort_index().to_dict()

    print(f"\n{'─'*65}")
    print(f"  Bobot Spasial (alpha) = {b}")
    print(f"  Ukuran klaster: { {f'K{k}': v for k, v in cluster_sizes.items()} }")
    print(f"  Rata-rata CV keseluruhan : {mean_cv:.2f}%")
    print(f"  CV maksimum              : {max_cv:.2f}%")
    print(f"\n  Tabel CV per Variabel per Klaster:\n")
    print(df_cv_wide.to_string())

    # Tambahkan kolom info ke df_cv_long
    df_cv_long["bobot_spasial"] = b
    all_cv_long.append(df_cv_long)

    summary_rows.append({
        "bobot_spasial"        : b,
        "mean_cv_persen"       : round(mean_cv, 2),
        "max_cv_persen"        : round(max_cv, 2),
        **{f"n_klaster_{k}": cluster_sizes.get(k, 0) for k in range(N_CLUSTERS)},
    })

    # Interpretasi cepat
    if mean_cv < 15:
        interp = "✅ Sangat Homogen"
    elif mean_cv < 30:
        interp = "⚠️  Homogenitas Sedang"
    else:
        interp = "❌ Heterogen"
    print(f"\n  Interpretasi: {interp} (mean CV {mean_cv:.2f}%)")


# =========================================================
# 6. Export ke CSV
# =========================================================
df_all_cv = pd.concat(all_cv_long, ignore_index=True)
df_all_cv.to_csv(OUTPUT_CSV, index=False)

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)

print(f"\n{'='*65}")
print(f"  Detail CV disimpan  : {OUTPUT_CSV}")
print(f"  Ringkasan CV disimpan: {OUTPUT_SUMMARY_CSV}")
print(f"{'='*65}")

# =========================================================
# 7. Tampilkan tabel perbandingan ringkasan
# =========================================================
print("\n  TABEL PERBANDINGAN RINGKASAN:\n")
print(df_summary.to_string(index=False))
print()
