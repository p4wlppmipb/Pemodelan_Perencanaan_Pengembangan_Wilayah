import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
import pandas as pd # Import pandas untuk memastikan kompatibilitas

# ================= KONFIGURASI PATH =================
# Ganti path ini sesuai lokasi file di komputer Anda
INPUT_SHP = 'C:/Users/Hanan/Documents/DATA/projek26/spatial_clustering/jbmur-skrip-dataset/updateDataJbmur.shp'
OUTPUT_SHP_ASSIGN = "C:/Users/Hanan/Documents/DATA/projek26/spatial_clustering/jbmur-skrip-dataset/updateDataJbmurAssignOptimize.shp"
# ====================================================

# Open file shp
print("Membaca data shapefile...")
try:
    jbmur = gpd.read_file(INPUT_SHP)
except Exception as e:
    print(f"Error membaca file: {e}")
    exit()

# ================= INISIALISASI KOLOM (DINAMIS) =================
print("Menambahkan kolom baru...")
# Kolom dasar
jbmur["NEIGHBORS"] = None
jbmur["m"] = 0
jbmur["sper_sWij"] = 0.00

# Jumlah variabel Z (Z1 sampai Z11)
num_vars = 11

# Menggunakan loop untuk membuat kolom Z secara otomatis agar kode tidak repetitif
for i in range(1, num_vars + 1):
    jbmur[f"sWijZ{i}j"] = 0.00
    jbmur[f"Z{i}i_sWijZ{i}j"] = 0.00
    jbmur[f"Z{i}_Adj2"] = 0.00

# ================= LOOP UTAMA (DIOPTIMALKAN) =================
print("Memulai pemrosesan spasial (ini mungkin memakan waktu)...")

# Menyiapkan nama kolom untuk efisiensi di dalam loop
z_cols_i = [f'Z{i}i' for i in range(1, num_vars + 1)]        # [Z1i, Z2i, ..., Z11i]
z_cols_j_sum = [f'sWijZ{i}j' for i in range(1, num_vars + 1)] # [sWijZ1j, ..., sWijZ11j]

# Menggunakan indeks spasial untuk mempercepat kueri
sindex = jbmur.sindex

for index, row in jbmur.iterrows():    
    # 1. Temukan kandidat tetangga menggunakan indeks spasial (cepat)
    possible_matches_index = list(sindex.intersection(row.geometry.bounds))
    possible_matches = jbmur.iloc[possible_matches_index]
    
    # 2. Lakukan pengecekan geometri presisi (lambat, tapi hanya pada kandidat)
    # Menggunakan logika asli: yang tidak disjoint (artinya bersinggungan/menyentuh)
    precise_neighbors = possible_matches[~possible_matches.geometry.disjoint(row.geometry)]
    
    # 3. Hapus diri sendiri dari daftar tetangga
    # (Menggunakan perbandingan index lebih aman dan cepat daripada membandingkan nama KECAMATAN)
    real_neighbors_gdf = precise_neighbors[precise_neighbors.index != index]

    # --- Isi Informasi Tetangga ---
    neighbors_names = real_neighbors_gdf.KECAMATAN.tolist()
    jbmur.at[index, "NEIGHBORS"] = ", ".join(neighbors_names)
    m_count = len(neighbors_names)
    jbmur.at[index, "m"] = m_count

    # --- Hitung Penjumlahan Variabel Z (Looping Z1-Z11) ---
    # Kita sudah punya GeoDataFrame tetangga (real_neighbors_gdf),
    # jadi kita tidak perlu melakukan query spasial lagi.
    if m_count > 0:
        for i in range(num_vars):
            col_name_i = z_cols_i[i]      # misal: Z1i
            target_col_sum = z_cols_j_sum[i] # misal: sWijZ1j
            
            current_val_i = row[col_name_i]
            
            # Terapkan filter logika asli:
            # Ambil nilai tetangga HANYA JIKA nilainya berbeda dengan nilai row saat ini.
            values_to_sum = real_neighbors_gdf[real_neighbors_gdf[col_name_i] != current_val_i][col_name_i]
            
            jbmur.at[index, target_col_sum] = float(values_to_sum.sum())

print("Pemrosesan spasial selesai.")

# ================= PERHITUNGAN VEKTOR (DIOPTIMALKAN) =================
print("Melakukan perhitungan kolom lanjutan...")

# --- PENANGANAN PEMBAGIAN DENGAN NOL ---
# Menghitung 1/m. Jika m = 0 (tidak ada tetangga), gunakan nilai 0.0 agar tidak error.
# np.where(kondisi, nilai_jika_benar, nilai_jika_salah)
jbmur['sper_sWij'] = np.where(jbmur['m'] > 0, 1 / jbmur['m'], 0.00)

# --- Loop Perhitungan Kolom Interaksi dan Adj2 ---
# PERINGATAN MATEMATIKA:
# np.sqrt(x) akan menghasilkan NaN (Not a Number) jika x < 0.
# Jika data Z Anda memungkinkan hasil perkalian interaksi menjadi negatif,
# Anda akan melihat warning dari numpy dan nilai NaN di kolom output.
# Pastikan metodologi statistik Anda mengizinkan ini.

for i in range(1, num_vars + 1):
    col_i = f'Z{i}i'
    col_j_sum = f'sWijZ{i}j'
    col_inter = f'Z{i}i_sWijZ{i}j'
    col_adj = f'Z{i}_Adj2'

    # 1. Hitung kolom interaksi (Zi * Sum(Zj))
    jbmur[col_inter] = jbmur[col_j_sum] * jbmur[col_i]

    # 2. Hitung Akar Kuadrat dari (Interaksi * (1/m))
    # Menggunakan np.sqrt pada pandas Series
    jbmur[col_adj] = np.sqrt(jbmur[col_inter] * jbmur['sper_sWij'])

print ('Processing complete.')

# ================= PROSES CLUSTERING (Telah di-uncomment) =================
print("Memulai proses K-Means Clustering...")

# Daftar kolom yang akan digunakan untuk clustering (Z1_Adj2 sampai Z11_Adj2)
cols_for_clustering = [f'Z{i}_Adj2' for i in range(1, num_vars + 1)]

# Mengambil data untuk dicluster
data_to_cluster = jbmur[cols_for_clustering]

# --- PENTING: Menangani NaN sebelum Scaling/Clustering ---
# Jika ada hasil akar kuadrat negatif (NaN), clustering akan gagal.
# Opsi penanganan: Mengisi NaN dengan 0 (asumsi sementara agar kode berjalan).
# Anda mungkin perlu meninjau kembali rumus matematikanya jika banyak muncul NaN.
if data_to_cluster.isnull().values.any():
    print("Peringatan: Ditemukan nilai NaN (kemungkinan dari akar kuadrat negatif). Mengisi NaN dengan 0 untuk clustering.")
    data_to_cluster = data_to_cluster.fillna(0)

# Standarisasi Data (Penting untuk K-Means)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_to_cluster)

# Inisialisasi dan fitting K-Means
km3 = cluster.KMeans(n_clusters=3, random_state=42) # random_state ditambahkan agar hasil konsisten
km3cls = km3.fit(data_scaled)

# Menampilkan hasil label kluster
print("Label Kluster:", km3cls.labels_)

# Menambahkan kolom "cluster" ke GeoDataFrame utama
jbmur["cluster"] = km3cls.labels_

# ================= VISUALISASI =================
print("Menampilkan plot...")
f, ax = plt.subplots(1, figsize=(10, 10))
jbmur.plot(column='cluster', categorical=True, legend=True,
          linewidth=0.5, cmap='viridis', edgecolor='black', ax=ax)
ax.set_title("Hasil K-Means Clustering (k=3) pada Variabel Spasial Tereduksi")
ax.set_axis_off()
plt.show()

# ================= MENYIMPAN HASIL =================
print(f"Menyimpan file output ke: {OUTPUT_SHP_ASSIGN}")
# Hapus kolom NEIGHBORS yang berisi list panjang sebelum menyimpan ke SHP agar tidak error pada beberapa driver GIS
# jbmur_to_save = jbmur.drop(columns=['NEIGHBORS'])
jbmur.to_file(OUTPUT_SHP_ASSIGN)
print("Selesai.")