import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import statistics as stt
import warnings
warnings.filterwarnings('ignore')

INPUT_SHP = "/Users/afanmahardika/Documents/2.Riset/BANGWIL/permodelan-perencanaan-pengembangan-wilayah/RQZM/1/data/jbmur.shp"
jbmur = gpd.read_file(INPUT_SHP)

jbmur["centroid_x"] = jbmur.geometry.centroid.x
jbmur["centroid_y"] = jbmur.geometry.centroid.y
minX = jbmur.centroid_x.min()
stdevX = stt.stdev(jbmur.centroid_x)
minY = jbmur.centroid_y.min()
stdevY = stt.stdev(jbmur.centroid_y)

conts_2 = 2
conts_05 = 0.5
jbmur["X_aksen"] = np.power((np.power((jbmur.centroid_x - minX), conts_2) / (stdevX * stdevY)), conts_05)
jbmur["Y_aksen"] = np.power((np.power((jbmur.centroid_y - minY), conts_2) / (stdevX * stdevY)), conts_05)

bobot_list = [0.5, 1, 2, 4]
jbmur["X_bobot05"] = np.power(0.5, conts_05) * jbmur.X_aksen
jbmur["Y_bobot05"] = np.power(0.5, conts_05) * jbmur.Y_aksen
jbmur["X_bobot1"] = np.power(1.0, conts_05) * jbmur.X_aksen
jbmur["Y_bobot1"] = np.power(1.0, conts_05) * jbmur.Y_aksen
jbmur["X_bobot2"] = np.power(2.0, conts_05) * jbmur.X_aksen
jbmur["Y_bobot2"] = np.power(2.0, conts_05) * jbmur.Y_aksen
jbmur["X_bobot4"] = np.power(4.0, conts_05) * jbmur.X_aksen
jbmur["Y_bobot4"] = np.power(4.0, conts_05) * jbmur.Y_aksen

zi_columns = ["Z1i", "Z2i", "Z3i", "Z4i", "Z5i", "Z6i", "Z7i", "Z8i", "Z9i", "Z10i", "Z11i"]
jbmur[zi_columns] = StandardScaler().fit_transform(jbmur[zi_columns])

print("=== EVALUASI DENGAN BUG FITUR (Kode Lama) ===")
# Sesuai kode lama yang drop satu per satu:
# km3cls05 drops: PROVINSI, KABKOT, KECAMATAN, geometry, centroid_x, centroid_y, X_aksen, Y_aksen
cols_old_05 = jbmur.drop(["PROVINSI", "KABKOT", "KECAMATAN", "geometry", "centroid_x", "centroid_y", "X_aksen", "Y_aksen"], axis=1).columns.tolist()
print(f"Jumlah fitur km3cls05 lama: {len(cols_old_05)} -> {cols_old_05}")

km3 = KMeans(n_clusters=3, random_state=42)
labels_old_05 = km3.fit_predict(jbmur[cols_old_05].values)
old_05_exploded = jbmur.copy().assign(cluster=labels_old_05).dissolve(by='cluster').explode(index_parts=True)
print(f"Total Spatial Fragments lama (bobot 0.5): {len(old_05_exploded)}")

zi_columns = ["Z1i", "Z2i", "Z3i", "Z4i", "Z5i", "Z6i", "Z7i", "Z8i", "Z9i", "Z10i", "Z11i"]
bobot_list = [0.5, 1, 2, 4]
eval_results = []

print("\n=== EVALUASI DENGAN FITUR YANG DIPERBAIKI (Sesuai Konsep RQZM) ===")
for b in bobot_list:
    suffix = str(b).replace('.', '')
    features = zi_columns + [f"X_bobot{suffix}", f"Y_bobot{suffix}"]
    X = jbmur[features].values
    
    km = KMeans(n_clusters=3, random_state=42)
    labels = km.fit_predict(X)
    
    sil = silhouette_score(X, labels)
    dbi = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    
    gdf_temp = jbmur.copy()
    gdf_temp['cluster'] = labels
    
    # Menghitung fragmentasi spasial
    dissolved = gdf_temp.dissolve(by='cluster')
    exploded = dissolved.explode(index_parts=True)
    num_fragments = len(exploded)
    
    # Homogenitas atribut non-spasial (Rata-rata standar deviasi dalam kluster untuk semua Zi)
    wc_std = 0
    for col in zi_columns:
        cluster_stds = [gdf_temp[gdf_temp['cluster'] == c][col].std() for c in range(3)]
        cluster_stds = [s for s in cluster_stds if not np.isnan(s)]
        wc_std += np.mean(cluster_stds) if cluster_stds else 0
    wc_std /= len(zi_columns)
    
    print(f"Bobot (alpha) = {b}:")
    print(f"  Fitur yang digunakan: {features}")
    print(f"  Silhouette Score: {sil:.4f}")
    print(f"  Davies-Bouldin Index: {dbi:.4f}")
    print(f"  Calinski-Harabasz: {ch:.4f}")
    print(f"  Jumlah Fragmen Spasial (lebih kecil = lebih kontigu): {num_fragments} (dari min 3)")
    print(f"  Rata-rata Std Dev Atribut (lebih kecil = lebih homogen): {wc_std:.4f}")
    print("-" * 50)

#  Simpan ke list hasil
    eval_results.append({
        "bobot_spasial": b,
        "silhouette_score": round(sil, 4),
        "davies_bouldin_index": round(dbi, 4),
        "calinski_harabasz_score": round(ch, 4),
        "jumlah_fragmen_spasial": num_fragments,
        "rerata_std_dev_atribut": round(wc_std, 4)
    })


# print("=== HASIL EVALUASI MODEL CLUSTER ===")
# for b in bobot_list:
#     suffix = str(b).replace('.', '')
#     # Perbaikan fitur: Hanya ambil Zi dan koordinat yang sesuai dengan bobot
#     features = zi_columns + [f"X_bobot{suffix}", f"Y_bobot{suffix}"]
#     X = jbmur[features].values
    
#     # Fit KMeans
#     km = KMeans(n_clusters=3, random_state=42)
#     labels = km.fit_predict(X)
    
#     # 1. Hitung Metrik Statistik
#     sil = silhouette_score(X, labels)
#     dbi = davies_bouldin_score(X, labels)
#     ch = calinski_harabasz_score(X, labels)
    
#     # Temp dataframe untuk evaluasi spasial
#     gdf_temp = jbmur.copy()
#     gdf_temp['cluster'] = labels
    
#     # 2. Hitung Fragmentasi Spasial (Kontiguitas)
#     dissolved = gdf_temp.dissolve(by='cluster')
#     exploded = dissolved.explode(index_parts=True)
#     num_fragments = len(exploded)
    
#     # 3. Hitung Homogenitas Atribut Zi (Rata-rata Standar Deviasi)
#     wc_std = 0
#     for col in zi_columns:
#         cluster_stds = [gdf_temp[gdf_temp['cluster'] == c][col].std() for c in range(3)]
#         cluster_stds = [s for s in cluster_stds if not np.isnan(s)]
#         wc_std += np.mean(cluster_stds) if cluster_stds else 0
#     wc_std /= len(zi_columns)
    
#     # Simpan ke list hasil
#     eval_results.append({
#         "bobot_spasial": b,
#         "silhouette_score": round(sil, 4),
#         "davies_bouldin_index": round(dbi, 4),
#         "calinski_harabasz_score": round(ch, 4),
#         "jumlah_fragmen_spasial": num_fragments,
#         "rerata_std_dev_atribut": round(wc_std, 4)
#     })
    
#     print(f"Bobot Spasial (alpha) = {b}:")
#     print(f"  - Silhouette Score: {sil:.4f} (lebih tinggi lebih baik)")
#     print(f"  - Davies-Bouldin Index: {dbi:.4f} (lebih rendah lebih baik)")
#     print(f"  - Calinski-Harabasz: {ch:.4f} (lebih tinggi lebih baik)")
#     print(f"  - Jumlah Fragmen Spasial: {num_fragments} (lebih rendah = lebih menyatu)")
#     print(f"  - Rerata Standar Deviasi Atribut: {wc_std:.4f} (lebih rendah = lebih homogen)")
#     print("-" * 60)

# Simpan ke CSV
df_eval = pd.DataFrame(eval_results)
df_eval.to_csv("RQZM/1/data/evaluasi_cluster2.csv", index=False)
print("Hasil evaluasi telah disimpan ke 'RQZM/1/data/evaluasi_cluster.csv'")
