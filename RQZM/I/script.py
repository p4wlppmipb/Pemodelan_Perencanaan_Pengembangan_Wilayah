# Script hasil konversi dari gambar kode pada dokumen
# Rustiadi's Quantitative Zoning Method (Spatial Clustering I)

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stt

from sklearn import cluster
from sklearn.preprocessing import StandardScaler

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


# =========================================================
# KONFIGURASI PATH
# Ganti path berikut sesuai lokasi file shapefile Anda
# =========================================================
INPUT_SHP = r"RQZM/1/data/jbmur.shp"
OUTPUT_SHP = r"RQZM/1/data/jbmur_result.shp"


# =========================================================
# 1. Membaca file shapefile
# =========================================================
jbmur = gpd.read_file(INPUT_SHP)


# =========================================================
# 2. Inisiasi konstanta dan bobot kontiguitas
# =========================================================
conts_2 = 2
conts_05 = 0.5

bobot_05 = 0.5
bobot_1 = 1
bobot_2 = 2
bobot_4 = 4


# =========================================================
# 3. Mengambil nilai centroid X dan centroid Y tiap poligon
# =========================================================
jbmur["centroid_x"] = jbmur.geometry.centroid.x
jbmur["centroid_y"] = jbmur.geometry.centroid.y


# =========================================================
# 4. Menyimpan nilai minimum dan standar deviasi koordinat
# =========================================================
minX = jbmur.centroid_x.min()
stdevX = stt.stdev(jbmur.centroid_x)

minY = jbmur.centroid_y.min()
stdevY = stt.stdev(jbmur.centroid_y)


# =========================================================
# 5. Standarisasi koordinat X dan Y
# =========================================================
jbmur["X_aksen"] = np.power(
    (np.power((jbmur.centroid_x - minX), conts_2) / (stdevX * stdevY)),
    conts_05,
)

jbmur["Y_aksen"] = np.power(
    (np.power((jbmur.centroid_y - minY), conts_2) / (stdevX * stdevY)),
    conts_05,
)


# =========================================================
# 6. Menghitung koordinat berbobot kontiguitas
# Formula contoh: Xi bobot 1 = (bobot_1^0.5) * Xi'
# =========================================================
jbmur["X_bobot05"] = (np.power((bobot_05), conts_05)) * jbmur.X_aksen
jbmur["Y_bobot05"] = (np.power((bobot_05), conts_05)) * jbmur.Y_aksen

jbmur["X_bobot1"] = (np.power((bobot_1), conts_05)) * jbmur.X_aksen
jbmur["Y_bobot1"] = (np.power((bobot_1), conts_05)) * jbmur.Y_aksen

jbmur["X_bobot2"] = (np.power((bobot_2), conts_05)) * jbmur.X_aksen
jbmur["Y_bobot2"] = (np.power((bobot_2), conts_05)) * jbmur.Y_aksen

jbmur["X_bobot4"] = (np.power((bobot_4), conts_05)) * jbmur.X_aksen
jbmur["Y_bobot4"] = (np.power((bobot_4), conts_05)) * jbmur.Y_aksen

print("Processing complete...")


# =========================================================
# 7. Standarisasi variabel non-koordinat Zi
# =========================================================
zi_columns = [
    "Z1i", "Z2i", "Z3i", "Z4i", "Z5i", "Z6i",
    "Z7i", "Z8i", "Z9i", "Z10i", "Z11i",
]

jbmur[zi_columns] = StandardScaler().fit_transform(jbmur[zi_columns])


# =========================================================
# 8. Proses clustering dengan K-Means
# Jumlah cluster pada dokumen: 3
# =========================================================
km3 = cluster.KMeans(n_clusters=3, random_state=42)

km3cls05 = km3.fit(jbmur[zi_columns + ["X_bobot05", "Y_bobot05"]].values)
km3cls1 = km3.fit(jbmur[zi_columns + ["X_bobot1", "Y_bobot1"]].values)
km3cls2 = km3.fit(jbmur[zi_columns + ["X_bobot2", "Y_bobot2"]].values)
km3cls4 = km3.fit(jbmur[zi_columns + ["X_bobot4", "Y_bobot4"]].values)


# =========================================================
# 9. Visualisasi hasil cluster
# Contoh berikut memakai hasil bobot 1
# =========================================================
f, ax = plt.subplots(1, figsize=(9, 9))
jbmur.assign(cls=km3cls2.labels_).plot(
    column="cls",
    categorical=True,
    legend=True,
    linewidth=0.1,
    cmap="viridis",
    edgecolor="black",
    ax=ax,
)
ax.set_axis_off()
plt.show()


# =========================================================
# 10. Menambahkan kolom cluster ke dataframe
# =========================================================
print(km3cls2.labels_)
jbmur["kluster"] = km3cls2.labels_
print(jbmur.head())


# =========================================================
# 11. Export hasil ke shapefile
# =========================================================
jbmur.to_file(OUTPUT_SHP)



