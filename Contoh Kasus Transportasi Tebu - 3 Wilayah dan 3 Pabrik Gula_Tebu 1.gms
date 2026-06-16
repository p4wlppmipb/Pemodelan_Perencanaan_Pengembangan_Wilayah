* Contoh Kasus Transportasi Tebu - 3 Wilayah dan 3 Pabrik Gula
* Data dari penelitian Wahid - Didit Okta Pribadi & Muhammad Wahid

Sets
 i wilayah tebu / banyuwangi, jember, kediri /
 j pabrik gula   / semboro, candi, krebet-baru / ;

Parameters
 a(i) kapasitas produksi tebu per tahun (ton)
 / banyuwangi   29661
   jember       46786
   kediri      131079 /
 
 b(j) kapasitas giling pabrik gula (ton per tahun)
 / semboro       600000
   candi         411000
   krebet-baru  1560000 / ;

Table d(i,j) jarak antar wilayah ke pabrik (km)
            semboro candi krebet-baru
 banyuwangi   84.5  187.7       174.4
 jember       20.1  129.9       109.9
 kediri      153.1   75.7        65.6 ;

Scalar f biaya_angkut_per_ton_per_km / 2280 / ;

Parameter c(i,j) biaya_transportasi_per_ton_dalam_ribu_rupiah ;
 c(i,j) = f * d(i,j) / 1000 ;

Variables
 x(i,j) jumlah_tebu_dikirim_dalam_ton
 z total_biaya_transportasi_dalam_juta_rupiah ;
 
Positive Variable x ;

Equations
 cost fungsi_tujuan
 supply batas_produksi_wilayah_i
 demand memenuhi_kapasitas_giling_pabrik_j ;

cost .. z =e= sum((i,j), c(i,j)*x(i,j)) ;
supply(i) .. sum(j, x(i,j)) =l= a(i) ;
demand(j) .. sum(i, x(i,j)) =g= b(j) ;

Model transport_tebu /all/ ;
Solve transport_tebu using lp minimizing z ;
Display x.l, x.m ;