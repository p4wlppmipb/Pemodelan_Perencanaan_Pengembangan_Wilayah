* Model Supply Chain Tebu 2 Tingkat (Wilayah Produksi ke Pabrik Gula)
* Data dari penelitian Wahid - Didit Okta Pribadi & Muhammad Wahid

Sets
    p   wilayah produksi tebu / banyuwangi, jember, kediri /
    pg  pabrik gula           / semboro, candi, krebet-baru /
    market                   / jakarta, bandung, surabaya / ;

* Parameter dari data Wahid
Parameters
    supply(p)    total produksi tebu per tahun (ton)
                 / banyuwangi   29661
                   jember       46786
                   kediri      131079 /

    cap(pg)      kapasitas giling pabrik gula (ton tebu per tahun)
                 / semboro       600000
                   candi         411000
                   krebet-baru  1560000 /

    demand(market)   kebutuhan pasar gula (ton per tahun)
                 / jakarta   5000
                   bandung   3000
                   surabaya  4000 / ;

* Jarak wilayah tebu ke pabrik gula (km)
Table jarak_tebu_pabrik(p,pg)
            semboro candi krebet-baru
 banyuwangi   84.5  187.7       174.4
 jember       20.1  129.9       109.9
 kediri      153.1   75.7        65.6 ;

* Jarak pabrik gula ke pasar (km)
Table jarak_pabrik_pasar(pg,market)
            jakarta bandung surabaya
 semboro       850    1000      250
 candi         800     950      150
 krebet-baru   700     850       80 ;

* Biaya transportasi (ribu rupiah per ton)
Parameter cost_tebu_pabrik(p,pg)  biaya angkut tebu ke pabrik ;
Parameter cost_gula_pasar(pg,market) biaya angkut gula ke pasar ;

cost_tebu_pabrik(p,pg) = 2280 * jarak_tebu_pabrik(p,pg) / 1000 ;
cost_gula_pasar(pg,market) = 2500 * jarak_pabrik_pasar(pg,market) / 1000 ;

* Variabel keputusan
Variables
    x(p,pg)     jumlah tebu dari wilayah ke pabrik (ton)
    y(pg,market) jumlah gula dari pabrik ke pasar (ton)
    z           total biaya transportasi (juta rupiah) ;

Positive Variables x, y ;

* Persamaan
Equations
    supply_con(p)           batas produksi tebu
    capacity_con(pg)        batas kapasitas giling pabrik
    demand_con(market)      pemenuhan permintaan pasar
    balance_pabrik(pg)      keseimbangan tebu masuk dan gula keluar
    cost_def                fungsi tujuan ;

supply_con(p)..
    sum(pg, x(p,pg)) =l= supply(p) ;

capacity_con(pg)..
    sum(p, x(p,pg)) =l= cap(pg) ;

demand_con(market)..
    sum(pg, y(pg,market)) =e= demand(market) ;

balance_pabrik(pg)..
    sum(p, x(p,pg)) * 0.075 =g= sum(market, y(pg,market)) ;

cost_def..
    z =e= sum((p,pg), cost_tebu_pabrik(p,pg) * x(p,pg)) + 
         sum((pg,market), cost_gula_pasar(pg,market) * y(pg,market)) ;

* Model dan pemecahan
Model supply_chain_tebu /all/ ;
Solve supply_chain_tebu using lp minimizing z ;

Display x.l, y.l, z.l ;
Display x.m, y.m ;