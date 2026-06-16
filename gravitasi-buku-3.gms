* ========================================================================
* Model Gravitasi dengan Kendala Ganda (Kendala Supply-Demand)
* Bobot Alpha = 1.7, Beta = 0.5, b = -0.6
* Buku Pemodelan Perencanaan Pengembangan Wilayah
* ========================================================================

SETS
I wilayah asal /A, B, C/ ;

ALIAS (I,J) ;

PARAMETERS
O(I) Kapasitas wilayah asal ke-i
         / A   100
           B   120
           C   140 /

D(J) Kapasitas wilayah tujuan ke-j
         / A   80
           B   150
           C   130 / ;

TABLE K(I,J) jarak

      A         B         C
A     2         7         2
B     7         2         6
C     2         6         2 ;

PARAMETER F(I,J) fungsi jarak;
F(I,J) = 1/K(I,J);

POSITIVE VARIABLES X(I,J);

VARIABLES
A(I)    Koefisien daya dorong
B(J)    Koefisien daya tarik
X(I,J)  Aliran dugaan
Z       Jumlah kuadrat selisih 1
ZZ      Jumlah kuadrat selisih 2
P(I)    Supply dugaan
S(J)    Demand dugaan ;


EQUATION
OBJ               Fungsi Tujuan 1
OBJ2              Fungsi Tujuan 2
PD                Supply dugaan
SD                Demand dugaan
Aliran            Aliran dugaan
Aliran2           Aliran dugaan 2 ;

OBJ..             Z=E=sum(I,sqr(O(I)-P(I)));
OBJ2..            ZZ=E=sum(I,sqr(O(I)-P(I)))+ sum(J,sqr(D(J)-S(J))) ;
PD(I)..           P(I)=E=sum (J,X(I,J)) ;
SD(J)..           S(J)=E=sum (I,X(I,J)) ;
Aliran(I,J)..     X(I,J)=E=A(I)*O(I)*D(J)*F(I,J);
Aliran2(I,J)..    X(I,J)=E=A(I)*O(I)**1.7*B(J)*D(J)**0.5*F(I,J)**-0.6; B.L(J)=1;
Model Gravi /OBJ, PD, Aliran / ;
Model Grav /OBJ2, PD, SD, Aliran2/ ;

OPTION NLP   = MINOS ;
OPTION RESLIM   = 9000 ;
OPTION ITERLIM  = 100000 ;

SOLVE GRAVI USING NLP MINIMIZING Z ;
SOLVE GRAV USING NLP MINIMIZING ZZ ;
DISPLAY F, X.L, P.L, S.L, A.L, B.L;