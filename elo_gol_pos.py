import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from scipy.stats import poisson,skellam
import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm
import statsmodels.formula.api as smf

###################
#st.cache ile paramatre ismi değişmedikce veri seti, program her çalıştığında yüklenmeyecek.
@st.cache(allow_output_mutation=True)
def veri_seti(dosya_adi):
    df=pd.read_csv(dosya_adi)
    veri=pd.DataFrame(df)
    veri = veri[["Date","HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
    return veri
##################


veri=veri_seti("/home/rdvn/challenge/2021-2022.csv")

###############
#veri=veri[["HomeTeam","AwayTeam","FTHG","FTAG"]]
###############

####################################
#E(A):Beklenen sonuç formülü. Takımların kazanma olasılığını verir.
def beklenen_sonuc(ev,dep):
    dr=ev-dep
    kazanma_olasiligi=(1/(10**(-dr/400)+1))
    return [np.round(kazanma_olasiligi,3),1-np.round(kazanma_olasiligi,3)]

#Beklenti değil, maç bittikten sonra görülen sonuç. A takımı kazandıysa 1, berabere kalırlarsa 0.5 gibi
#S(A)
def gercek_sonuc(ev,dep):
    if ev<dep:
        ev_kazanan=0
        deplasman_kazanan=1
    elif ev>dep:
        deplasman_kazanan=0
        ev_kazanan=1
    elif ev==dep:
        deplasman_kazanan=0.5
        ev_kazanan=0.5
    return [ev_kazanan,deplasman_kazanan]


#Elo'yu hesaplayan fonksiyon. R'(A)=R(A)+K(S(A)-E(A))
def elo_hesaplayici(ev_elo,deplasman_elo,ev_golleri,deplasman_golleri):
  k=20
  ev_gol,dep_gol=gercek_sonuc(ev_golleri,deplasman_golleri)
  kazanan_eloE,kazanan_eloD=beklenen_sonuc(ev_elo,deplasman_elo)

  elo_ev=ev_elo+k*(ev_gol-kazanan_eloE)
  elo_dep=deplasman_elo+k*(dep_gol-kazanan_eloD)

  return elo_ev,elo_dep


#####################################
#######################################

elo={}
for index,satir in veri.iterrows():
    #
    ev_sahibi=satir['HomeTeam']
    deplasman=satir['AwayTeam']
    ev_gol=satir['FTHG']
    deplasman_gol=satir['FTAG']
    # İlk maç olduğu için her takım için 1300'e sabitlendi.
    if ev_sahibi not in elo.keys():
        elo[ev_sahibi]=1300

    if deplasman not in elo.keys():
        elo[deplasman]=1300

    #Her takım için elo hesabı.
    ev_elo=elo[ev_sahibi]
    deplasman_elo=elo[deplasman]
    ev_eloY,deplasman_eloY=elo_hesaplayici(ev_elo,deplasman_elo,ev_gol,deplasman_gol)

    elo[ev_sahibi]=ev_eloY
    elo[deplasman]=deplasman_eloY

    #Veri setine yeni elo eklenmesi.
    veri.loc[index,'Ev_yeni_elo']=ev_eloY
    veri.loc[index,'Deplasman_yeni_elo']=deplasman_eloY
    veri.loc[index,'Ev_onceki_elo']=ev_elo
    veri.loc[index,'Deplasman_onceki_elo']=deplasman_elo
##################################################

################################
#date=df[["Date"]]
#veri["date"]=date
#####

maclar =veri[['Date','HomeTeam','AwayTeam','FTHG','FTAG','Deplasman_onceki_elo',"Ev_onceki_elo"]]


#Gol sayısı 6'dan küçük olmasını istedim
maclar = maclar[(maclar['FTHG']<6)&(maclar['FTAG']<6)].reset_index(drop=True)


ev=maclar[['Date','HomeTeam','FTHG','Deplasman_onceki_elo',"Ev_onceki_elo"]]\
    .rename(columns={'HomeTeam':"Team","FTHG":"Atilan_gol","Deplasman_onceki_elo":"Rakip elo","Ev_onceki_elo":"Elo"})
deplasman=maclar[['Date','AwayTeam','FTAG','Deplasman_onceki_elo',"Ev_onceki_elo"]]\
    .rename(columns={'AwayTeam':"Team","FTAG":"Atilan_gol","Deplasman_onceki_elo":"Rakip elo","Ev_onceki_elo":"Elo"})


veri1 = pd.concat([ev,deplasman],ignore_index=True).sort_values("Date").reset_index(drop=True)


#Artırımlı olarak, ortalama gol hesabı yapar. Rolling fonksiyonu hareketli ortalamayı hesaplar.
#Shift ile veri seti gezilir.
veri1["Atilan_gol_ort"]=veri1.groupby('Team')['Atilan_gol'].transform(lambda x: x.rolling(3).mean()).shift()
veri1["Atilan_gol_ort"]=veri1.groupby("Team")["Atilan_gol_ort"].shift()



#Elo farki
veri1["Elo_farki"] = veri1["Elo"] - veri1["Rakip elo"]

#Gerek duyulmayan sütünlar atıldı.
veri1=veri1.dropna()
veri1=veri1.drop(columns=["Team","Elo","Rakip elo"])

###########################################
#Model oluşturldu. Atılan_gol hedef, hareketli ortalamadan bulunan Atilan_gol_ort ve Elo_farki ile
#hesaplandı. "1",ev sahibi olmayı temsil ediyor. Benim için önemli bir parametre olmadığı için herkesi
#ev sahibi saydım.
model = smf.glm(formula=f"Atilan_gol ~ {1}  + Atilan_gol_ort + Elo_farki", data=veri1,
                        family=sm.families.Poisson()).fit()
###########################################3
def hesapla (takim1_elo,takim2_elo,ort1,ort2):
    #takim1_elo=1750
    #takim2_elo=1763.62
    #ort1=2.0
    #ort2=2.0
    takim1_data = pd.DataFrame(data={'Atilan_gol_ort':ort1,'Elo_farki':takim1_elo-takim2_elo},index=[1])
    takim2_data = pd.DataFrame(data={'Atilan_gol_ort':ort2,'Elo_farki':takim2_elo-takim1_elo},index=[1])

    takim1_gol = model.predict(takim1_data).values[0]
    takim2_gol = model.predict(takim2_data).values[0]

#### 1-5 arası gol atma olasılıkları poisson ile hesaplandı.
    takim_tahmin = [[poisson.pmf(i, takim_ort) for i in range(0, 5)] for takim_ort in [takim1_gol, takim2_gol]]


#### Matris içerisinde gol sayıları ve olasılık dağılımları yerleştirildi.
    joint_proba=np.outer(np.array(takim_tahmin[0]), np.array(takim_tahmin[1]))
#### Matris üzerinde işlemler yaparak;kazanma,kaybetme ve beraberlik olasıkları bulundu.
    return pd.Series([1-np.sum(np.triu(joint_proba, 1))-np.sum(np.diag(joint_proba)),
           np.sum(np.diag(joint_proba)),np.sum(np.triu(joint_proba, 1))],index=['Ev','Beraberlik','Deplasman'])


#######################################33

###Streamlit kısmı########

img=Image.open("/home/rdvn/Desktop/foto.webp")
st.image(img,use_column_width=True)
st.title("Futbol Mac Sonucu Olasıkları")
st.markdown("""
            Poisson modeli kullanarak üç ihtimalli maç sonucu olasıklarını bulma.
            * **Python kütüphaneleri**: pandas,streamlit,scipy,statsmodel,numpy.
            * **Veri seti**: [fbref.com/en/comps/9/Premier-League-Stats](https://fbref.com/en/comps/9/Premier-League-Stats)
            
            """)

st.title("Poisson Dağılımı")
st.markdown(" * **Belli bir zaman aralığında bir olayın kaç kez gerçekleştiğinin olasılığıdır.**")
st.markdown("* **Bir mucit kişinin çalışma hayatı boyunca patentini aldığı keşifler sayısı.**")
st.markdown("* **Yarım saat içinde bir nakliyat deposuna yükleme-boşatılma için gelen kamyon sayısı.**")
st.markdown("* **Bir maçta atılan gol sayısı.**")
img3=Image.open("/home/rdvn/Pictures/Screenshots/poisson.png")
st.image(img3,use_column_width=True)

st.title("Elo")
st.markdown("* **Oyuncu A ve oyuncu B olsun. Önce E(A) yani beklenen sonuç bulunur.**")
st.markdown("* **E(A)=1/1+10^((R(B)-R(A))/400) beklenen sonuç yani A oyuncusunun kazanma olasığıdır.**")
st.markdown("* **S(A)=Karşılaşma sonucunda alınacak puandır. A müsabakayı kazanırsa 1 puan, berabere kalırsa 0.5 ve yenilirse 0 puan alacaktır.**")
st.markdown("* **K=Ağırlık için kullanılır. Oyuncuların durumuna göre değişir, düşük seviyeli bir oyuncu yüksek seviyeli oyuncuyu yenerse K değeri büyük olacaktır.**")
st.markdown("* **R'(A)=R(A)+K(S(A)-E(A)) Formülü ile oyuncunun karşılaşma sonrası yeni değeri bulunur.**")

veri_sırala=sorted(veri["HomeTeam"].unique())
veri_sırala1=sorted(veri["AwayTeam"].unique())
takim=st.multiselect("Takım",veri_sırala,veri_sırala[:1])
secenekler=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Ev_yeni_elo',
       'Deplasman_yeni_elo', 'Ev_onceki_elo', 'Deplasman_onceki_elo', 'date']
secilen_takim=st.multiselect("Seçenekler",secenekler,secenekler)

df_secilen_takim=veri[(veri["HomeTeam"].isin(takim)) | veri["AwayTeam"].isin(takim)]

st.header("Seçilen Takımların Bigilerini Göster")
st.write("Veri boyutu: " + str(df_secilen_takim.shape[0]) + ' satır ' + str(df_secilen_takim.shape[1]) + ' sütun')
st.dataframe(df_secilen_takim)

st.header("Maç başına atılan goller ve bu gollerin ortalaması")
st.dataframe(veri1)

img1=Image.open("/home/rdvn/Pictures/Screenshots/joint.png")
st.image(img1,use_column_width=True)


def sonuc_goster():
    elo_1 = st.sidebar.number_input('Ev sahibi takımın elosu')
    st.sidebar.write('Elo: ', elo_1)

    elo_2=st.sidebar.number_input("Deplasman takımının elosu")
    st.sidebar.write("Rakip elo: ",elo_2)


    ort1=st.sidebar.number_input("Ev sahibi takımının ortalama golü")
    st.sidebar.write("Ev ortalama gol ",ort1)

    ort2=st.sidebar.number_input("Deplasman takımınının ortalama golü")
    st.sidebar.write("Rakip ortalama gol: ",ort2)


    st.sidebar.write(hesapla(elo_1,elo_2,ort1,ort2))



sonuc_goster()






