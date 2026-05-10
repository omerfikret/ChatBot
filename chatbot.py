"""
============================================================
  Makine Öğrenmesi Tabanlı Türkçe ChatBot  v2.0
  ─────────────────────────────────────────────
  Pandas │ TF-IDF │ NumPy │ Cosine Similarity
  + Anahtar Kelime Tabanlı Cümle Üretimi (Bigram Markov)
============================================================
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re, sys, random


# ══════════════════════════════════════════════════════════
#  BÖLÜM 1 ─ VERİ YÜKLEME VE TEMİZLEME  (Pandas)
# ══════════════════════════════════════════════════════════

def veri_yukle(csv_yolu: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_yolu, encoding="utf-8")
    except FileNotFoundError:
        sys.exit(f"[HATA] '{csv_yolu}' bulunamadı.")
    except Exception as e:
        sys.exit(f"[HATA] CSV okunamadı: {e}")

    df.columns = df.columns.str.strip().str.lower()

    if "girdi" not in df.columns or "cevap" not in df.columns:
        sys.exit("[HATA] CSV'de 'girdi' ve 'cevap' sütunları olmalı.")

    # Pandas vektörel temizlik — tek geçiş, RAM dostu
    df = df.dropna(subset=["girdi", "cevap"]).copy()
    df["girdi"]      = df["girdi"].str.strip()
    df["cevap"]      = df["cevap"].str.strip()
    df["girdi_norm"] = df["girdi"].apply(_normalize)
    df = (df.drop_duplicates(subset="girdi_norm", keep="first")
            .reset_index(drop=True))

    print(f"[BİLGİ] {len(df)} eğitim cümlesi yüklendi.")
    return df


def _normalize(metin: str) -> str:
    metin = str(metin).lower().strip()
    metin = re.sub(r"[^\w\s]", " ", metin)
    metin = re.sub(r"\s+", " ", metin)
    return metin


# ══════════════════════════════════════════════════════════
#  BÖLÜM 2 ─ TF-IDF VEKTÖRLEŞTİRME  (Scikit-learn)
# ══════════════════════════════════════════════════════════

def vektorlestir(df: pd.DataFrame):
    vec = TfidfVectorizer(
        analyzer="char_wb",   # karakter n-gram, yazım hatalarına dayanıklı
        ngram_range=(1, 3),
        min_df=1,
        sublinear_tf=True,    # log(tf) → sık karakterleri bastır
        max_features=5000,    # bellek tavanı
    )
    matris = vec.fit_transform(df["girdi_norm"])
    print(f"[BİLGİ] TF-IDF matrisi: {matris.shape[0]} x {matris.shape[1]}")
    return vec, matris


# ══════════════════════════════════════════════════════════
#  BÖLÜM 3 ─ BİGRAM CÜMLE ÜRETİCİ  (Markov-light)
# ══════════════════════════════════════════════════════════

class CumleUretici:
    """
    Cevap cümlelerinden bigram (kelime çifti) tablosu öğrenir.
    Verilen bir anahtar kelimeden başlayarak yeni cümle üretir.

    Neden ML'yi aşmaz?
      Bigram modeli istatistiksel dil modellemedir ve makine
      öğrenmesinin bir alt dalıdır. GPT gibi büyük modeller
      bunun derin sinir ağı versiyonudur. Küçük veriyle
      hafıza dostu çalışır.
    """

    def __init__(self):
        self.bigram: dict = defaultdict(list)   # {kelime: [sonraki, ...]}
        self.baslar: list = []                  # cümle başı kelimeleri

    def egit(self, df: pd.DataFrame):
        for cumle in df["cevap"]:
            kelimeler = str(cumle).split()
            if len(kelimeler) < 2:
                continue
            self.baslar.append(kelimeler[0])
            for i in range(len(kelimeler) - 1):
                self.bigram[kelimeler[i].lower()].append(kelimeler[i + 1])
        print(f"[BİLGİ] Bigram tablosu: {len(self.bigram)} eşsiz kelime")

    def uret(self, anahtar: str, maks: int = 12) -> str | None:
        ak = anahtar.lower()
        baslangic = None

        if ak in self.bigram:
            baslangic = ak
        else:
            for k in self.bigram:
                if ak in k or k in ak:
                    baslangic = k
                    break

        if not baslangic:
            return None

        cumle = [baslangic]
        simdiki = baslangic
        for _ in range(maks - 1):
            secenekler = self.bigram.get(simdiki.lower(), [])
            if not secenekler:
                break
            simdiki = random.choice(secenekler)
            cumle.append(simdiki)
            if simdiki[-1] in ".!?":
                break

        metin = " ".join(cumle)
        if not metin[-1] in ".!?":
            metin += "."
        return metin.capitalize()


# ══════════════════════════════════════════════════════════
#  BÖLÜM 4 ─ CEVAP BULMA  (NumPy + Cosine)
# ══════════════════════════════════════════════════════════

_BELIRSIZLIK = [
    "Bunu tam anlayamadım, farklı şekilde sorabilir misiniz?",
    "Bu konuda yeterli bilgim yok ama öğrenmeye çalışıyorum!",
    "Üzgünüm, sizi anlayamadım. Daha açık anlatır mısınız?",
    "Henüz bu konuda eğitilmedim. Başka soru sorabilirsiniz.",
]

def cevap_bul(girdi, df, vec, matris, uretici, esik=0.12, top_k=3, uretim_modu=False):
    norm = _normalize(girdi)
    v = vec.transform([norm])
    skorlar = cosine_similarity(v, matris)[0]          # NumPy dizisi

    en_iyi = np.argsort(skorlar)[::-1][:top_k]        # büyükten küçüğe
    en_iyi_skor = skorlar[en_iyi[0]]

    if en_iyi_skor < esik:
        return random.choice(_BELIRSIZLIK)

    ana_cevap = df.loc[en_iyi[0], "cevap"]

    # Üretim modu: bigram ile ek cümle
    if uretim_modu:
        kelimeler = [k for k in norm.split() if len(k) > 3]
        for k in kelimeler:
            uretilen = uretici.uret(k)
            if uretilen and uretilen.lower() != ana_cevap.lower():
                return f"{ana_cevap}\n  + Üretilen: {uretilen}"

    # Düşük güven → 2. eşleşmeyi ekle
    if en_iyi_skor < 0.5 and len(en_iyi) > 1:
        ikinci_skor = skorlar[en_iyi[1]]
        if ikinci_skor >= esik * 0.8:
            ikinci = df.loc[en_iyi[1], "cevap"]
            if ikinci != ana_cevap:
                return f"{ana_cevap} Ayrıca: {ikinci}"

    return ana_cevap


# ══════════════════════════════════════════════════════════
#  BÖLÜM 5 ─ ANA DÖNGÜ
# ══════════════════════════════════════════════════════════

YARDIM = """
  /uretim ac|kapat  → Bigram cümle üretimini aç/kapat
  /uret <kelime>    → O kelimeden cümle üret
  /istatistik       → Model bilgilerini göster
  q veya /cikis     → Çıkış
"""

def main():
    print("\n" + "="*52)
    print("  Makine Ogrenmesi Tabanli Turkce ChatBot v2.0")
    print("  Pandas | TF-IDF | NumPy | Bigram Uretici")
    print("="*52 + "\n")

    csv_yolu = sys.argv[1] if len(sys.argv) > 1 else "egitim_verisi_birlesik.csv"

    df          = veri_yukle(csv_yolu)
    vec, matris = vektorlestir(df)
    uretici     = CumleUretici()
    uretici.egit(df)

    print(f"\n  Egitim satiri  : {len(df)}")
    print(f"  TF-IDF ozellik : {matris.shape[1]}")
    print(f"  Bigram kelime  : {len(uretici.bigram)}")
    print("\nChatbot hazir! /yardim ile komutlari goruntuleyin.")
    print("-"*52)

    uretim_modu = False

    while True:
        try:
            girdi = input("Sen : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGorusuruz!")
            break

        if not girdi:
            continue
        if girdi.lower() in ("q", "/cikis", "exit"):
            print("Bot : Gorusuruz!")
            break
        if girdi.lower() == "/yardim":
            print(YARDIM); continue
        if girdi.lower() == "/istatistik":
            print(f"  Satir={len(df)}  Ozellik={matris.shape[1]}  Bigram={len(uretici.bigram)}")
            continue
        if "/uretim" in girdi.lower():
            uretim_modu = "ac" in girdi.lower() or "aç" in girdi.lower()
            print(f"Bot : Uretim modu {'ACIK' if uretim_modu else 'KAPALI'}")
            continue
        if girdi.lower().startswith("/uret "):
            k = girdi[6:].strip()
            u = uretici.uret(k)
            print(f"Bot : {u or 'Bu kelimeden cümle uretemedi.'}")
            print("-"*52); continue

        cevap = cevap_bul(girdi, df, vec, matris, uretici, uretim_modu=uretim_modu)
        print(f"Bot : {cevap}")

if __name__ == "__main__":
    main()
