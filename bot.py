import sys, re, random, time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from scipy.sparse import hstack
from fuzzywuzzy import fuzz

# ─────────────────────────────────────────────────────────────
# BÖLÜM 1 — DÖNEM AÇIKLAMALARI (sadece /donemler komutu için)
# ─────────────────────────────────────────────────────────────
DONEM_ACIKLAMA = {
    "kuruluş":   "1299–1402  |  Osman Gazi'den Fetret Devri'ne",
    "fetret":    "1402–1453  |  Timur yenilgisinden İstanbul fethine",
    "yükseliş":  "1453–1566  |  Fatih'ten Kanuni'nin ölümüne",
    "duraklama": "1566–1699  |  II. Selim'den Karlofça'ya",
    "gerileme":  "1699–1839  |  Karlofça'dan Tanzimat'a",
    "çöküş":     "1839–1924  |  Tanzimat'tan Cumhuriyet'e",
}

# ─────────────────────────────────────────────────────────────
# BÖLÜM 2 — STOP WORDS
# ─────────────────────────────────────────────────────────────
STOP_WORDS = {
    "ve", "ile", "da", "de", "te", "ya", "ki", "bu", "bir",
    "o", "ama", "ancak", "fakat", "ne", "hem", "ya", "veya",
    "için", "ise", "bile", "dahi", "kadar", "gibi", "göre",
    "sonra", "önce", "ardından", "üzerine", "karşı", "olarak",
    "her", "hiç", "en", "çok", "az", "daha", "çok", "bazı",
    "tüm", "bütün", "son", "ilk", "yeni", "büyük", "küçük",
    "önemli", "kalıcı", "gerçek", "tam", "kısa",
    "etti", "oldu", "olarak", "aldı", "geldi", "geçti",
    "kurdu", "verdi", "çıktı", "başladı", "başlattı",
    "yapıldı", "edildi", "katıldı", "sağladı",
    "kaybetti", "yenildi", "kazandı", "bıraktı", "kaldı",
    "alındı", "indirildi", "çekildi", "durduruldu",
    "tanındı", "imzalandı", "ilan", "yaşandı", "sonuçlandı",
    "uğratıldı", "pekiştirildi", "tamamlandı", "sürdü",
    "kaldırıldı", "bastırıldı", "gönderildi",
    "i", "ı", "u", "ü", "a", "e",
    "nın", "nin", "nun", "nün",
    "nda", "nde", "nde", "nda",
    "nın", "daki", "deki", "taki", "teki",
    "ya", "ye", "yı", "yi", "yu", "yü",
    "un", "ün", "in", "ın",
    "stanbul", "ngiltere", "syanı",
    "ii", "iii",
    "hale", "getirdi",
    "ele", "geçirdi",
}

# ─────────────────────────────────────────────────────────────
# BÖLÜM 3 — METİN NORMALİZASYONU
# ─────────────────────────────────────────────────────────────
def _norm(t: str) -> str:
    t = str(t).lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)
    tokenlar = t.split()
    tokenlar = [tok for tok in tokenlar if tok not in STOP_WORDS and len(tok) > 1]
    if len(tokenlar) < 2:
        ham = str(t).lower().strip()
        ham = re.sub(r"[^\w\s]", " ", ham)
        return re.sub(r"\s+", " ", ham).strip()
    return " ".join(tokenlar).strip()

# ─────────────────────────────────────────────────────────────
# BÖLÜM 4 — VERİ YÜKLEME
# ─────────────────────────────────────────────────────────────
def veri_yukle(yol: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(yol, encoding="utf-8")
    except FileNotFoundError:
        sys.exit(f"[HATA] '{yol}' bulunamadı.")
    except Exception as e:
        sys.exit(f"[HATA] CSV okunamadı: {e}")

    if not {"soru", "cevap", "dönem"}.issubset(df.columns):
        sys.exit("[HATA] CSV'de 'soru', 'cevap', 'dönem' sütunları bulunmalıdır.")

    df = df.dropna(subset=["soru", "cevap", "dönem"])
    df["soru"]  = df["soru"].astype(str).str.strip()
    df["cevap"] = df["cevap"].astype(str).str.strip()
    df["dönem"] = df["dönem"].astype(str).str.strip().str.lower()
    df = df[df["soru"].str.len() >= 5]
    df = df[df["cevap"].str.len() >= 5]
    df["gn"] = df["soru"].apply(_norm)
    df = df[df["gn"].str.len() >= 3]
    df = df.drop_duplicates(subset=["gn"], keep="first").reset_index(drop=True)
    df = df.rename(columns={"soru": "girdi"})
    return df[["girdi", "cevap", "dönem", "gn"]]

# ─────────────────────────────────────────────────────────────
# BÖLÜM 5 — ADIM 1: LİNEAR SVC İLE DÖNEM (INTENT) SINIFLANDIRMA
#
#   Kullanıcı girdisi → TF-IDF (word + char n-gram) → LinearSVC
#   Çıktı: tahmin edilen dönem etiketi  (örn. "kuruluş", "çöküş" …)
# ─────────────────────────────────────────────────────────────
def intent_siniflandirici_egit(df: pd.DataFrame):
    """
    Tüm soruları kullanarak dönem (intent) sınıflandırıcısını eğitir.
    İki TF-IDF kanalı (kelime + karakter n-gram) birleştirilerek LinearSVC'ye verilir.
    """
    vw = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        sublinear_tf=True, max_features=4000,
        stop_words=list(STOP_WORDS),
    )
    vc = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4),
        sublinear_tf=True, max_features=4000,
    )
    Xw = vw.fit_transform(df["gn"])
    Xc = vc.fit_transform(df["gn"])
    X  = hstack([Xw, Xc], format="csr")

    svc = LinearSVC(C=1.2, max_iter=3000, class_weight="balanced")
    svc.fit(X, df["dönem"])
    return svc, vw, vc


def intent_tahmin_et(girdi_gn: str, svc, vw, vc) -> str:
    """
    Normalize edilmiş girdiyi alır; LinearSVC ile dönem etiketini döndürür.
    """
    v = hstack([vw.transform([girdi_gn]), vc.transform([girdi_gn])], format="csr")
    return svc.predict(v)[0]

# ─────────────────────────────────────────────────────────────
# BÖLÜM 6 — DÖNEM HAVUZLARI (sadece metin; vektör yok)
# ─────────────────────────────────────────────────────────────
def havuz_olustur(df: pd.DataFrame) -> dict:
    """Her dönem için soru-cevap-gn üçlülerini ayrı DataFrame'lerde saklar."""
    return {
        donem: df[df["dönem"] == donem][["girdi", "cevap", "gn"]].reset_index(drop=True)
        for donem in df["dönem"].unique()
    }

# ─────────────────────────────────────────────────────────────
# BÖLÜM 7 — ADIM 2: FUZZYWUZZY İLE HAVUZ İÇİ BENZERLİK ARAMA
#
#   Tahmin edilen dönem havuzundaki her soruya karşı
#   token_set_ratio skoru hesaplanır; eşik üstü en iyisi seçilir.
#   Bulunamazsa tüm havuzlarda arama yapılır.
# ─────────────────────────────────────────────────────────────
FUZZY_ESIK = 45   # 0-100 arası; altında cevap üretilmez

_BILMIYORUM = [
    "Bu konuda veri setimde bilgi bulamadım. Osmanlı tarihi hakkında başka bir şey sorabilirsiniz.",
    "Aradığınız bilgi veri setimde yok. Farklı bir konu dener misiniz?",
    "Bu soruya yanıt verecek kayıt bulamadım. Dönem adı ya da olay adıyla tekrar sorabilirsiniz.",
]


def _fuzzy_ara(df: pd.DataFrame, gn_sorgu: str) -> tuple[str | None, float]:
    """
    df["gn"] (normalize edilmiş sorular) üzerinde token_set_ratio ile karşılaştırma yapar.
    En yüksek skorlu sorunun cevabını döndürür; skor FUZZY_ESIK altındaysa (None, 0.0) döner.
    iterrows yerine enumerate + liste kullanılır: indeks tutarsızlığı riski olmaz.
    """
    sorular = df["gn"].tolist()
    cevaplar = df["cevap"].tolist()

    best_pos   = -1
    best_score = 0
    for i, soru_gn in enumerate(sorular):
        skor = fuzz.token_set_ratio(gn_sorgu, soru_gn)
        if skor > best_score:
            best_score = skor
            best_pos   = i

    if best_pos != -1 and best_score >= FUZZY_ESIK:
        return cevaplar[best_pos], round(best_score / 100.0, 2)
    return None, 0.0


# ─────────────────────────────────────────────────────────────
# BÖLÜM 8 — ANA CEVAP PIPELINE
#
#   girdi
#     └─► [ADIM 1] LinearSVC  → dönem tahmini
#               └─► [ADIM 2] FuzzyWuzzy (dönem havuzunda)
#                       ├─ bulundu  → dönem + cevap + skor
#                       └─ bulunamadı → FuzzyWuzzy (tüm havuzlarda)
#                               ├─ bulundu  → "?" + cevap + skor
#                               └─ bulunamadı → bilmiyorum mesajı
# ─────────────────────────────────────────────────────────────
def cevap_bul(girdi: str, havuzlar: dict, svc, vw, vc) -> tuple[str, str, float]:
    gn = _norm(girdi)

    # ADIM 1 — intent sınıflandırma
    donem = intent_tahmin_et(gn, svc, vw, vc)
    if donem not in havuzlar:
        donem = random.choice(list(havuzlar.keys()))

    # ADIM 2a — tahmin edilen dönem havuzunda fuzzy arama
    cevap, skor = _fuzzy_ara(havuzlar[donem], gn)
    if cevap is not None:
        return donem, cevap, skor

    # ADIM 2b — tüm havuzlarda fuzzy arama (fallback)
    tum_df = pd.concat(havuzlar.values(), ignore_index=True)
    cevap, skor = _fuzzy_ara(tum_df, gn)
    if cevap is not None:
        return "?", cevap, skor

    # Hiçbir şey bulunamadı
    return "?", random.choice(_BILMIYORUM), 0.0

# ─────────────────────────────────────────────────────────────
# BÖLÜM 9 — YARDIM METNİ & ANA DÖNGÜ
# ─────────────────────────────────────────────────────────────
YARDIM = """
  /donemler          Tüm dönemleri ve satır sayılarını göster
  /goster <dönem>    O döneme ait örnek soruları listele
  /info              Model istatistikleri
  q / /cikis         Çıkış

  Örnek sorular:
    Osman Gazi ne zaman beyliği kurdu?
    Fatih İstanbul'u nasıl fethetti?
    Karlofça Antlaşması ne zaman imzalandı?
    Tanzimat Fermanı nedir?
    Çanakkale Savaşı hakkında ne biliyorsun?
"""


def main():
    print("\n" + "=" * 60)
    print("Osmanlı Tarihi ChatBot")
    print("=" * 60 + "\n")

    yol = sys.argv[1] if len(sys.argv) > 1 else "egitim_verisi_temiz.csv"

    t0 = time.perf_counter()
    print("[…] Veri yükleniyor…")
    df = veri_yukle(yol)
    print(f"[✓] {len(df)} soru-cevap çifti yüklendi")

    print("[…] Intent sınıflandırıcısı eğitiliyor (LinearSVC)…")
    svc, vw, vc = intent_siniflandirici_egit(df)
    dagilim = df["dönem"].value_counts()
    print(f"[✓] Dönem dağılımı:\n{dagilim.to_string()}")

    print("[…] Dönem havuzları oluşturuluyor…")
    havuzlar = havuz_olustur(df)
    print(f"[✓] {len(havuzlar)} dönem havuzu hazır")
    print(f"[✓] Toplam süre: {time.perf_counter() - t0:.2f}s\n")
    print("Yardım için /yardim  |  Çıkış için q")
    print("-" * 60)

    while True:
        try:
            g = input("Sen : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGörüşmek üzere!")
            break

        if not g:
            continue
        gl = g.lower()

        if gl in ("q", "/cikis", "exit"):
            print("Bot : Görüşmek üzere!")
            break

        if gl in ("/yardim", "/help"):
            print(YARDIM)
            continue

        if gl == "/info":
            print(
                f"  Toplam soru  : {len(df)}\n"
                f"  Dönem sayısı : {len(havuzlar)}\n"
                f"  Kelime vocab : {len(vw.vocabulary_)}\n"
                f"  Char vocab   : {len(vc.vocabulary_)}\n"
                f"  Fuzzy eşik   : {FUZZY_ESIK}"
            )
            continue

        if gl == "/donemler":
            print("\n  Dönem Dağılımı:")
            for et, acik in DONEM_ACIKLAMA.items():
                sayi = dagilim.get(et, 0)
                bar  = "█" * (sayi // 3)
                print(f"  {et:<12} {acik:<45} {sayi:>3} soru   {bar}")
            print()
            continue

        if gl.startswith("/goster "):
            et = gl[8:].strip()
            if et not in havuzlar:
                print(
                    f"Bot : '{et}' dönemi bulunamadı. "
                    f"Geçerli dönemler: {list(havuzlar.keys())}"
                )
            else:
                ornekler = havuzlar[et]["girdi"].head(6).tolist()
                print(f"\n  [{et}] dönemi örnek sorular:")
                for o in ornekler:
                    print(f"   • {o}")
                print()
            continue

        # ── Ana pipeline: SVC intent → fuzzy cevap ──────────────
        etiket, yanit, skor = cevap_bul(g, havuzlar, svc, vw, vc)
        etiket_goster = f"[{etiket}|{skor:.2f}]" if etiket != "?" else f"[?|{skor:.2f}]"
        print(f"Bot {etiket_goster}: {yanit}\n")


if __name__ == "__main__":
    main()