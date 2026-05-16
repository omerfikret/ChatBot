"""
Osmanlı Tarihi ChatBot v1.0
Tek sütunlu bilgi CSV'si | Dönem bazlı etiketleme | TF-IDF (word+char) | LinearSVC
"""

import sys, re, random, time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, vstack

# ─────────────────────────────────────────────────────────────
# BÖLÜM 1 — DÖNEM ETİKETLERİ (yalnızca veri setindeki dönemler)
# ─────────────────────────────────────────────────────────────
#
# Veri setindeki tarih aralıklarına göre belirlenen 6 dönem:
#
#   kuruluş     : 1299 – 1402  (Osman Gazi → Fetret öncesi)
#   fetret      : 1402 – 1453  (Timur yenilgisi → İstanbul'un fethine kadar)
#   yükseliş    : 1453 – 1566  (Fatih → Kanuni'nin ölümü)
#   duraklama   : 1566 – 1699  (II. Selim → Karlofça)
#   gerileme    : 1699 – 1839  (Karlofça → Tanzimat öncesi)
#   çöküş       : 1839 – 1924  (Tanzimat → Halifeliğin kaldırılması)
#
# Her etikete ait anahtar kelimeler yalnızca veri setinden türetilmiştir.

ETIKETLER = {
    "kuruluş": [
        "osman gazi", "söğüt", "beylik", "1299", "bağımsızlık",
        "yenişehir", "koyunhisar", "bafeus", "bizans", "1302",
        "dimbos", "köse mihal", "gemlik", "mudanya", "orhan gazi",
        "bursa", "1326", "akçe", "pelekanon", "iznik", "medrese",
        "izmit", "karesi", "rumeli", "çimpe", "gelibolu",
        "edirne", "yeniçeri", "i. murat", "sırpsındığı", "trakya",
        "kosova", "niğbolu", "yıldırım bayezid", "anadolu birliği",
        "kuruluş", "ilk başkent", "1300", "1306", "1313",
        "1321", "1331", "1337", "1345", "1352", "1354", "1361",
        "1362", "1364", "1369", "1389", "1396",
    ],
    "fetret": [
        "fetret devri", "timur", "ankara muharebesi", "1402",
        "taht kavgası", "şehzade", "1413", "i. mehmed", "musa çelebi",
        "ii. murad", "1421", "istanbul kuşatma", "1422",
        "varna", "1444", "ii. kosova", "1448", "fetret",
        "1451", "birlik yeniden",
    ],
    "yükseliş": [
        "fatih", "istanbul fethi", "1453", "bizans sona", "ayasofya",
        "sırbistan seferi", "1454", "trabzon", "1461",
        "otlukbeli", "uzun hasan", "1473", "kırım", "kefe", "azak", "1475",
        "otranto", "1480", "ii. bayezid", "cem sultan", "1481",
        "yahudiler", "1492", "yavuz", "1512", "çaldıran", "1514",
        "mercidabık", "suriye", "filistin", "1516",
        "ridaniye", "mısır fethi", "halifelik", "1517",
        "cezayir", "barbaros", "1519",
        "kanuni", "1520", "belgrad", "1521", "rodos", "1522",
        "mohaç", "macar", "budin", "1526", "viyana", "1529",
        "preveze", "1538", "bağdat", "irak", "1535",
        "macaristan", "1541", "akdeniz", "sokollu", "1546",
        "amasya antlaşması", "1555", "malta", "1565",
        "zigetvar", "ii. selim", "1566", "imparatorluk", "yükseliş",
    ],
    "duraklama": [
        "kıbrıs", "1570", "1571", "inebahti", "donanma yenilgisi",
        "tunus", "1574", "iii. murad", "kafkaslar", "1578",
        "ferhat paşa", "1590", "avusturya", "1593",
        "haçova", "iii. mehmed", "1596", "kanije", "1600",
        "i. ahmed", "celali", "1603", "zitvatorok", "1606",
        "köprülü", "1656", "erdel", "1658",
        "girit", "venedik", "1669", "kamaniçe", "podolya", "bucaş", "1672",
        "viyana kuşatması", "merzifonlu", "1683",
        "budin kaybı", "1686", "ii. süleyman", "1687",
        "zenta", "1697", "karlofça", "1699", "duraklama",
    ],
    "gerileme": [
        "istanbul antlaşması", "1700", "azak rusya",
        "edirne vakası", "ii. mustafa", "iii. ahmed", "1703",
        "prut", "petro", "1711",
        "mora geri", "1715", "pasarofça", "banat", "1718",
        "lale devri", "matbaa",
        "patrona halil", "i. mahmud", "1730",
        "belgrad antlaşması", "1739",
        "kerden", "1746",
        "iii. osman", "iii. mustafa", "1754",
        "osmanlı-rus savaşı", "1768", "çeşme", "rus donanması", "1770",
        "küçük kaynarca", "kırım bağımsız", "1774",
        "yaş antlaşması", "dinyester", "1787", "1792",
        "iii. selim", "nizam-ı cedid", "1789",
        "napolyon", "mısır seferi", "1798",
        "sırp isyanı", "balkanlar", "1804",
        "kabakçı mustafa", "1807",
        "alemdar", "ii. mahmud", "1808",
        "bükreş antlaşması", "besarabya", "1812",
        "yunan isyanı", "mora bağımsızlık", "1821",
        "vaka-i hayriye", "yeniçeri kaldırıldı", "1826",
        "navarin", "1827",
        "edirne antlaşması", "yunanistan bağımsız", "1829",
        "cezayir fransız", "1830",
        "takvim-i vekayi", "1831",
        "kütahya antlaşması", "kavalalı", "1833",
        "nizip", "donanma teslim", "1839",
        "gerileme",
    ],
    "çöküş": [
        "tanzimat", "can mal namus güvencesi", "hukuk eşitliği",
        "kırım savaşı", "1853", "1856",
        "ıslahat fermanı", "gayrimüslim haklar",
        "paris antlaşması", "karadeniz tarafsız",
        "abdülaziz", "demiryolu", "dış borç", "1861",
        "genç osmanlılar", "meşrutiyet mücadelesi",
        "osmanlı iflası", "1875",
        "v. murad", "ii. abdülhamid", "1876",
        "i. meşrutiyet", "kanun-u esasi",
        "93 harbi", "rusya yenilgisi", "1877", "1878",
        "ayastefanos", "büyük bulgaristan", "berlin kongresi",
        "sırbistan bağımsız", "romanya bağımsız", "bosna avusturya",
        "kıbrıs ingiliz", "1878",
        "düyun-u umumiye", "1881",
        "tunus fransız", "mısır ingiliz", "1882",
        "hamidiye alayları", "ermeni katliam", "1894",
        "osmanlı-yunan savaşı", "girit özerklik", "1897",
        "ii. meşrutiyet", "jön türk", "1908",
        "bosna ilhak", "bulgaristan bağımsız",
        "31 mart", "abdülhamid tahttan", "v. mehmed reşad", "1909",
        "trablusgarp", "libya", "italya", "uşi", "1911",
        "balkan savaşları", "makedonya", "arnavutluk", "selanik", "1912",
        "bab-ı ali baskını", "ittihat terakki", "1913",
        "almanya ittifakı", "1914",
        "i. dünya savaşı",
        "çanakkale savaşı", "1915",
        "tehcir kanunu", "ermeni göç",
        "arap isyanı", "şerif hüseyin", "1916",
        "sarıkamış", "enver paşa", "1917",
        "mondros", "teslim", "1918",
        "vi. mehmed", "vahideddin",
        "izmir işgali", "milli mücadele", "1919",
        "mustafa kemal", "samsun",
        "sevr antlaşması", "1920", "tbmm",
        "sakarya muharebesi", "1921",
        "büyük taarruz", "yunan yenilgisi", "1922",
        "saltanat kaldırıldı",
        "lozan antlaşması", "türkiye cumhuriyeti", "1923",
        "cumhuriyet ilan", "29 ekim",
        "halifelik kaldırıldı", "1924", "hanedan sürgün",
        "çöküş", "osmanlı sona erdi",
    ],
}

# ─────────────────────────────────────────────────────────────
# BÖLÜM 2 — STOP WORDS (veri setinden türetildi)
# ─────────────────────────────────────────────────────────────
#
# Üç katmanlı liste:
#
# [A] TÜRKÇE GENEL BAĞLAÇLAR / EDATLAR / ZAMİRLER
#     "ve", "ile", "da", "de" gibi hiçbir anlam taşımayan kelimeler.
#     Bunlar TF-IDF'de yüksek frekans nedeniyle IDF ağırlığını düşürür,
#     gürültü olarak dönem sınırını bulanıklaştırır.
#
# [B] OSMANLICADAKİ YÜKSEK-FREKANS EYLEM KÖKLERİ
#     "etti", "oldu", "aldı", "geldi", "geçti" gibi neredeyse her
#     cümlede geçen, hangi döneme ait olduğunu söylemeyen fiiller.
#     Bunlar TF-IDF vektörünü tüm cümleler arasında düzleştirir;
#     cosine benzerliği anlamsız yüksek çıkar.
#
# [C] MORFOLOJİK ARTIKLAR (ek halleri)
#     Tokenizasyon sonrası kelimenin başından kopan harf/hece kalıntıları.
#     Örn: "İstanbul'u" → "stanbul" + "u", "İngiltere" → "ngiltere".
#     Bunlar vocab'ı kirletir ve char n-gram'ları yanıltır.
#
# NOT: "osmanlı", "sultan", "paşa", "gazi" gibi Osmanlı'ya özgü
#      kelimeler stop words'e ALINMADI — dönem içi benzerlik için
#      bu kelimeler hâlâ ayırt edici sinyal taşır.

STOP_WORDS = {
    # [A] Türkçe bağlaçlar / edatlar / zamirler
    "ve", "ile", "da", "de", "te", "ya", "ki", "bu", "bir",
    "o", "ama", "ancak", "fakat", "ne", "hem", "ya", "veya",
    "için", "ise", "bile", "dahi", "kadar", "gibi", "göre",
    "sonra", "önce", "ardından", "üzerine", "karşı", "olarak",
    "her", "hiç", "en", "çok", "az", "daha", "çok", "bazı",
    "tüm", "bütün", "son", "ilk", "yeni", "büyük", "küçük",
    "önemli", "kalıcı", "gerçek", "tam", "kısa",

    # [B] Yüksek frekanslı anlamsız eylem kökleri
    "etti", "oldu", "olarak", "aldı", "geldi", "geçti",
    "kurdu", "verdi", "çıktı", "başladı", "başlattı",
    "yapıldı", "edildi", "katıldı", "sağladı",
    "kaybetti", "yenildi", "kazandı", "bıraktı", "kaldı",
    "alındı", "indirildi", "çekildi", "durduruldu",
    "tanındı", "imzalandı", "ilan", "yaşandı", "sonuçlandı",
    "uğratıldı", "pekiştirildi", "tamamlandı", "sürdü",
    "kaldırıldı", "bastırıldı", "gönderildi",
    # NOT: fethetti/fethederek/fethiyle BIRAKILDI — anlam taşır

    # [C] Morfolojik artıklar (tokenizasyon kalıntıları)
    "i", "ı", "u", "ü", "a", "e",        # tek harf ek kalıntıları
    "nın", "nin", "nun", "nün",            # iyelik ekleri
    "nda", "nde", "nde", "nda",            # bulunma hali kalıntıları
    "nın", "daki", "deki", "taki", "teki", # sıfat fiil ekleri
    "ya", "ye", "yı", "yi", "yu", "yü",   # yönelme/belirtme ekleri
    "un", "ün", "in", "ın",               # ilgi hali ekleri
    "stanbul", "ngiltere", "syanı",        # unicode/encoding kırıkları
    "ii", "iii",                           # Osmanlı hükümdar sayıları tek başına
    "hale", "getirdi",                     # "kalıcı hale getirdi" klişesi
    "ele", "geçirdi",                      # "ele geçirdi" parçaları
}

# ─────────────────────────────────────────────────────────────
# BÖLÜM 3 — METİN NORMALİZASYONU
# ─────────────────────────────────────────────────────────────
def _norm(t: str) -> str:
    """
    Küçük harf → noktalama temizliği → stop words çıkar → boşluk normalize.
    Stop words çıkarma yalnızca bağımsız token'larda yapılır;
    'muharebesi' içindeki 'i' harfi etkilenmez.
    """
    t = str(t).lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)
    tokenlar = t.split()
    # Stop words'e tam eşleşme — tek harf kalıntılar da kaldır
    tokenlar = [tok for tok in tokenlar if tok not in STOP_WORDS and len(tok) > 1]
    # Çok kısa kalan metni (< 2 token) olduğu gibi bırak; aşırı filtreleme önlenir
    if len(tokenlar) < 2:
        # Stop words uygulamadan sadece noktalama temizliği yap
        ham = str(t).lower().strip()
        ham = re.sub(r"[^\w\s]", " ", ham)
        return re.sub(r"\s+", " ", ham).strip()
    return " ".join(tokenlar).strip()


# ─────────────────────────────────────────────────────────────
# BÖLÜM 4 — VERİ YÜKLEME (tek sütunlu CSV)
# ─────────────────────────────────────────────────────────────
CHUNK_N = 256

def veri_yukle(yol: str) -> pd.DataFrame:
    """
    CSV formatı: yalnızca tek sütun ('bilgi').
    Her satır bağımsız bir Osmanlı tarihi cümlesidir.
    girdi = cevap = o cümle (benzerlik araması yapılır).
    """
    try:
        reader = pd.read_csv(
            yol, encoding="utf-8",
            header=0, names=["bilgi"],
            chunksize=CHUNK_N, on_bad_lines="skip",
        )
    except FileNotFoundError:
        sys.exit(f"[HATA] '{yol}' bulunamadı.")

    gorduk, parclar = set(), []
    for chunk in reader:
        chunk = chunk.dropna(subset=["bilgi"])
        chunk["bilgi"] = chunk["bilgi"].astype(str).str.strip()
        # Başlık satırını ve çok kısa satırları filtrele
        chunk = chunk[chunk["bilgi"].str.lower() != "bilgi"]
        chunk = chunk[chunk["bilgi"].str.len() >= 10]
        # Her satır hem sorgu (girdi) hem yanıt (cevap) olarak kullanılır
        chunk = chunk.rename(columns={"bilgi": "girdi"})
        chunk["cevap"] = chunk["girdi"]
        chunk["gn"] = chunk["girdi"].apply(_norm)
        chunk = chunk[chunk["gn"].str.len() >= 5]
        chunk = chunk[~chunk["gn"].isin(gorduk)]
        gorduk.update(chunk["gn"].tolist())
        parclar.append(chunk[["girdi", "cevap", "gn"]])

    if not parclar:
        sys.exit("[HATA] CSV boş veya okunamadı.")

    df = pd.concat(parclar, ignore_index=True)
    df.drop_duplicates("gn", keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────
# BÖLÜM 5 — LINEAR SVC ETİKETLEME
# ─────────────────────────────────────────────────────────────
def svc_etiketle(df: pd.DataFrame):
    """
    Tohum veri seti yalnızca ETIKETLER sözlüğündeki
    dönem anahtar kelimeleriyle oluşturulur.
    """
    tohum_giris, tohum_etiket = [], []
    for etiket, kelimeler in ETIKETLER.items():
        for k in kelimeler:
            varyasyonlar = [
                k,
                f"{k} dönemi",
                f"{k} tarihi",
                f"{k} osmanlı",
            ]
            for v in varyasyonlar:
                tohum_giris.append(_norm(v))
                tohum_etiket.append(etiket)

    tum_metin = tohum_giris + df["gn"].tolist()

    vw = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        sublinear_tf=True, max_features=4000,
        stop_words=list(STOP_WORDS),   # ← stop words buraya verildi
    )
    vc = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4),
        sublinear_tf=True, max_features=4000,
        # char n-gram'da stop words uygulanmaz; _norm zaten temizledi
    )
    vw.fit(tum_metin)
    vc.fit(tum_metin)

    def _vec(metinler):
        return hstack(
            [vw.transform(metinler), vc.transform(metinler)],
            format="csr",
        )

    X_tohum  = _vec(tohum_giris)
    X_gercek = _vec(df["gn"].tolist())

    svc = LinearSVC(C=1.2, max_iter=3000, class_weight="balanced")
    svc.fit(X_tohum, tohum_etiket)

    df["etiket"] = svc.predict(X_gercek)
    return df, (vw, vc), svc


# ─────────────────────────────────────────────────────────────
# BÖLÜM 6 — ETİKET BAZLI VEKTÖR HAVUZLARI
# ─────────────────────────────────────────────────────────────
def havuz_olustur(df: pd.DataFrame, vecs):
    vw, vc = vecs
    havuzlar = {}
    for etiket in df["etiket"].unique():
        alt = df[df["etiket"] == etiket].copy().reset_index(drop=True)
        Mw = vw.transform(alt["gn"])
        Mc = vc.transform(alt["gn"])
        M  = hstack([Mw, Mc], format="csr")
        havuzlar[etiket] = {"df": alt, "M": M}
    return havuzlar


def kullanici_vec(girdi_norm: str, vecs):
    vw, vc = vecs
    return hstack(
        [vw.transform([girdi_norm]), vc.transform([girdi_norm])],
        format="csr",
    )


# ─────────────────────────────────────────────────────────────
# BÖLÜM 7 — CEVAP BULMA
# ─────────────────────────────────────────────────────────────
_BILMIYORUM = [
    "Bu konuda veri setimde bilgi bulamadım. Osmanlı tarihi hakkında başka bir şey sorabilirsiniz.",
    "Aradığınız bilgi veri setimde yok. Farklı bir konu dener misiniz?",
    "Bu soruya yanıt verecek kayıt bulamadım. Dönem adı ya da olay adıyla tekrar sorabilirsiniz.",
]

ESIK_UZUN  = 0.30   # 4+ token → temiz sorgu, yüksek eşik
ESIK_KISA  = 0.15   # 1-3 token → kısa sorgu, toleranslı eşik


def _esik(gn: str) -> float:
    """Sorgu uzunluğuna göre dinamik eşik döndür."""
    return ESIK_UZUN if len(gn.split()) >= 4 else ESIK_KISA


def _tum_havuz(havuzlar):
    dfs, Ms = [], []
    for et, h in havuzlar.items():
        d = h["df"].copy()
        d["etiket"] = et
        dfs.append(d)
        Ms.append(h["M"])
    return pd.concat(dfs, ignore_index=True), vstack(Ms, format="csr")


def cevap_bul(girdi: str, havuzlar: dict, svc, vecs, top_k: int = 3):
    """
    1. Girdi etiketini LinearSVC ile tahmin et.
    2. İlgili dönem havuzunda kosinüs benzerliği ara.
    3. Eşik altındaysa tüm havuzda fallback ara.
    4. (etiket, yanıt, skor) döndür.
    """
    gn = _norm(girdi)
    esik = _esik(gn)
    v  = kullanici_vec(gn, vecs)
    etiket = svc.predict(v)[0]

    if etiket not in havuzlar:
        etiket = random.choice(list(havuzlar.keys()))

    havuz = havuzlar[etiket]
    sk    = cosine_similarity(v, havuz["M"])[0]
    idx   = np.argsort(sk)[::-1][:top_k]
    skor  = float(sk[idx[0]])

    if skor >= esik:
        return etiket, havuz["df"].iloc[idx[0]]["cevap"], skor

    # Fallback: tüm havuzlarda ara
    tum_df, tum_M = _tum_havuz(havuzlar)
    sk2   = cosine_similarity(v, tum_M)[0]
    idx2  = np.argsort(sk2)[::-1][:top_k]
    skor2 = float(sk2[idx2[0]])

    if skor2 < esik:
        return "?", random.choice(_BILMIYORUM), 0.0

    satir = tum_df.iloc[idx2[0]]
    return satir["etiket"], satir["cevap"], skor2


# ─────────────────────────────────────────────────────────────
# BÖLÜM 8 — YARDIM & ANA DÖNGÜ
# ─────────────────────────────────────────────────────────────
YARDIM = """
  /donemler          Tüm dönemleri ve satır sayılarını göster
  /goster <dönem>    O döneme ait örnek cümleleri listele
  /info              Model istatistikleri
  q / /cikis         Çıkış

  Örnek sorular:
    Osman Gazi ne zaman beyliği kurdu?
    Fatih İstanbul'u nasıl fethetti?
    Karlofça Antlaşması ne zaman imzalandı?
    Tanzimat Fermanı nedir?
    Çanakkale Savaşı hakkında ne biliyorsun?
"""

DONEM_ACIKLAMA = {
    "kuruluş":   "1299–1402  |  Osman Gazi'den Fetret Devri'ne",
    "fetret":    "1402–1453  |  Timur yenilgisinden İstanbul fethine",
    "yükseliş":  "1453–1566  |  Fatih'ten Kanuni'nin ölümüne",
    "duraklama": "1566–1699  |  II. Selim'den Karlofça'ya",
    "gerileme":  "1699–1839  |  Karlofça'dan Tanzimat'a",
    "çöküş":     "1839–1924  |  Tanzimat'tan Cumhuriyet'e",
}


def main():
    print("\n" + "=" * 60)
    print("  Osmanlı Tarihi ChatBot  v1.0")
    print("  Tek Sütunlu CSV | Dönem Etiketleme | TF-IDF + LinearSVC")
    print("=" * 60 + "\n")

    yol = sys.argv[1] if len(sys.argv) > 1 else "egitim_verisi_temiz.csv"

    t0 = time.perf_counter()
    print("[…] Veri yükleniyor…")
    df = veri_yukle(yol)
    print(f"[✓] {len(df)} cümle yüklendi")

    print("[…] Dönem etiketlemesi yapılıyor…")
    df, vecs, svc = svc_etiketle(df)
    dagilim = df["etiket"].value_counts()
    print(f"[✓] Dönem dağılımı:\n{dagilim.to_string()}")

    print("[…] Havuzlar oluşturuluyor…")
    havuzlar = havuz_olustur(df, vecs)
    print(f"[✓] {len(havuzlar)} dönem havuzu hazır")
    print(f"[✓] Süre: {time.perf_counter() - t0:.2f}s\n")
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
                f"  Toplam cümle : {len(df)}\n"
                f"  Dönem sayısı : {len(havuzlar)}\n"
                f"  Kelime vocab : {len(vecs[0].vocabulary_)}\n"
                f"  Char vocab   : {len(vecs[1].vocabulary_)}"
            )
            continue

        if gl == "/donemler":
            print("\n  Dönem Dağılımı:")
            for et, acik in DONEM_ACIKLAMA.items():
                sayi = dagilim.get(et, 0)
                bar  = "█" * (sayi // 3)
                print(f"  {et:<12} {acik:<45} {sayi:>3} cümle  {bar}")
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
                ornekler = havuzlar[et]["df"]["girdi"].head(6).tolist()
                print(f"\n  [{et}] dönemi örnekleri:")
                for o in ornekler:
                    print(f"   • {o}")
                print()
            continue

        etiket, yanit, skor = cevap_bul(g, havuzlar, svc, vecs)
        etiket_goster = f"[{etiket}|{skor:.2f}]" if etiket != "?" else "[?]"
        print(f"Bot {etiket_goster}: {yanit}\n")


if __name__ == "__main__":
    main()