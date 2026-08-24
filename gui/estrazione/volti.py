"""I volti di un fotogramma come li usa la revisione: aritmetica pura.

**Zero import di Qt.** Da qui passano numeri che un altro processo ha letto
da un file dell'utente e serializzato in JSON, e il posto dove
morirebbero e' un paintEvent -- che chiama qFatal e si porta via il
processo con dentro ogni training aperto. Tenerli fuori da Qt vuol dire
poterli provare senza una finestra e senza la piattaforma offscreen.

Il campo `maschera` di `Volto` E' un oggetto Qt (un QImage gia' colorato)
o None, e il campo `affine` sono i sei argomenti di QTransform: li mette
dentro la pagina, questo modulo non li guarda mai. Un namedtuple non ha
bisogno di conoscere il tipo di cio' che contiene.
"""
from collections import namedtuple

from gui import numeri

Volto = namedtuple("Volto", ["percorso", "rect", "punti", "maschera",
                             "affine", "lato_allineato", "nome_maschera"])


def _disegnabile(x):
    """Entrambi i predicati, sempre: `numero_finito` da solo lascia
    passare 1e300, che e' finito e uccide comunque l'int() di un
    paintEvent con l'OverflowError della firma a 32 bit."""
    return numeri.numero_finito(x) and numeri.intero_qt_utilizzabile(x)


def rect_utilizzabile(rect):
    """(l, t, r, b) di float con l < r e t < b, o None.

    Un rettangolo degenere si scarta e non si "ripara": largo zero non e'
    ne' disegnabile ne' colpibile da un click, e inventargli un lato
    darebbe un bersaglio che non corrisponde a niente sul disco.
    """
    try:
        l, t, r, b = rect
    except (TypeError, ValueError):
        return None
    if not all(_disegnabile(v) for v in (l, t, r, b)):
        return None
    l, t, r, b = float(l), float(t), float(r), float(b)
    if r <= l or b <= t:
        return None
    return (l, t, r, b)


def punti_utilizzabili(punti):
    """Le coppie su cui si puo' fare aritmetica, in coordinate del
    fotogramma. Ogni punto si scarta per conto suo: perdere l'intera
    mascella per un punto fuori scala sarebbe peggio del buco."""
    fuori = []
    for punto in punti or ():
        try:
            x, y = punto
        except (TypeError, ValueError):
            continue
        if _disegnabile(x) and _disegnabile(y):
            fuori.append((float(x), float(y)))
    return fuori


def affine_utilizzabile(mat):
    """I sei argomenti di QTransform, **nell'ordine di Qt**, o None.

    DFL scrive la 2x3 per RIGHE -- [[a, b, c], [d, e, f]] significa
    x' = a*x + b*y + c -- mentre il costruttore di Qt prende
    (m11, m12, m21, m22, dx, dy) con x' = m11*x + m21*y + dx. La
    corrispondenza e' quindi (a, d, b, e, c, f): argomenti
    INTERLACCIATI, non nell'ordine in cui si leggono sulla matrice.
    Scriverli nell'ordine naturale non solleva niente e da' una maschera
    ruotata e traslata in modo plausibile, che e' il difetto peggiore
    disponibile qui. La convenzione e' stata verificata eseguendo:
    QTransform(2, 3, 5, 7, 11, 13).map(QPointF(1, 0)) da' (13, 16), cioe'
    m11 + dx e m12 + dy.

    None anche per una matrice SINGOLARE: la sua inversa non esiste, e
    chiedere l'inversa dentro un paintEvent per scoprirlo li' e' il posto
    sbagliato per accorgersene. Qui i sei numeri devono solo essere
    finiti, non stare in un int a 32 bit: sono coefficienti, non
    coordinate -- e' il PUNTO trasformato che dovra' entrarci, e quel
    controllo lo fa chi disegna.
    """
    try:
        (a, b, c), (d, e, f) = mat
    except (TypeError, ValueError):
        return None
    if not all(numeri.numero_finito(v) for v in (a, b, c, d, e, f)):
        return None
    if abs(float(a) * float(e) - float(b) * float(d)) < 1e-12:
        return None
    return (float(a), float(d), float(b), float(e), float(c), float(f))


def volto_da_voce(voce):
    """Un `Volto` da una voce del protocollo, o None.

    Serve almeno il percorso: senza, il volto non e' apribile, e un
    bersaglio di click che non porta da nessuna parte e' peggio di nessun
    bersaglio. Rettangolo, punti e matrice mancano uno per conto proprio.
    """
    if not isinstance(voce, dict):
        return None
    percorso = voce.get("path")
    if not isinstance(percorso, str) or not percorso:
        return None
    return Volto(percorso=percorso,
                 rect=rect_utilizzabile(voce.get("rect")),
                 punti=punti_utilizzabili(voce.get("source_landmarks")),
                 maschera=None,
                 affine=affine_utilizzabile(voce.get("mat")),
                 lato_allineato=_lato_allineato(voce.get("shape")),
                 nome_maschera=_nome_maschera(voce.get("mask")))


def _nome_maschera(nome):
    """Il NOME del file, mai un percorso: la cartella la conosce chi ha
    avviato il servizio, e un nome che contenesse un separatore
    porterebbe la lettura fuori dal workdir. Un nome cosi' si scarta."""
    if not isinstance(nome, str) or not nome:
        return None
    if "/" in nome or "\\" in nome or nome.startswith("."):
        return None
    return nome


def _lato_allineato(shape):
    """Il lato in pixel dell'allineato, da `shape` = [h, w, c], o None.

    Serve alla scala raster->allineato: la maschera dentro il JPEG e'
    ricompressa a piacere da DFLJPG.set_xseg_mask e non ha la dimensione
    dell'allineato. Senza questo numero la maschera si disegnerebbe alla
    scala del proprio raster, cioe' nel posto sbagliato -- e un allineato
    e' quadrato, quindi un lato solo basta.
    """
    try:
        lato = shape[0]
    except (TypeError, IndexError, KeyError):
        return None
    if not numeri.numero_finito(lato) or lato <= 0:
        return None
    return float(lato)


def volti_da_risposta(risposta):
    """I `Volto` di una risposta `framed`, nell'ordine del servizio.

    Una voce malformata perde se stessa e non le altre: e' cio' che uno
    Stop a meta' estrazione lascia dietro."""
    if not isinstance(risposta, dict):
        return []
    fuori = []
    for voce in risposta.get("volti") or ():
        volto = volto_da_voce(voce)
        if volto is not None:
            fuori.append(volto)
    return fuori


def percorsi(volti):
    return [v.percorso for v in volti]


def volto_al_punto(volti, x, y):
    """Il volto il cui rettangolo contiene (x, y), scandendo per AREA
    CRESCENTE.

    Per area e non per ordine di elenco: un volto piccolo dentro il
    rettangolo di uno grande -- un volto sullo sfondo, un bambino in
    braccio -- resterebbe altrimenti irraggiungibile, e non ci sarebbe
    nessun altro modo di aprirlo.
    """
    if not (numeri.numero_finito(x) and numeri.numero_finito(y)):
        return None
    candidati = []
    for volto in volti:
        r = volto.rect
        if r is None:
            continue
        l, t, rr, b = r
        if l <= x <= rr and t <= y <= b:
            candidati.append(((rr - l) * (b - t), volto))
    if not candidati:
        return None
    return min(candidati, key=lambda coppia: coppia[0])[1]
