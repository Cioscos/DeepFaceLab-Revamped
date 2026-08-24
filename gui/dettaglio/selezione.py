"""Chi e' selezionato, e cosa succede quando lo si trascina.

Niente Qt e nessuno stato: funzioni da insiemi a insiemi. Il punto di
tenerle qui e' che si provano senza un widget, e sono la parte che puo'
sbagliare in modi che a schermo si notano solo dopo.

I punti sono coppie in coordinate del volto allineato. A scala 1:1 sono
anche quelle del widget, quindi qui non c'e' nessuna conversione.
"""
from gui.dettaglio.gruppi import indici_gruppo
from gui.numeri import intero_qt_utilizzabile, numero_finito


def _coppia(punto):
    """(x, y) su cui si puo' fare aritmetica, o None.

    Non solleva: chi chiama sta gia' proteggendo un paintEvent o uno slot,
    e un predicato che solleva sposta il guasto dentro di lui. Servono
    ENTRAMBI i predicati di gui/numeri.py: 1e300 e' finito e fa traboccare
    l'int a 32 bit di Qt.
    """
    try:
        x, y = punto
    except (TypeError, ValueError):
        return None
    for v in (x, y):
        if not (numero_finito(v) and intero_qt_utilizzabile(v)):
            return None
    return float(x), float(y)


def indici_ammessi(aree_attive):
    """Gli indici selezionabili, data la lista delle aree accese."""
    ammessi = frozenset()
    for nome in aree_attive:
        ammessi |= indici_gruppo(nome)
    return ammessi


def nel_laccio(punti, x0, y0, x1, y1, ammessi):
    """Gli indici ammessi che cadono nel rettangolo, angoli in
    qualunque ordine: il laccio si trascina anche all'indietro."""
    sinistra, destra = (x0, x1) if x0 <= x1 else (x1, x0)
    alto, basso = (y0, y1) if y0 <= y1 else (y1, y0)
    presi = set()
    for i, punto in enumerate(punti):
        if i not in ammessi:
            continue
        coppia = _coppia(punto)
        if coppia is None:
            continue
        x, y = coppia
        if sinistra <= x <= destra and alto <= y <= basso:
            presi.add(i)
    return frozenset(presi)


def punto_vicino(punti, x, y, raggio, ammessi):
    """L'indice ammesso piu' vicino a (x, y) entro `raggio`, o None.

    Il piu' vicino e non il primo trovato: nelle zone dense -- l'angolo
    dell'occhio, il contorno della bocca -- due punti stanno a meno di un
    raggio l'uno dall'altro, e prendere il primo dell'ordine di indice
    darebbe sempre lo stesso indipendentemente da dove si e' cliccato.
    """
    migliore = None
    migliore_d2 = raggio * raggio
    for i, punto in enumerate(punti):
        if i not in ammessi:
            continue
        coppia = _coppia(punto)
        if coppia is None:
            continue
        px, py = coppia
        d2 = (px - x) ** 2 + (py - y) ** 2
        if d2 <= migliore_d2:
            migliore_d2 = d2
            migliore = i
    return migliore


def commuta(selezione, indice):
    """La selezione con `indice` aggiunto se mancava, tolto se c'era."""
    if indice in selezione:
        return frozenset(selezione) - {indice}
    return frozenset(selezione) | {indice}


def trasla(punti, selezione, dx, dy):
    """Una NUOVA lista coi soli punti selezionati spostati di (dx, dy).

    Si itera sui punti e si guarda l'appartenenza, invece di iterare sulla
    selezione: cosi' un indice fuori intervallo non solleva, e nessun
    punto puo' essere mosso due volte. Quest'ultima non e' teorica -- la
    tupla del naso in GRUPPI_68 contiene il 30 due volte, e iterare su una
    sequenza di gruppo invece che su un insieme lo muoverebbe di 2*dx.

    Un punto su cui non si puo' fare aritmetica passa intatto: non lo si
    puo' muovere, e cancellarlo cambierebbe la lunghezza della lista e
    quindi il significato di ogni indice dopo di lui.
    """
    fuori = []
    for i, punto in enumerate(punti):
        coppia = _coppia(punto) if i in selezione else None
        if coppia is None:
            fuori.append(punto)
            continue
        fuori.append([coppia[0] + dx, coppia[1] + dy])
    return fuori
