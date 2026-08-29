"""Il grafico della loss, disegnato con QPainter.

Mezzo milione di iterazioni non si disegnano punto per punto: per ogni
colonna di pixel si prende minimo e massimo delle iterazioni che ci cadono e
si traccia la banda fra i due. E' la stessa riduzione che
ModelBase.get_loss_history_preview fa per la striscia dentro la finestra
cv2, riscritta qui perche' gui/ non importa models/ -- e isolata dal disegno
in `bande` apposta, per poterla provare senza guardare lo schermo.

**Le colonne vicine vanno unite, e la striscia cv2 non lo faceva perche' non
ne aveva bisogno**: quella mostra sempre tutta la storia sulla sua larghezza,
quindi in ogni colonna cadono decine di iterazioni e le bande, alte, si
toccano da sole -- il risultato *sembra* una curva. Qui l'intervallo si
sceglie ("Last 5 000"), e in una finestra stretta ogni colonna raccoglie due
o tre iterazioni: la banda diventa un trattino da un pixel e la curva si
legge come una nuvola di puntini staccati. Il segmento fra due colonne
consecutive occupate e' quello che rimette insieme la linea. Non e' un punto
inventato -- e' il tratto fra due campioni veri, cioe' cio' che qualunque
grafico a linee disegna; a non essere mai inventato e' il *punto*, e le
colonne dove non cade nessuna iterazione restano senza banda propria.

Un valore che non e' un numero finito e' un buco a tutti gli effetti -- ed
e' la *seconda* difesa contro di lui, non la prima: la prima sta in
`gui.loss_source`, dove entra e viene ricordato. Sono due momenti diversi,
non ridondanza. Un CSV gia' avvelenato da una corsa precedente arriva qui
all'apertura della scheda, senza che nessun evento sia passato dalla prima
difesa, e `paintEvent` e' un metodo virtuale chiamato da Qt: un'eccezione
la' dentro non risale a nessuno, PyQt5 chiama `qFatal` e con la finestra se
ne vanno tutti i training aperti.

## Il costo, che qui e' una funzionalita' e non un dettaglio

Questo widget e' l'unica cosa nella scheda che cresce con la lunghezza
dell'allenamento, e per un po' e' cresciuta addosso all'utente: a 75 500
iterazioni un evento `iter` costava **150 ms** di interfaccia ferma (12,6 ms
per ripubblicare tutta la storia piu' 70 ms di ridisegno), e gli eventi
arrivano fino a due al secondo. Le prime tre scelte -- `aggiungi_punto` per
l'evento normale, la finestra trovata con una ricerca binaria, la riduzione
in una passata per tutte le serie -- non bastavano: la passata era Python
puro su ogni punto della finestra, e con l'intervallo di default ("All
iterations") la finestra e' tutta la storia. Misurato il 2026-08-29: un
evento `iter` costava 52 ms a 75 500 iterazioni, 121 a 200 000, **374 a
500 000**; un tick del cursore dello storico 70 ms; un ridimensionamento
fino a 232 ms.

Adesso la riduzione e' numpy: i punti stanno in array (`_x`, `_y`, `_w`,
con capienza che raddoppia, cosi' `aggiungi_punto` resta un'assegnazione),
l'indice di colonna si calcola in blocco e minimo e massimo per colonna
vengono da `reduceat` sulle colonne gia' ordinate -- le iterazioni crescono,
quindi le colonne anche. Le liste `_iters`/`_serie` restano per chi le legge
(i test, `_colonne`), gli array sono cio' che il disegno usa. La cache
sull'ultima riduzione resta, per i ridisegni che non cambiano niente.

La scala verticale va dal minimo al massimo della finestra, non da zero:
ancorata a zero, una loss che scende da 0,50 a 0,28 occupava meno di meta'
dell'altezza e a training maturo diventava una linea piatta in alto. Le tre
linee di griglia portano il loro valore, senza il quale non si distingue
0,5 da 0,05.
"""
import bisect

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QWidget

from gui.numeri import MASSIMA_ITERAZIONE, iterazione_utilizzabile, numero_finito

# Gli stessi cinque del tasto 'l' della finestra cv2 (mainscripts/Trainer.py).
# 0 vuol dire "tutto".
INTERVALLI = (0, 5000, 10000, 50000, 100000)

# Le due prime serie sono src e dst: gli stessi ruoli del tema scuro.
COLORI = ("#7fb4ff", "#ffb37f", "#8fd18f", "#d18fd1")
SFONDO = "#232629"
GRIGLIA = "#3a3f44"
ETICHETTA = "#9aa0a6"
CURSORE = "#e0e0e0"

#Frazione dell'altezza lasciata sopra il massimo e sotto il minimo, cosi'
#la curva non tocca i bordi.
MARGINE = 0.06


def _in_array(iters, serie):
    """(x, y, w) da liste o array: iterazioni float64, valori float64 con
    NaN nei buchi, larghezza di ogni riga.

    Le righe possono avere lunghezze diverse (un cambio di modello sulla
    stessa cartella): la matrice e' larga quanto la riga piu' lunga e `w`
    dice fin dove ogni riga vale. Il caso comune -- tutte uguali -- passa da
    un solo `np.array`, i `None` dei buchi diventano NaN in blocco.
    """
    n = len(iters)
    x = np.asarray(iters, dtype=np.float64)
    if n == 0:
        return x, np.zeros((0, 0), dtype=np.float64), np.zeros(0, dtype=np.int64)
    if isinstance(serie, np.ndarray) and serie.dtype == np.float64 and serie.ndim == 2:
        return x, serie, np.full(n, serie.shape[1], dtype=np.int64)
    larghezze = np.fromiter((len(r) for r in serie), dtype=np.int64, count=n)
    massima = int(larghezze.max()) if n else 0
    if massima and larghezze.min() == massima:
        try:
            #Nessun buco: una conversione sola, in C. Con mezzo milione di
            #righe la via degli oggetti qui sotto costava 278 ms.
            y = np.array(serie, dtype=np.float64)
        except (TypeError, ValueError):
            grezzo = np.array(serie, dtype=object)
            y = np.where(grezzo == None, np.nan, grezzo).astype(np.float64)   # noqa: E711
    else:
        y = np.full((n, massima), np.nan, dtype=np.float64)
        for indice, riga in enumerate(serie):
            for c, v in enumerate(riga):
                y[indice, c] = np.nan if v is None else v
    return x, y, larghezze


def _riduzione_np(x, y, w, quante, larghezza, primo, ultimo):
    """Per ogni serie, minimo e massimo per colonna (NaN dove non c'e' niente).

    Due regimi, perche' sono due problemi diversi. Quando le posizioni
    possibili (`span`) sono almeno quante le colonne, si ripartisce la
    finestra in `larghezza` intervalli di uguale ampiezza -- ogni colonna
    copre lo stesso numero di iterazioni, senza ancorare gli estremi.
    Quando invece sono meno -- l'inizio di ogni training, quando le
    iterazioni scritte sono ancora poche -- la ripartizione uniforme
    lascerebbe vuoto il bordo destro (l'ultima iterazione non arriva mai
    alla colonna `larghezza - 1`): li' si ancora la prima posizione alla
    colonna 0 e l'ultima alla colonna `larghezza - 1`, perche' ogni
    iterazione ha spazio per una colonna propria e ancorare gli estremi non
    reintroduce la disuniformita' che era il difetto della formula ad
    ancoraggio applicata al caso opposto.

    Torna (minimi, massimi), matrici [quante, larghezza].
    """
    minimi = np.full((quante, larghezza), np.nan)
    massimi = np.full((quante, larghezza), np.nan)
    if len(x) == 0 or quante == 0:
        return minimi, massimi
    #Un buco resta un buco, e il controllo di finestra non basta a
    #fermare un NaN: ogni confronto con lui e' falso, quindi
    #`iterazione < primo` lo lascia passare e l'indice di colonna
    #diventa lui stesso.
    dentro = np.isfinite(x) & (x >= primo) & (x <= ultimo)
    if not dentro.any():
        return minimi, massimi
    x, y, w = x[dentro], y[dentro], w[dentro]
    # `span` e' l'ampiezza della finestra in numero di posizioni (non la
    # differenza fra estremi).
    span = max(1, ultimo - primo + 1)
    ancorato = span < larghezza
    passo = larghezza - 1 if ancorato else larghezza
    divisore = (span - 1) if ancorato else span
    offset = (x - primo).astype(np.int64)
    colonne = (offset * passo // divisore) if divisore > 0 else np.zeros(len(x), dtype=np.int64)
    for c in range(quante):
        v = y[:, c] if c < y.shape[1] else np.full(len(x), np.nan)
        validi = np.isfinite(v) & (w > c)
        if not validi.any():
            continue
        col, val = colonne[validi], v[validi]
        #Le colonne crescono con le iterazioni, quindi `reduceat` sui
        #tratti a colonna costante e' minimo e massimo per colonna in una
        #passata sola.
        inizi = np.flatnonzero(np.r_[True, col[1:] != col[:-1]])
        minimi[c, col[inizi]] = np.minimum.reduceat(val, inizi)
        massimi[c, col[inizi]] = np.maximum.reduceat(val, inizi)
    return minimi, massimi


def _a_bande(minimi, massimi):
    """La forma di sempre: per ogni serie, per ogni colonna, (min, max) o None."""
    risultato = []
    for mn, mx in zip(minimi, massimi):
        risultato.append([None if np.isnan(a) else (float(a), float(b)) for a, b in zip(mn, mx)])
    return risultato


def bande_multi(iters, serie, quante, larghezza, primo, ultimo):
    """Per ogni serie, per ogni colonna di pixel, (minimo, massimo) o None.

    Una passata sola su tutte le serie: `bande` qui sotto e' il caso a una
    serie, non una seconda implementazione. Le regole della ripartizione in
    colonne sono in `_riduzione_np`.
    """
    if larghezza <= 0 or quante <= 0:
        return [[] for _ in range(max(0, quante))]
    if not len(iters):
        return [[None] * larghezza for _ in range(quante)]
    x, y, w = _in_array(iters, serie)
    return _a_bande(*_riduzione_np(x, y, w, quante, larghezza, primo, ultimo))


def bande(iters, valori, larghezza, primo, ultimo):
    """Il caso a una serie di `bande_multi`, con `valori` gia' srotolati."""
    if larghezza <= 0:
        return []
    return bande_multi(iters, [(v,) for v in valori], 1, larghezza, primo, ultimo)[0]


def _giunzione(prima, dopo):
    """Il tratto verticale che manca fra due bande consecutive, o None.

    Sono due intervalli [min, max] su colonne vicine: se si sovrappongono la
    linea c'e' gia' e non serve niente, altrimenti si unisce l'estremo che
    guarda l'altro. Congiungere i punti medi invece degli estremi
    disegnerebbe una riga in mezzo a ogni banda alta -- corretta e inutile,
    e visibile proprio dove la curva e' gia' fitta.
    """
    if dopo[0] > prima[1]:
        return prima[1], dopo[0]
    if dopo[1] < prima[0]:
        return prima[0], dopo[1]
    return None


def _tutte_utilizzabili(iters):
    """`iterazione_utilizzabile` su tutta la lista, senza una chiamata per
    elemento: il tipo si guarda in Python (25 ms su mezzo milione), la
    grandezza in numpy. Falso al primo dubbio: chi chiama ripiega sul
    filtro elemento per elemento, che e' la regola.
    """
    if not iters:
        return True
    if not all(type(i) is int for i in iters):
        return False
    try:
        x = np.fromiter(iters, dtype=np.int64, count=len(iters))
    except OverflowError:
        return False
    return bool((x >= 0).all() and (x <= MASSIMA_ITERAZIONE).all())


def _etichetta_valore(valore):
    """Un numero della griglia, con le cifre che servono a distinguerlo."""
    if abs(valore) >= 100:
        return "%.0f" % valore
    if abs(valore) >= 10:
        return "%.1f" % valore
    if abs(valore) >= 1:
        return "%.2f" % valore
    return "%.3f" % valore


class LossPlot(QWidget):
    """Le serie della loss, una banda per colonna di pixel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._iters = []
        self._serie = []
        self._intervallo = 0
        self._cursore = None
        #Cresce a ogni cambio dei dati: e' la parte della chiave della cache
        #che una sostituzione della stessa lunghezza non potrebbe dare.
        self._versione = 0
        self._cache = None          # (chiave, quante, bande, minimo, massimo)
        self._x = np.zeros(0, dtype=np.float64)
        self._y = np.zeros((0, 0), dtype=np.float64)
        self._w = np.zeros(0, dtype=np.int64)
        self._n = 0                 # righe valide dentro gli array
        self.setMinimumHeight(120)

    def imposta_dati(self, iters, serie):
        """I punti da disegnare, filtrati **prima** di essere ricordati.

        E' la seconda difesa sulle iterazioni, con lo stesso argomento che
        vale per i valori non finiti: sono due momenti diversi, non
        ridondanza. La prima sta in `gui.loss_source`; qui arrivano anche i
        punti di un CSV letto all'apertura della scheda, senza che nessun
        evento sia passato di la'.

        Filtrare qui, e non dentro la riduzione, e' cio' che salva anche il
        calcolo della finestra: `finestra()` prende gli estremi da questa
        lista, quindi un solo valore storto in coda rende storto lo span e
        con lui l'indice di colonna di **tutte** le iterazioni sane. Una
        guardia sul singolo punto, piu' a valle, lascerebbe morire il
        processo lo stesso.

        E' la via delle **sostituzioni**: il CSV riconsegnato, un `hello`,
        un rollback. L'evento normale passa da `aggiungi_punto`, che costa
        un punto invece di tutta la storia -- vedi il docstring del modulo.
        """
        iters, serie = list(iters), list(serie)
        if _tutte_utilizzabili(iters):
            self._iters, self._serie = iters[:len(serie)], serie[:len(iters)]
        else:
            tenuti = [(i, r) for i, r in zip(iters, serie) if iterazione_utilizzabile(i)]
            self._iters = [i for i, _ in tenuti]
            self._serie = [r for _, r in tenuti]
        self._x, self._y, self._w = _in_array(self._iters, self._serie)
        self._n = len(self._iters)
        self._cambiato()

    def aggiungi_punto(self, iterazione, valori):
        """Un punto in coda. **True se e' stato accettato**, False se no.

        Rifiuta cio' che `imposta_dati` filtrerebbe, e in piu' un punto che
        non viene dopo l'ultimo: la ricerca binaria della finestra mostrata
        vale finche' le iterazioni crescono, e una coda fuori ordine la
        farebbe sbagliare in silenzio -- una parte della curva sparirebbe
        senza che niente lo dica.
        """
        if not iterazione_utilizzabile(iterazione):
            return False
        if self._iters and iterazione <= self._iters[-1]:
            return False
        valori = list(valori)
        self._iters.append(iterazione)
        self._serie.append(valori)
        self._accoda(iterazione, valori)
        self._cambiato()
        return True

    def _accoda(self, iterazione, valori):
        """Il punto nuovo dentro gli array, allargando la capienza a
        raddoppi cosi' l'operazione resta costante ammortizzata."""
        larghezza = max(self._y.shape[1], len(valori))
        if self._n == len(self._x) or larghezza > self._y.shape[1]:
            capienza = max(64, 2 * max(self._n, len(self._x)))
            x = np.zeros(capienza, dtype=np.float64)
            y = np.full((capienza, larghezza), np.nan, dtype=np.float64)
            w = np.zeros(capienza, dtype=np.int64)
            x[:self._n] = self._x[:self._n]
            y[:self._n, :self._y.shape[1]] = self._y[:self._n]
            w[:self._n] = self._w[:self._n]
            self._x, self._y, self._w = x, y, w
        self._x[self._n] = iterazione
        self._y[self._n, :] = np.nan
        for c, v in enumerate(valori):
            self._y[self._n, c] = np.nan if v is None or not numero_finito(v) else v
        self._w[self._n] = len(valori)
        self._n += 1

    def imposta_intervallo(self, n):
        self._intervallo = n
        self.update()

    def imposta_cursore(self, iterazione):
        """L'iterazione a cui il pannello e' fermo, o None in diretta.

        Muove la **finestra**, non i dati: fermarsi nello storico non toglie
        niente da qui, e non deve -- ripubblicare la storia tagliata a ogni
        scatto del cursore era il costo che rendeva lento il trascinamento,
        e disegnava esattamente la stessa curva.
        """
        self._cursore = iterazione
        self.update()

    def _cambiato(self):
        self._versione += 1
        self._cache = None
        self.update()

    def finestra(self):
        """(prima, ultima) iterazione mostrata, secondo intervallo e cursore."""
        if not self._iters:
            return (0, 0)
        ultima = self._cursore if self._cursore is not None else self._iters[-1]
        prima = self._iters[0]
        if self._intervallo:
            prima = max(prima, ultima - self._intervallo + 1)
        return (prima, ultima)

    def _colonne(self):
        """Le serie trasposte: una lista di valori per ogni colonna di loss.

        Il CSV su disco puo' accumulare righe con un numero di loss diverso
        (un cambio di modello sulla stessa cartella): il minimo su tutte le
        righe, non sulla sola finestra mostrata, e' quello che garantisce
        che ogni indice qui sotto sia valido su ogni riga, dentro o fuori
        finestra -- senza dover rifiltrare a ogni ridisegno.
        """
        if not self._serie:
            return []
        quante = min(len(r) for r in self._serie)
        return [[r[c] for r in self._serie] for c in range(quante)]

    def _finestra_indici(self, primo, ultimo):
        """(inizio, fine) dentro `_iters` per la finestra mostrata.

        Ricerca binaria, non scansione: e' cio' che rende il costo di un
        ridisegno proporzionale a quanto si guarda e non a quanto si e'
        allenato. Regge perche' `imposta_dati` e `aggiungi_punto` tengono
        insieme l'unica precondizione che serve -- le iterazioni crescono.
        """
        return (bisect.bisect_left(self._iters, primo),
                bisect.bisect_right(self._iters, ultimo))

    def _riduzione(self, larghezza, altezza, primo, ultimo):
        """(quante, bande per serie, minimo, massimo) per la finestra mostrata.

        Tenuta in cache finche' non cambia niente di cio' che la determina;
        un ridisegno qualunque (la scheda che torna in primo piano,
        un'altra finestra che passa sopra) non la rifa'. Rifarla costa
        pochi millisecondi anche su mezzo milione di punti (vedi il
        docstring del modulo), quindi cursore e ridimensionamento possono
        permettersela a ogni tick.
        """
        chiave = (self._versione, larghezza, altezza, primo, ultimo)
        if self._cache is not None and self._cache[0] == chiave:
            return self._cache[1:]
        inizio, fine = self._finestra_indici(primo, ultimo)
        if fine <= inizio:
            self._cache = (chiave, 0, [], 0.0, 0.0)
            return self._cache[1:]
        w = self._w[inizio:fine]
        #Il minimo si prende sulle righe della **finestra**, non su tutte: e'
        #quello che serve perche' ogni indice qui sotto esista su ogni riga
        #che verra' letta, e le altre non vengono lette.
        quante = int(w.min())
        minimi, massimi = _riduzione_np(self._x[inizio:fine], self._y[inizio:fine], w,
                                        quante, larghezza, primo, ultimo)
        gruppi = _a_bande(minimi, massimi)
        #La scala si prende dalle bande gia' costruite, cioe' dai soli
        #valori disegnabili: un NaN non entra ne' nel minimo ne' nel massimo.
        minimo = float(np.nanmin(minimi)) if np.isfinite(minimi).any() else 0.0
        massimo = float(np.nanmax(massimi)) if np.isfinite(massimi).any() else 0.0
        self._cache = (chiave, quante, gruppi, minimo, massimo)
        return self._cache[1:]

    def paintEvent(self, _event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(SFONDO))
        larghezza, altezza = self.width(), self.height()
        if larghezza < 2 or altezza < 2 or not self._serie:
            return
        primo, ultimo = self.finestra()
        quante, gruppi, minimo, massimo = self._riduzione(larghezza, altezza, primo, ultimo)
        if not quante:
            return
        #Senza nessun valore finito non c'e' scala, e senza scala non si
        #disegna. Con la scala a zero (tutte le loss a zero, l'inizio di
        #certi allenamenti, o un solo valore) si normalizza per uno invece
        #di dividere per zero.
        ampiezza = massimo - minimo
        if ampiezza <= 0.0:
            ampiezza = 1.0
        basso = minimo - ampiezza * MARGINE
        alto = massimo + ampiezza * MARGINE

        def y_di(valore):
            return int((alto - valore) / (alto - basso) * (altezza - 1))

        p.setPen(QPen(QColor(GRIGLIA), 1))
        for frazione in (0.25, 0.5, 0.75):
            y = int(altezza * frazione)
            p.drawLine(0, y, larghezza, y)

        #I valori delle tre linee di griglia, a destra sopra la linea: senza
        #non si distingue una loss a 0,5 da una a 0,05. Prima delle curve,
        #cosi' un'etichetta non copre mai un pixel della linea.
        p.setPen(QPen(QColor(ETICHETTA), 1))
        carattere = p.font()
        carattere.setPointSizeF(max(6.0, carattere.pointSizeF() * 0.8))
        p.setFont(carattere)
        for frazione in (0.25, 0.5, 0.75):
            y = int(altezza * frazione)
            valore = alto - (alto - basso) * y / (altezza - 1)
            p.drawText(0, y - 14, larghezza - 4, 14, Qt.AlignRight | Qt.AlignBottom,
                       _etichetta_valore(valore))

        for indice, colonne in enumerate(gruppi):
            p.setPen(QPen(QColor(COLORI[indice % len(COLORI)]), 1))
            precedente = None       # (x, banda) dell'ultima colonna occupata
            for x, banda in enumerate(colonne):
                if banda is None:
                    continue
                p.drawLine(x, y_di(banda[1]), x, y_di(banda[0]))
                if precedente is not None:
                    tratto = _giunzione(precedente[1], banda)
                    if tratto is not None:
                        p.drawLine(precedente[0], y_di(tratto[0]), x, y_di(tratto[1]))
                precedente = (x, banda)

        if self._cursore is not None:
            p.setPen(QPen(QColor(CURSORE), 1, Qt.DashLine))
            p.drawLine(larghezza - 1, 0, larghezza - 1, altezza)
