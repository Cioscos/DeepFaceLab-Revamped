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
arrivano fino a due al secondo. Non era una regressione di nessun ciclo --
misurato identico un ciclo prima, 147,5 ms contro 151,1 -- era il numero di
iterazioni che era cresciuto. Tre scelte tengono il costo legato a cio' che
si vede invece che a cio' che si e' allenato:

* `aggiungi_punto` per l'evento normale: un punto in coda, non tutta la
  storia da capo. `imposta_dati` resta per le sostituzioni vere (il CSV
  riletto, un `hello`), che sono rare.
* la finestra mostrata si trova con una ricerca binaria, non con una
  scansione: "Last 5 000" ridisegna cinquemila punti anche quando la storia
  ne ha mezzo milione.
* la riduzione si fa in **una** passata per tutte le serie insieme, e il
  risultato si tiene finche' non cambia niente di cio' che lo determina --
  ridimensionare la finestra, muovere il cursore o cambiare scheda non la
  rifanno.
"""
import bisect

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QWidget

from gui.numeri import iterazione_utilizzabile, numero_finito

# Gli stessi cinque del tasto 'l' della finestra cv2 (mainscripts/Trainer.py).
# 0 vuol dire "tutto".
INTERVALLI = (0, 5000, 10000, 50000, 100000)

# Le due prime serie sono src e dst: gli stessi ruoli del tema scuro.
COLORI = ("#7fb4ff", "#ffb37f", "#8fd18f", "#d18fd1")
SFONDO = "#232629"
GRIGLIA = "#3a3f44"
CURSORE = "#e0e0e0"


def bande_multi(iters, serie, quante, larghezza, primo, ultimo):
    """Per ogni serie, per ogni colonna di pixel, (minimo, massimo) o None.

    Una passata sola su tutte le serie: `bande` qui sotto e' il caso a una
    serie, non una seconda implementazione. Con due serie e settantacinquemila
    punti la differenza fra una passata e quattro non e' un'eleganza, e'
    l'interfaccia che risponde o non risponde.

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
    """
    if larghezza <= 0 or quante <= 0:
        return [[] for _ in range(max(0, quante))]
    risultato = [[None] * larghezza for _ in range(quante)]
    if not iters:
        return risultato
    # `span` e' l'ampiezza della finestra in numero di posizioni (non la
    # differenza fra estremi).
    span = max(1, ultimo - primo + 1)
    ancorato = span < larghezza
    passo = larghezza - 1 if ancorato else larghezza
    divisore = (span - 1) if ancorato else span
    for iterazione, riga in zip(iters, serie):
        #Un buco resta un buco, e il controllo di finestra non basta a
        #fermare un NaN: ogni confronto con lui e' falso, quindi
        #`iterazione < primo` lo lascia passare e l'indice di colonna
        #diventa lui stesso.
        if not numero_finito(iterazione):
            continue
        if iterazione < primo or iterazione > ultimo:
            continue
        offset = iterazione - primo
        colonna = (offset * passo // divisore) if divisore > 0 else 0
        for indice in range(quante):
            valore = riga[indice]
            if not numero_finito(valore):
                continue
            corrente = risultato[indice][colonna]
            if corrente is None:
                risultato[indice][colonna] = (valore, valore)
            else:
                risultato[indice][colonna] = (min(corrente[0], valore),
                                              max(corrente[1], valore))
    return risultato


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
        self._cache = None          # (chiave, quante, bande, massimo)
        self.setMinimumHeight(120)

    def imposta_dati(self, iters, serie):
        """I punti da disegnare, filtrati **prima** di essere ricordati.

        E' la seconda difesa sulle iterazioni, con lo stesso argomento che
        vale per i valori non finiti: sono due momenti diversi, non
        ridondanza. La prima sta in `gui.loss_source`; qui arrivano anche i
        punti di un CSV letto all'apertura della scheda, senza che nessun
        evento sia passato di la'.

        Filtrare qui, e non dentro `bande`, e' cio' che salva anche il
        calcolo della finestra: `finestra()` prende gli estremi da questa
        lista, quindi un solo valore storto in coda rende storto lo span e
        con lui l'indice di colonna di **tutte** le iterazioni sane. Una
        guardia sul singolo punto, piu' a valle, lascerebbe morire il
        processo lo stesso.

        E' la via delle **sostituzioni**: il CSV riconsegnato, un `hello`,
        un rollback. L'evento normale passa da `aggiungi_punto`, che costa
        un punto invece di tutta la storia -- vedi il docstring del modulo.
        """
        tenuti = [(i, r) for i, r in zip(iters, serie) if iterazione_utilizzabile(i)]
        self._iters = [i for i, _ in tenuti]
        self._serie = [r for _, r in tenuti]
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
        self._iters.append(iterazione)
        self._serie.append(list(valori))
        self._cambiato()
        return True

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
        """(quante, bande per serie, massimo) per la finestra mostrata.

        Tenuta in cache finche' non cambia niente di cio' che la determina:
        un ridimensionamento, un cambio di intervallo o di cursore la
        rifanno, un ridisegno qualunque (la scheda che torna in primo piano,
        un'altra finestra che passa sopra) no.
        """
        chiave = (self._versione, larghezza, altezza, primo, ultimo)
        if self._cache is not None and self._cache[0] == chiave:
            return self._cache[1:]
        inizio, fine = self._finestra_indici(primo, ultimo)
        righe = self._serie[inizio:fine]
        if not righe:
            self._cache = (chiave, 0, [], 0.0)
            return self._cache[1:]
        #Il minimo si prende sulle righe della **finestra**, non su tutte: e'
        #quello che serve perche' ogni indice qui sotto esista su ogni riga
        #che verra' letta, e le altre non vengono lette. Prenderlo su tutte
        #-- il CSV puo' accumulare righe con un numero di loss diverso, un
        #cambio di modello sulla stessa cartella -- costerebbe una passata
        #sull'intera storia a ogni ridisegno per lasciar decidere quante
        #curve mostrare a righe che non si stanno guardando.
        quante = min(len(r) for r in righe)
        gruppi = bande_multi(self._iters[inizio:fine], righe, quante,
                             larghezza, primo, ultimo)
        #La scala si prende dalle bande gia' costruite, cioe' dai soli
        #valori disegnabili: `max` su valori grezzi non e' una difesa -- su
        #[nan, 0.3] torna il primo elemento, perche' ogni confronto con NaN
        #e' falso, quindi il NaN passa o non passa a seconda dell'ordine
        #delle righe.
        massimo = 0.0
        for colonne in gruppi:
            for banda in colonne:
                if banda is not None and banda[1] > massimo:
                    massimo = banda[1]
        self._cache = (chiave, quante, gruppi, massimo)
        return self._cache[1:]

    def paintEvent(self, _event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(SFONDO))
        larghezza, altezza = self.width(), self.height()
        if larghezza < 2 or altezza < 2 or not self._serie:
            return
        primo, ultimo = self.finestra()
        quante, gruppi, massimo = self._riduzione(larghezza, altezza, primo, ultimo)
        if not quante:
            return
        #Senza nessun valore finito non c'e' scala, e senza scala non si
        #disegna. Con la scala a zero (tutte le loss a zero, l'inizio di
        #certi allenamenti) si normalizza per uno invece di dividere per
        #zero.
        if massimo <= 0.0:
            massimo = 1.0

        p.setPen(QPen(QColor(GRIGLIA), 1))
        for frazione in (0.25, 0.5, 0.75):
            y = int(altezza * frazione)
            p.drawLine(0, y, larghezza, y)

        def y_di(valore):
            return altezza - int(valore / massimo * (altezza - 1))

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
