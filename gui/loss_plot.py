"""Il grafico della loss, disegnato con QPainter.

Mezzo milione di iterazioni non si disegnano punto per punto: per ogni
colonna di pixel si prende minimo e massimo delle iterazioni che ci cadono e
si traccia la banda fra i due. E' la stessa riduzione che
ModelBase.get_loss_history_preview fa per la striscia dentro la finestra
cv2, riscritta qui perche' gui/ non importa models/ -- e isolata dal disegno
in `bande` apposta, per poterla provare senza guardare lo schermo.

I buchi restano buchi: dove non cade nessuna iterazione non si interpola, o
la curva mostrerebbe punti che non esistono. Un valore che non e' un numero
finito e' un buco allo stesso titolo -- ed e' la *seconda* difesa contro di
lui, non la prima: la prima sta in `gui.loss_source`, dove entra e viene
ricordato. Sono due momenti diversi, non ridondanza. Un CSV gia' avvelenato
da una corsa precedente arriva qui all'apertura della scheda, senza che
nessun evento sia passato dalla prima difesa, e `paintEvent` e' un metodo
virtuale chiamato da Qt: un'eccezione la' dentro non risale a nessuno, PyQt5
chiama `qFatal` e con la finestra se ne vanno tutti i training aperti.
"""
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


def bande(iters, valori, larghezza, primo, ultimo):
    """Per ogni colonna di pixel, (minimo, massimo) o None se e' vuota.

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
    if larghezza <= 0:
        return []
    risultato = [None] * larghezza
    if not iters:
        return risultato
    # `span` e' l'ampiezza della finestra in numero di posizioni (non la
    # differenza fra estremi).
    span = max(1, ultimo - primo + 1)
    ancorato = span < larghezza
    for iterazione, valore in zip(iters, valori):
        #Un buco resta un buco, e il controllo di finestra non basta a
        #fermare un NaN: ogni confronto con lui e' falso, quindi
        #`iterazione < primo` lo lascia passare e l'indice di colonna
        #diventa lui stesso.
        if not numero_finito(iterazione) or not numero_finito(valore):
            continue
        if iterazione < primo or iterazione > ultimo:
            continue
        offset = iterazione - primo
        if ancorato:
            colonna = offset * (larghezza - 1) // (span - 1) if span > 1 else 0
        else:
            colonna = offset * larghezza // span
        corrente = risultato[colonna]
        if corrente is None:
            risultato[colonna] = (valore, valore)
        else:
            risultato[colonna] = (min(corrente[0], valore), max(corrente[1], valore))
    return risultato


class LossPlot(QWidget):
    """Le serie della loss, una banda per colonna di pixel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._iters = []
        self._serie = []
        self._intervallo = 0
        self._cursore = None
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
        """
        tenuti = [(i, r) for i, r in zip(iters, serie) if iterazione_utilizzabile(i)]
        self._iters = [i for i, _ in tenuti]
        self._serie = [r for _, r in tenuti]
        self.update()

    def imposta_intervallo(self, n):
        self._intervallo = n
        self.update()

    def imposta_cursore(self, iterazione):
        """L'iterazione a cui il pannello e' fermo, o None in diretta."""
        self._cursore = iterazione
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

    def paintEvent(self, _event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(SFONDO))
        larghezza, altezza = self.width(), self.height()
        primo, ultimo = self.finestra()
        colonne = self._colonne()
        if not colonne or larghezza < 2 or altezza < 2:
            return

        #Qui il controllo di finestra basta da solo, e vale la pena dire
        #perche': `primo <= nan` e' falso e `inf <= ultimo` anche, quindi
        #un'iterazione non finita resta fuori senza bisogno di guardarla.
        #Dentro `bande`, dove non c'e' nessun confronto a coprirla, la
        #guardia esplicita serve invece davvero.
        dentro = [(i, r) for i, r in zip(self._iters, self._serie)
                  if primo <= i <= ultimo]
        if not dentro:
            return
        #La scala si prende dai soli valori disegnabili. `max` non e' una
        #difesa: su [nan, 0.3] torna il primo elemento, perche' ogni
        #confronto con NaN e' falso -- quindi il NaN passa o non passa a
        #seconda dell'ordine delle righe. Senza nessun valore finito non
        #c'e' scala, e senza scala non si disegna.
        finiti = [v for _, r in dentro for v in r if numero_finito(v)]
        if not finiti:
            return
        massimo = max(finiti) or 1.0

        p.setPen(QPen(QColor(GRIGLIA), 1))
        for frazione in (0.25, 0.5, 0.75):
            y = int(altezza * frazione)
            p.drawLine(0, y, larghezza, y)

        iters_dentro = [i for i, _ in dentro]
        for indice, _serie in enumerate(colonne):
            valori = [r[indice] for _, r in dentro]
            p.setPen(QPen(QColor(COLORI[indice % len(COLORI)]), 1))
            for x, banda in enumerate(bande(iters_dentro, valori, larghezza, primo, ultimo)):
                if banda is None:
                    continue
                y_alto = altezza - int(banda[1] / massimo * (altezza - 1))
                y_basso = altezza - int(banda[0] / massimo * (altezza - 1))
                p.drawLine(x, y_alto, x, y_basso)

        if self._cursore is not None:
            p.setPen(QPen(QColor(CURSORE), 1, Qt.DashLine))
            p.drawLine(larghezza - 1, 0, larghezza - 1, altezza)
