"""La striscia delle tacche: dove sono i fratelli lungo lo scorrimento.

Il problema che risolve: dopo un sort i volti dello stesso frame finiscono
a migliaia di celle di distanza, e una cornice verde su una cella fuori
schermo non aiuta nessuno. La striscia comprime l'intera cartella in
un'altezza di poche centinaia di pixel e ci segna una tacca per fratello,
piu' un riquadro che dice quale porzione si sta guardando.

La geometria sta in tre funzioni pure, provabili senza costruire un
widget, e il widget non fa aritmetica per conto suo: e' lo stesso taglio
di gui/faceset/heatmap.py, che tiene «il conto, senza widget».

Le righe sono quelle VISIBILI (il modello filtrato), non quelle della
cartella: la striscia deve mappare cio' che la griglia mostra adesso.
"""
from PyQt5.QtCore import QRect, pyqtSignal
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import QWidget

from gui import numeri

COLORE_TACCA = QColor(80, 220, 140)
COLORE_BANDA = QColor(255, 255, 255, 40)
ALTEZZA_TACCA = 3


def posizione_tacca(riga, totale, altezza):
    """La y della tacca della riga `riga`, o None se non c'e' niente da
    mappare.

    Con una riga sola la tacca sta in mezzo: la formula generale
    dividerebbe per `totale - 1`, cioe' per zero, e la cartella con un
    volto solo esiste.
    """
    if totale <= 0 or altezza <= 0:
        return None
    if totale == 1:
        return altezza // 2
    riga = max(0, min(totale - 1, riga))
    return int(round(riga * (altezza - 1) / float(totale - 1)))


def riga_alla_y(y, totale, altezza):
    """La riga piu' vicina a `y`: l'inverso di `posizione_tacca`.

    Il risultato si stringe fra 0 e `totale - 1` prima di uscire: il clic
    puo' arrivare da un pixel di bordo, e una riga fuori intervallo
    diventerebbe un `QModelIndex` non valido tre chiamate piu' in la'.
    """
    if totale <= 0 or altezza <= 0:
        return None
    if totale == 1:
        return 0
    riga = int(round(y * (totale - 1) / float(altezza - 1)))
    return max(0, min(totale - 1, riga))


def banda_visibile(prima, ultima, totale, altezza):
    """(y_alto, y_basso) della porzione a schermo, alta almeno un pixel.

    Su 50 000 volti una schermata e' una frazione invisibile dell'altezza:
    senza il minimo la banda sparirebbe proprio sulle cartelle per cui la
    striscia serve.
    """
    if totale <= 0 or altezza <= 0:
        return None
    alto = posizione_tacca(prima, totale, altezza)
    basso = posizione_tacca(ultima, totale, altezza)
    if alto is None or basso is None:
        return None
    if basso <= alto:
        basso = alto + 1
    return alto, min(basso, altezza)


class StrisciaTacche(QWidget):
    """Sottile, a larghezza fissa, accanto alla griglia."""

    riga_scelta = pyqtSignal(int)

    LARGHEZZA = 12

    def __init__(self, parent=None):
        super().__init__(parent)
        self._totale = 0
        self._righe = []
        self._banda = None
        self.setFixedWidth(self.LARGHEZZA)

    def imposta(self, totale, righe):
        """Quante righe ha la vista adesso, e quali sono dei fratelli.

        Solo le righe che sono numeri interi utilizzabili da Qt restano: gli
        altri si scartano in silenzio. Un None, una stringa o un float('nan')
        dentro righe non solleva e non produce una tacca fuori scala.
        """
        self._totale = int(totale or 0)
        buone = []
        for riga in righe or ():
            if numeri.intero_qt_utilizzabile(riga):
                buone.append(riga)
        self._righe = buone
        self.update()

    def imposta_banda(self, prima, ultima):
        """Imposta i bordi visibili della banda. Se non sono utilizzabili,
        azzera la banda invece di salvarne una malformata.

        Un None, una stringa o un float('nan') passato come prima/ultima
        non solleva e non produce una banda fuori scala nel paintEvent.
        """
        if numeri.intero_qt_utilizzabile(prima) and \
                numeri.intero_qt_utilizzabile(ultima):
            self._banda = (prima, ultima)
        else:
            self._banda = None
        self.update()

    #override
    def mousePressEvent(self, event):
        riga = riga_alla_y(event.pos().y(), self._totale, self.height())
        if riga is not None:
            self.riga_scelta.emit(riga)
        event.accept()

    #override
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        altezza = self.height()
        if self._banda is not None:
            banda = banda_visibile(self._banda[0], self._banda[1],
                                   self._totale, altezza)
            if banda is not None:
                painter.fillRect(QRect(0, banda[0], self.width(),
                                       banda[1] - banda[0]), COLORE_BANDA)
        for riga in self._righe:
            y = posizione_tacca(riga, self._totale, altezza)
            if y is None:
                continue
            painter.fillRect(QRect(1, max(0, y - ALTEZZA_TACCA // 2),
                                   self.width() - 2, ALTEZZA_TACCA),
                             COLORE_TACCA)
