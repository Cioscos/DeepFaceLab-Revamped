"""La pellicola: la striscia di miniature con l'esito a colori.

E' la ragione per cui questo modulo esiste: il delegato dipinge la
miniatura VERA del frame, non soltanto una barretta colorata. Il colpo
d'occhio -- buio o mosso, senza aprire il frame -- e' proprio quello che
una barretta da sola non da'. La miniatura arriva da
gui.faceset.decodifica.Decodificatore, in un pool di thread, e si consegna
al thread dell'interfaccia solo via segnale Qt (`pronta`): il delegato non
tocca mai il disco, chiede l'immagine e la disegna quando arriva, come fa
gia' gui/faceset/griglia.py.

Il delegato NON chiama super().paint(): quello segfaultava sotto tema
scuro reale (vedi griglia.py).
"""
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets

from gui.estrazione import modello
from gui.faceset.decodifica import Decodificatore

LATO_MINIATURA = 64
MARGINE = 4
ALTEZZA_BARRA = 4
SPAZIO_BARRA = 2

COLORE_ZERO = QtGui.QColor(200, 80, 80)
COLORE_UNO = QtGui.QColor(90, 160, 110)
COLORE_MOLTI = QtGui.QColor(215, 170, 70)
COLORE_IGNOTO = QtGui.QColor(110, 110, 110)


def _colore(voce):
    if not isinstance(voce, dict):
        return COLORE_IGNOTO
    n = voce.get("n_volti")
    if not isinstance(n, int):
        return COLORE_IGNOTO
    if n == 0:
        return COLORE_ZERO
    if n == 1:
        return COLORE_UNO
    return COLORE_MOLTI


class _Delegato(QtWidgets.QStyledItemDelegate):
    def __init__(self, pellicola):
        super().__init__(pellicola)
        self._pellicola = pellicola

    #override
    def paint(self, pittore, opzione, index):
        pittore.save()
        rect = opzione.rect
        pittore.fillRect(rect, QtGui.QColor(35, 35, 38))

        percorso = index.data(modello.RUOLO_PERCORSO)
        cella = QtCore.QRect(rect.left() + MARGINE, rect.top() + MARGINE,
                             LATO_MINIATURA, LATO_MINIATURA)
        immagine = self._pellicola.immagine_per(percorso)
        if immagine is not None:
            pittore.drawImage(cella, immagine)
        else:
            # In volo o fallita: la barra dell'esito resta comunque
            # leggibile, solo il riquadro della miniatura e' vuoto.
            pittore.fillRect(cella, QtGui.QColor(60, 60, 64))

        voce = index.data(modello.RUOLO_VOCE)
        barra = QtCore.QRect(cella.left(), cella.bottom() + SPAZIO_BARRA,
                             cella.width(), ALTEZZA_BARRA)
        pittore.fillRect(barra, _colore(voce))

        if opzione.state & QtWidgets.QStyle.State_Selected:
            pittore.setPen(QtGui.QPen(QtGui.QColor(120, 190, 250), 2))
            pittore.drawRect(rect.adjusted(1, 1, -2, -2))
        pittore.restore()

    #override
    def sizeHint(self, _opzione, _index):
        return QtCore.QSize(LATO_MINIATURA + MARGINE * 2,
                            LATO_MINIATURA + MARGINE * 2 + SPAZIO_BARRA + ALTEZZA_BARRA)


class Pellicola(QtWidgets.QListView):
    frame_scelto = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.decodificatore = Decodificatore(self)
        self.decodificatore.pronta.connect(self._su_immagine_pronta)
        self.setFlow(QtWidgets.QListView.LeftToRight)
        self.setWrapping(False)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        # Sempre presente, mai "AsNeeded": una cartella ha migliaia di frame,
        # e se lo spazio non e' riservato quando i frame sono pochi (come in
        # un test) la vista lo regala alla riga, che si stira di nuovo.
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.setItemDelegate(_Delegato(self))
        self.clicked.connect(self._su_click)
        # Verticalmente Fixed: il sizeHint qui sotto e' l'altezza vera, e
        # senza questa policy il layout gliene concederebbe di piu' avendone.
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Fixed)

    def immagine_per(self, percorso):
        if not isinstance(percorso, Path):
            return None
        immagine = self.decodificatore.in_cache(percorso, LATO_MINIATURA)
        if immagine is None:
            self.decodificatore.richiedi(percorso, LATO_MINIATURA)
        return immagine

    def _su_immagine_pronta(self, *_args):
        self.viewport().update()

    def _su_click(self, index):
        percorso = index.data(modello.RUOLO_PERCORSO)
        if isinstance(percorso, Path):
            self.frame_scelto.emit(percorso)

    def selezione(self):
        return [i.data(modello.RUOLO_PERCORSO)
                for i in self.selectedIndexes()
                if isinstance(i.data(modello.RUOLO_PERCORSO), Path)]

    def _altezza_utile(self):
        """La cella del delegato piu' cio' che la vista le mette attorno.

        La barra di scorrimento orizzontale c'e' sempre -- una cartella ha
        migliaia di frame -- e la sua altezza cambia con la scala
        tipografica: si chiede a lei invece di scrivere un numero, o alla
        scala xlarge la striscia taglierebbe la barra colorata.
        """
        cella = self.itemDelegate().sizeHint(
            QtWidgets.QStyleOptionViewItem(), QtCore.QModelIndex()).height()
        return (cella + 2 * self.frameWidth()
                + self.horizontalScrollBar().sizeHint().height())

    #override
    def sizeHint(self):
        """L'altezza vera, non i 256x192 che QAbstractScrollArea regala a
        chiunque. Misurato il 2026-08-21: con quel sizeHint ereditato la
        striscia prendeva 192 px per un contenuto alto 78, e siccome in
        ListMode con Flow.LeftToRight la riga si stira all'altezza del
        viewport, il riquadro della selezione era alto tutti e 192.
        Vincolare l'altezza chiude i due sintomi con una modifica sola."""
        base = super().sizeHint()
        return QtCore.QSize(base.width(), self._altezza_utile())
