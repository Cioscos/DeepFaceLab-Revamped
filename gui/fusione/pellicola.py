"""La striscia di miniature della sessione di fusione: stessa forma di
gui/estrazione/pellicola.py (delegato senza super().paint(), miniature
dal Decodificatore in un pool di thread), ma con lo stato della fusione
per riga invece delle voci del rapporto d'estrazione."""
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets

from gui.faceset.decodifica import Decodificatore
from gui.fusione import timeline as tl

LATO_MINIATURA = 64
MARGINE = 4
ALTEZZA_BARRA = 4
RUOLO_PERCORSO = QtCore.Qt.UserRole + 1
RUOLO_STATO = QtCore.Qt.UserRole + 2


class ModelloFrameFusione(QtCore.QAbstractListModel):
    """Un frame per riga: percorso e stato di fusione."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._percorsi = []
        self._stati = []

    def imposta(self, percorsi):
        self.beginResetModel()
        self._percorsi = list(percorsi)
        self._stati = [tl.STATO_DA_FARE] * len(self._percorsi)
        self.endResetModel()

    def imposta_stato(self, idx, stato):
        """Come `segna_fatto`/`segna_da_fare` della timeline, che dipinge lo
        stesso stato con la stessa tinta: chi non ha un volto resta rosso.
        La protezione sta qui e non nei chiamanti perche' i due widget non
        possono contraddirsi -- il rosso non tornerebbe fino a una nuova
        sessione."""
        if not isinstance(idx, int) or not (0 <= idx < len(self._stati)):
            return
        if self._stati[idx] == tl.STATO_SENZA_VOLTO and stato != tl.STATO_SENZA_VOLTO:
            return
        self._stati[idx] = stato
        indice = self.index(idx, 0)
        self.dataChanged.emit(indice, indice)

    def rowCount(self, _parent=QtCore.QModelIndex()):
        return len(self._percorsi)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._percorsi)):
            return None
        if role == RUOLO_PERCORSO:
            return self._percorsi[index.row()]
        if role == RUOLO_STATO:
            return self._stati[index.row()]
        if role == QtCore.Qt.DisplayRole:
            return self._percorsi[index.row()].name
        return None


class _Delegato(QtWidgets.QStyledItemDelegate):
    def __init__(self, pellicola):
        super().__init__(pellicola)
        self._pellicola = pellicola

    #override
    def paint(self, pittore, opzione, index):
        pittore.save()
        rect = opzione.rect
        pittore.fillRect(rect, QtGui.QColor(35, 35, 38))
        percorso = index.data(RUOLO_PERCORSO)
        cella = QtCore.QRect(rect.left() + MARGINE, rect.top() + MARGINE, LATO_MINIATURA, LATO_MINIATURA)
        immagine = self._pellicola.immagine_per(percorso)
        if immagine is not None:
            pittore.drawImage(cella, immagine)
        else:
            pittore.fillRect(cella, QtGui.QColor(60, 60, 64))
        stato = index.data(RUOLO_STATO)
        colore = tl.COLORE_PER_STATO.get(stato, tl.COLORE_PER_STATO[tl.STATO_DA_FARE])
        pittore.fillRect(QtCore.QRect(cella.left(), cella.bottom() + 2, cella.width(), ALTEZZA_BARRA), colore)
        if opzione.state & QtWidgets.QStyle.State_Selected:
            pittore.setPen(QtGui.QPen(QtGui.QColor(240, 240, 240), 2))
            pittore.drawRect(cella.adjusted(-1, -1, 1, 1))
        pittore.restore()

    #override
    def sizeHint(self, _opzione, _index):
        return QtCore.QSize(LATO_MINIATURA + 2 * MARGINE, LATO_MINIATURA + 2 * MARGINE + ALTEZZA_BARRA + 2)


class PellicolaFusione(QtWidgets.QListView):
    """La striscia orizzontale dei frame della sessione."""

    frame_scelto = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.decodificatore = Decodificatore(self)
        self.decodificatore.pronta.connect(self._su_immagine_pronta)
        self.setFlow(QtWidgets.QListView.LeftToRight)
        self.setWrapping(False)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.setItemDelegate(_Delegato(self))
        self.clicked.connect(self._su_click)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def _altezza_utile(self):
        """La cella del delegato piu' la barra di scorrimento orizzontale vera
        (sempre presente, il flusso e' orizzontale): un numero fisso al posto
        di interrogarla taglia la barretta colorata alla scala tipografica
        piu' grande, stessa trappola gia' misurata su gui/estrazione/pellicola.py."""
        cella = self.itemDelegate().sizeHint(
            QtWidgets.QStyleOptionViewItem(), QtCore.QModelIndex()).height()
        return (cella + 2 * self.frameWidth()
               + self.horizontalScrollBar().sizeHint().height())

    #override
    def sizeHint(self):
        base = super().sizeHint()
        return QtCore.QSize(base.width(), self._altezza_utile())

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
        self.frame_scelto.emit(index.row())

    def scorri_a(self, idx):
        modello = self.model()
        if modello is None or not (0 <= idx < modello.rowCount()):
            return
        indice = modello.index(idx, 0)
        self.setCurrentIndex(indice)
        self.scrollTo(indice, QtWidgets.QAbstractItemView.EnsureVisible)
