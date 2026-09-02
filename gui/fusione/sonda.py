"""La striscia dei frame sondati: `n` frame equispaziati che il pool fonde
per primi, cosi' una regolazione si giudica su tutto il video senza
fonderlo tutto. Un riquadro per indice, «in attesa» finche' il suo
frame_pronto non arriva, poi la miniatura del PNG fuso (stesso
Decodificatore della pellicola); click = frame_scelto. Una nuova sonda
sostituisce la striscia intera."""
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import QWidget

from gui import numeri
from gui.faceset.decodifica import Decodificatore
from gui.fusione.pellicola import LATO_MINIATURA

MARGINE = 4
COLORE_ATTESA = QColor(70, 70, 70)
COLORE_BORDO = QColor(120, 120, 120)


class StrisciaSonda(QWidget):
    frame_scelto = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.indici = []
        self._pronti = set()
        self._percorso_di = lambda _i: None
        self.decodificatore = Decodificatore(self)
        self.decodificatore.pronta.connect(self._su_immagine_pronta)
        self.setFixedHeight(LATO_MINIATURA + 2 * MARGINE)
        self.setVisible(False)

    def imposta(self, indici, percorso_di):
        self.indici = sorted({int(i) for i in indici
                              if numeri.intero_qt_utilizzabile(i) and int(i) >= 0})
        self._pronti = set()
        self._percorso_di = percorso_di
        self.setVisible(bool(self.indici))
        self.update()

    def svuota(self):
        self.imposta([], lambda _i: None)

    def quanti_pronti(self):
        return len(self._pronti)

    def segna_pronto(self, idx):
        if not numeri.intero_qt_utilizzabile(idx) or int(idx) not in self.indici:
            return
        self._pronti.add(int(idx))
        self.update()

    def centro_del_riquadro(self, posizione):
        return MARGINE + posizione * (LATO_MINIATURA + MARGINE) + LATO_MINIATURA // 2

    def _posizione_da_x(self, x):
        passo = LATO_MINIATURA + MARGINE
        pos = (x - MARGINE) // passo
        return pos if 0 <= pos < len(self.indici) else None

    def _immagine(self, idx):
        try:
            percorso = self._percorso_di(idx)
        except Exception:
            return None
        if percorso is None:
            return None
        immagine = self.decodificatore.in_cache(percorso, LATO_MINIATURA)
        if immagine is None:
            self.decodificatore.richiedi(percorso, LATO_MINIATURA)
        return immagine

    def _su_immagine_pronta(self, *_args):
        self.update()

    def mousePressEvent(self, evento):
        pos = self._posizione_da_x(evento.pos().x())
        if pos is not None:
            self.frame_scelto.emit(self.indici[pos])

    def paintEvent(self, _evento):
        pittore = QPainter(self)
        try:
            for posizione, idx in enumerate(self.indici):
                x = MARGINE + posizione * (LATO_MINIATURA + MARGINE)
                immagine = self._immagine(idx) if idx in self._pronti else None
                if immagine is None or immagine.isNull():
                    pittore.fillRect(x, MARGINE, LATO_MINIATURA, LATO_MINIATURA, COLORE_ATTESA)
                else:
                    pittore.drawImage(x, MARGINE, immagine)
                pittore.setPen(COLORE_BORDO)
                pittore.drawRect(x, MARGINE, LATO_MINIATURA - 1, LATO_MINIATURA - 1)
        finally:
            pittore.end()
