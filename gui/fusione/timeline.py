"""Una colonna per frame, ridotta in numpy come gui/loss_plot.py: a 5000
frame su 1000 px cinque frame per colonna, e vince lo stato peggiore
(senza volto > da fare > fatto), cosi' un frame da rifare non sparisce
dietro quattro fatti."""
import numpy as np
from PyQt5.QtCore import QRect, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import QWidget

from gui.numeri import intero_qt_utilizzabile

STATO_DA_FARE = 0
STATO_FATTO = 1
STATO_SENZA_VOLTO = 2

#Pubblica: usata anche dal delegato della pellicola, che dipinge lo
#stesso stato con la stessa tinta.
COLORE_PER_STATO = {STATO_DA_FARE: QColor(70, 70, 76), STATO_FATTO: QColor(90, 160, 110),
                    STATO_SENZA_VOLTO: QColor(200, 80, 80)}
_CURSORE = QColor(240, 240, 240)
_KEYFRAME = QColor(215, 170, 70)
# Con questa priorita' il massimo per colonna e' lo stato che deve vincere.
_PRIORITA = np.array([1, 0, 2], dtype=np.uint8)      # da_fare=1 > fatto=0; senza_volto=2
_DA_PRIORITA = np.array([STATO_FATTO, STATO_DA_FARE, STATO_SENZA_VOLTO], dtype=np.uint8)


def riduci(stati, larghezza):
    """Riduce `stati` a `larghezza` colonne: per colonna vince lo stato
    peggiore (senza volto > da fare > fatto)."""
    stati = np.asarray(stati, dtype=np.uint8)
    larghezza = max(1, int(larghezza))
    if stati.size == 0:
        return np.zeros(larghezza, dtype=np.uint8)
    bordi = np.linspace(0, stati.size, larghezza + 1).astype(np.int64)
    # `reduceat` invece del ciclo Python lungo quanto la larghezza in
    # pixel: gli inizi sono gia' non decrescenti, e un bin vuoto (piu'
    # colonne che frame) prende da se' il valore del frame in cui cade,
    # che e' cio' che il ciclo faceva col suo `min(fine, size)`.
    inizio = np.minimum(bordi[:-1], stati.size - 1)
    return _DA_PRIORITA[np.maximum.reduceat(_PRIORITA[stati], inizio)]


class Timeline(QWidget):
    """La barra della sessione: stato per frame, cursore e keyframe."""

    frame_scelto = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stati = np.zeros(0, dtype=np.uint8)
        self._cursore = 0
        self._keyframes = []
        self.setMinimumHeight(18)
        self.setMouseTracking(False)

    def imposta(self, totali, senza_volto):
        self._stati = np.full(max(0, int(totali)), STATO_DA_FARE, dtype=np.uint8)
        for i in senza_volto:
            self._segna(i, STATO_SENZA_VOLTO)
        #Un residuo del giro precedente (cursore o keyframe oltre il nuovo
        #totale) resterebbe fuori tela: si riparte da zero a ogni sessione.
        self._cursore = 0
        self._keyframes = []
        self.update()

    def _segna(self, idx, stato):
        if isinstance(idx, int) and 0 <= idx < self._stati.size:
            self._stati[idx] = stato

    def segna_fatto(self, idx):
        if isinstance(idx, int) and 0 <= idx < self._stati.size and self._stati[idx] != STATO_SENZA_VOLTO:
            self._stati[idx] = STATO_FATTO
        self.update()

    def segna_da_fare(self, idx):
        if isinstance(idx, int) and 0 <= idx < self._stati.size and self._stati[idx] != STATO_SENZA_VOLTO:
            self._stati[idx] = STATO_DA_FARE
        self.update()

    def imposta_cursore(self, idx):
        #Un indice non finito o fuori dal raggio dell'int di Qt non deve
        #arrivare al paintEvent: si scarta qui, il cursore resta dov'era.
        if not intero_qt_utilizzabile(idx):
            self.update()
            return
        self._cursore = max(0, min(int(idx), max(0, self._stati.size - 1)))
        self.update()

    def imposta_keyframes(self, indici):
        self._keyframes = [int(i) for i in indici
                           if intero_qt_utilizzabile(i) and 0 <= int(i) < self._stati.size]
        self.update()

    def _idx_da_x(self, x):
        if self._stati.size == 0 or self.width() <= 0:
            return 0
        return max(0, min(self._stati.size - 1, int(x * self._stati.size / self.width())))

    def mousePressEvent(self, evento):
        if evento.button() == Qt.LeftButton:
            self.frame_scelto.emit(self._idx_da_x(evento.x()))

    def mouseMoveEvent(self, evento):
        if evento.buttons() & Qt.LeftButton:
            self.frame_scelto.emit(self._idx_da_x(evento.x()))

    def paintEvent(self, _evento):
        pittore = QPainter(self)
        w, h = self.width(), self.height()
        pittore.fillRect(self.rect(), COLORE_PER_STATO[STATO_DA_FARE])
        if self._stati.size and w > 0:
            colonne = riduci(self._stati, w)
            x0 = 0
            for x in range(1, w + 1):
                if x == w or colonne[x] != colonne[x0]:
                    pittore.fillRect(QRect(x0, 0, x - x0, h), COLORE_PER_STATO[int(colonne[x0])])
                    x0 = x
            for k in self._keyframes:
                xk = int(k * w / self._stati.size)
                pittore.fillRect(QRect(xk, 0, 2, h // 2), _KEYFRAME)
            #Largo un solo pixel: a due il cursore sull'indice 0 copre anche
            #la colonna 1, e il colore sotto non si vede piu' finche' non si
            #sposta -- misurato con la colonna 0 fatta e il cursore fermo.
            xc = int(self._cursore * w / self._stati.size)
            pittore.fillRect(QRect(xc, 0, 1, h), _CURSORE)
        pittore.end()
