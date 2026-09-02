"""Il frame fuso, l'interruttore delle tre viste, la lente a 1:1.

Ogni numero che arriva dal figlio (il rect) passa da gui/numeri.py con
ENTRAMBI i predicati prima di toccare un QRect: un'eccezione in paintEvent
e' qFatal. La scacchiera sotto l'alpha e' la stessa idea del tasto «v»
della finestra cv2 (MergerScreen.show)."""
from pathlib import Path

from PyQt5.QtCore import QRect
from PyQt5.QtGui import QColor, QImage, QPainter, QPen
from PyQt5.QtWidgets import QWidget

from gui import numeri

_SCACCO = 8
_GRIGIO_A = QColor(96, 96, 96)
_GRIGIO_B = QColor(160, 160, 160)
# Il riquadro del volto sulla tela: sottile, per non coprire cio' che
# circonda -- e' un riferimento, non una decorazione.
_RIQUADRO = QColor(120, 200, 255)
_PENNA_RIQUADRO = QPen(_RIQUADRO, 1)


def carica_tre(originale, fuso, maschera):
    """Legge i tre piani da disco. `None` per un file assente o illeggibile."""
    def _uno(p):
        if p is None or not Path(p).exists():
            return None
        img = QImage(str(p))
        return None if img.isNull() else img
    return _uno(originale), _uno(fuso), _uno(maschera)


def componi_maschera(fuso, maschera):
    """Il fuso dove la maschera e' bianca, la scacchiera dove e' nera."""
    w, h = fuso.width(), fuso.height()
    out = QImage(w, h, QImage.Format_RGB32)
    pittore = QPainter(out)
    for y in range(0, h, _SCACCO):
        for x in range(0, w, _SCACCO):
            pittore.fillRect(x, y, _SCACCO, _SCACCO, _GRIGIO_A if ((x + y) // _SCACCO) % 2 else _GRIGIO_B)
    con_alpha = fuso.convertToFormat(QImage.Format_ARGB32)
    m = maschera.convertToFormat(QImage.Format_Grayscale8).scaled(w, h)
    con_alpha.setAlphaChannel(m)
    pittore.drawImage(0, 0, con_alpha)
    pittore.end()
    return out


def rect_utilizzabile(rect):
    """Il rect del figlio, passato dai due predicati e ridotto a QRect, o None."""
    if not isinstance(rect, (list, tuple)) or len(rect) != 4:
        return None
    if not all(numeri.numero_finito(v) and numeri.intero_qt_utilizzabile(v) for v in rect):
        return None
    x0, y0, x1, y1 = [int(v) for v in rect]
    if x1 <= x0 or y1 <= y0:
        return None
    return QRect(x0, y0, x1 - x0, y1 - y0)


class Tela(QWidget):
    """Il frame corrente nelle tre viste, con l'interruttore fra loro."""

    VISTE = ("original", "merged", "mask")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._immagini = {"original": None, "merged": None, "mask": None}
        self._da_comporre = None     # (fuso, maschera) finche' la maschera non serve
        self._vista = "merged"
        self._scala = 1.0
        self._rect = None
        self._in_attesa = False
        self.setMinimumSize(160, 120)

    def vista(self):
        return self._vista

    def imposta_vista(self, vista):
        if vista in self.VISTE:
            self._vista = vista
            self.update()

    def zoom(self, delta):
        self._scala = max(0.1, min(4.0, self._scala + delta))
        self.update()

    def mostra(self, originale, fuso, maschera, rect, in_attesa):
        self._immagini["original"] = originale
        self._immagini["merged"] = fuso
        # La scacchiera sotto l'alpha e' un doppio ciclo di fillRect su
        # tutta l'immagine: si compone alla PRIMA richiesta della vista
        # `mask`, non a ogni frame mostrato -- chi non la guarda mai non la
        # paga mai.
        self._immagini["mask"] = None
        self._da_comporre = None if (fuso is None or maschera is None) else (fuso, maschera)
        self._rect = rect_utilizzabile(rect)
        self._in_attesa = bool(in_attesa)
        self.update()

    def _immagine_della_vista(self):
        if self._vista == "mask" and self._immagini["mask"] is None and self._da_comporre is not None:
            self._immagini["mask"] = componi_maschera(*self._da_comporre)
        return self._immagini.get(self._vista)

    def paintEvent(self, _evento):
        pittore = QPainter(self)
        pittore.fillRect(self.rect(), QColor(35, 35, 38))
        img = self._immagine_della_vista()
        if img is None:
            img = self._immagini.get("original")
        if img is not None and img.width() > 0 and img.height() > 0:
            fattore = min(self.width() / img.width(), self.height() / img.height()) * self._scala
            w, h = max(1, int(img.width() * fattore)), max(1, int(img.height() * fattore))
            x, y = (self.width() - w) // 2, (self.height() - h) // 2
            pittore.drawImage(QRect(x, y, w, h), img)
            if self._rect is not None:
                # Il riquadro del volto, nelle coordinate dell'immagine
                # portate a quelle dello schermo: e' il rettangolo che la
                # lente ingrandisce, e senza disegnarlo non si vede dove
                # stia guardando.
                pittore.setPen(_PENNA_RIQUADRO)
                pittore.drawRect(QRect(x + int(self._rect.x() * fattore),
                                       y + int(self._rect.y() * fattore),
                                       max(1, int(self._rect.width() * fattore)),
                                       max(1, int(self._rect.height() * fattore))))
        if self._in_attesa:
            pittore.fillRect(QRect(8, 8, 24, 24), QColor(230, 200, 60))
        pittore.end()


class Lente(QWidget):
    """Il ritaglio 1:1 intorno al centro del rect, senza scala."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._immagine = None
        self._rect = None
        self.setMinimumSize(96, 96)

    def mostra(self, immagine, rect):
        self._immagine = immagine
        self._rect = rect_utilizzabile(rect)
        self.update()

    def paintEvent(self, _evento):
        pittore = QPainter(self)
        pittore.fillRect(self.rect(), QColor(35, 35, 38))
        if self._immagine is not None and self._rect is not None:
            cx, cy = self._rect.center().x(), self._rect.center().y()
            sorgente = QRect(cx - self.width() // 2, cy - self.height() // 2, self.width(), self.height())
            pittore.drawImage(0, 0, self._immagine, sorgente.x(), sorgente.y(), sorgente.width(), sorgente.height())
        pittore.end()
