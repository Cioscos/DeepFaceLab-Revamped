"""La tela interattiva dell'estrazione manuale.

Possiede il trascinamento: il vettore non attraversa nessun processo,
quindi non c'e' latenza da nascondere. Il calcolo dei landmark e la
scrittura stanno dall'altra parte, nel servizio (gui/estrazione/servizio.py).

**Due spazi di coordinate, e confonderli e' un difetto silenzioso.** Il
frame arriva a risoluzione nativa (1920x1080 e' il caso normale) e la tela
e' larga quanto la scheda: il pixmap si scala per starci tutto dentro
(proporzioni conservate, centrato), quindi le coordinate del widget e
quelle del frame **non coincidono piu'**. Rect e landmark arrivano in
coordinate del FRAME e si portano a quelle del widget solo per disegnarli;
il vettore tracciato col mouse nasce in coordinate del WIDGET e si riporta
a quelle del frame prima di emetterlo -- il servizio salva il volto sul
frame a risoluzione piena, e un fattore dimenticato mette il rettangolo da
un'altra parte senza che niente lo segnali. E' la stessa scala che faceva
la finestra cv2 sostituita da questa tela (`view_scale` in
mainscripts/Extractor.py), che scalava anche in su.

Ogni numero che arriva da fuori passa da gui/numeri.py con ENTRAMBI i
predicati: numero_finito da solo non basta, 1e300 e' finito e uccide
comunque l'int() di un paintEvent. Un paintEvent che solleva chiama qFatal
e porta via il processo con dentro ogni training aperto -- e un'eccezione
in uno slot costa esattamente lo stesso. **Il controllo si ripete DOPO la
scala**, non solo sul dato in ingresso: un frame alto un pixel dentro una
tela alta 900 da' un fattore di 900, e 1e7 (che entra) diventa 9e9 (che
non entra piu' in un int a 32 bit).
"""
from PyQt5 import QtCore, QtGui, QtWidgets

from gui import numeri

RAGGIO_PUNTO = 2


def _coppie_utilizzabili(punti):
    """Le coppie (x, y) su cui si puo' fare aritmetica, in coordinate del
    frame. Non si costruisce ancora nessun QPoint: la scala viene dopo, e
    con lei il solo posto in cui il limite dell'int a 32 bit conta."""
    if not punti:
        return []
    fuori = []
    for punto in punti:
        try:
            x, y = punto
        except (TypeError, ValueError):
            continue
        if not (numeri.numero_finito(x) and numeri.numero_finito(y)):
            continue
        if not (numeri.intero_qt_utilizzabile(x) and numeri.intero_qt_utilizzabile(y)):
            continue
        fuori.append((float(x), float(y)))
    return fuori


def _rect_utilizzabile(rect):
    """(l, t, r, b) in coordinate del frame, o None."""
    if not rect:
        return None
    try:
        l, t, r, b = rect
    except (TypeError, ValueError):
        return None
    for v in (l, t, r, b):
        if not (numeri.numero_finito(v) and numeri.intero_qt_utilizzabile(v)):
            return None
    return float(l), float(t), float(r), float(b)


def _punto_qt(x, y):
    """QPoint, non QPointF: drawEllipse su un QPointF resta in virgola
    mobile e un 1e300 ci passa dentro senza sollevare -- la guardia
    servirebbe a niente. QPoint(int(x), int(y)) e' il punto in cui PyQt5
    solleva OverflowError sopra i 2**31, cioe' il posto dove la seconda
    meta' del controllo (intero_qt_utilizzabile) diventa davvero
    necessaria e non solo dichiarata (vedi gui/faceset/dettaglio.py).

    Torna None invece di sollevare: qui si arriva da un paintEvent."""
    if not (numeri.numero_finito(x) and numeri.numero_finito(y)):
        return None
    if not (numeri.intero_qt_utilizzabile(x) and numeri.intero_qt_utilizzabile(y)):
        return None
    return QtCore.QPoint(int(x), int(y))


class Tela(QtWidgets.QWidget):
    vettore_tracciato = QtCore.pyqtSignal(object, object)
    confermato = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMouseTracking(True)
        self._pixmap = None
        self._rect = None
        self._punti = []
        self._centro = None
        self._punta = None

    def mostra(self, pixmap, rect, landmarks):
        """`rect` e `landmarks` sono in coordinate del FRAME, come li
        produce il servizio: la scala la applica il disegno."""
        self._pixmap = pixmap
        self._rect = _rect_utilizzabile(rect)
        self._punti = _coppie_utilizzabili(landmarks)
        self.update()

    # -- la scala ----------------------------------------------------------

    def trasformazione(self):
        """(fattore, dx, dy) per andare dal frame al widget, o None.

        Il fattore non si tiene in un attributo e non si imposta da fuori:
        e' interamente determinato dalla dimensione del pixmap e da quella
        del widget, e un attributo sarebbe solo una copia da tenere in
        sincronia con un resizeEvent.

        None quando la trasformazione non esiste -- nessun pixmap, pixmap
        nullo, un lato a zero, il widget non ancora dimensionato. Chi
        chiama e' un paintEvent o uno slot: qui non si divide mai per zero.
        """
        if self._pixmap is None or self._pixmap.isNull():
            return None
        larghezza_frame = self._pixmap.width()
        altezza_frame = self._pixmap.height()
        if larghezza_frame <= 0 or altezza_frame <= 0:
            return None
        if self.width() <= 0 or self.height() <= 0:
            return None
        fattore = min(self.width() / float(larghezza_frame),
                      self.height() / float(altezza_frame))
        if not numeri.numero_finito(fattore) or fattore <= 0.0:
            return None
        dx = (self.width() - larghezza_frame * fattore) / 2.0
        dy = (self.height() - altezza_frame * fattore) / 2.0
        return fattore, dx, dy

    def _al_widget(self, x, y):
        t = self.trasformazione()
        if t is None:
            return float(x), float(y)
        fattore, dx, dy = t
        return x * fattore + dx, y * fattore + dy

    def _al_frame(self, x, y):
        """L'inversa. Senza di lei il rettangolo finirebbe in un punto
        diverso da quello indicato -- il difetto peggiorerebbe invece di
        risolversi, perche' oggi (frame piu' piccolo della tela) i due
        spazi coincidono per caso."""
        t = self.trasformazione()
        if t is None:
            return float(x), float(y)
        fattore, dx, dy = t
        return (x - dx) / fattore, (y - dy) / fattore

    # -- il mouse ----------------------------------------------------------

    def mousePressEvent(self, evento):
        if evento.button() == QtCore.Qt.LeftButton:
            self._centro = evento.pos()
            self._punta = evento.pos()
            self.update()

    def mouseMoveEvent(self, evento):
        if self._centro is not None:
            self._punta = evento.pos()
            self.update()

    def mouseReleaseEvent(self, evento):
        if evento.button() == QtCore.Qt.LeftButton and self._centro is not None:
            centro = self._al_frame(self._centro.x(), self._centro.y())
            punta = self._al_frame(evento.pos().x(), evento.pos().y())
            self._centro = None
            self._punta = None
            self.update()
            self.vettore_tracciato.emit(centro, punta)

    def keyPressEvent(self, evento):
        if evento.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            self.confermato.emit()
        else:
            super().keyPressEvent(evento)

    # -- il disegno --------------------------------------------------------

    def paintEvent(self, _evento):
        pittore = QtGui.QPainter(self)
        pittore.fillRect(self.rect(), self.palette().window())
        trasformazione = self.trasformazione()
        if trasformazione is not None:
            fattore, dx, dy = trasformazione
            # Senza questo il ridimensionamento di un 1080p e' a vicino piu'
            # prossimo: sui capelli e sui bordi del volto e' proprio cio'
            # che serve guardare per decidere dove tracciare il vettore.
            pittore.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
            destinazione = QtCore.QRect(
                int(dx), int(dy),
                max(1, int(self._pixmap.width() * fattore)),
                max(1, int(self._pixmap.height() * fattore)))
            pittore.drawPixmap(destinazione, self._pixmap)
        if self._rect is not None:
            l, t, r, b = self._rect
            alto_sinistra = _punto_qt(*self._al_widget(l, t))
            basso_destra = _punto_qt(*self._al_widget(r, b))
            if alto_sinistra is not None and basso_destra is not None:
                pittore.setPen(QtGui.QPen(QtGui.QColor(90, 200, 250), 2))
                pittore.drawRect(QtCore.QRect(alto_sinistra, basso_destra))
        if self._punti:
            pittore.setPen(QtGui.QPen(QtGui.QColor(250, 220, 90), 1))
            for x, y in self._punti:
                punto = _punto_qt(*self._al_widget(x, y))
                if punto is not None:
                    pittore.drawEllipse(punto, RAGGIO_PUNTO, RAGGIO_PUNTO)
        if self._centro is not None and self._punta is not None:
            # Gia' in coordinate del widget: nascono dal mouse e muoiono al
            # rilascio, non passano mai dal frame.
            pittore.setPen(QtGui.QPen(QtGui.QColor(250, 120, 90), 1))
            pittore.drawLine(self._centro, self._punta)
        pittore.end()
