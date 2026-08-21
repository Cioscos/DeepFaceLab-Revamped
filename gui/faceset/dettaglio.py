"""Il client del servizio di dettaglio, e la finestra che lo consuma.

Il servizio si avvia PIGRAMENTE al primo doppio click e muore da se' dopo
qualche minuto di inattivita'. Se muore per conto suo, la richiesta
successiva lo riavvia: una richiesta senza risposta non blocca niente, la
finestra mostra il volto (il JPEG c'e' comunque) e dichiara che i dati DFL
non sono disponibili.

`trasporto` e' iniettabile perche' un test non deve avviare un processo
per verificare che il comando parta con il percorso giusto.
"""
import json
import time
from pathlib import Path

from PyQt5.QtCore import QObject, QPoint, QPointF, QRect, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPixmap, QPolygonF
from PyQt5.QtWidgets import (QCheckBox, QHBoxLayout, QLabel,
                             QVBoxLayout, QWidget)

from gui import testi
from gui import theme
from gui.faceset.griglia import MASCHERA_OFF, MODI_MASCHERA
from gui.numeri import intero_qt_utilizzabile

TIMEOUT_MS = 8000


class ClienteDettaglio(QObject):
    pronto = pyqtSignal(dict)
    fallito = pyqtSignal(object)

    def __init__(self, workdir, trasporto=None, parent=None):
        super().__init__(parent)
        self._workdir = Path(workdir)
        self._trasporto = trasporto
        self._id = 0
        self._processo = None

    def apri(self, percorso):
        self._id += 1
        comando = {"op": "open", "id": self._id, "path": str(percorso)}
        try:
            risposta = self._invia(json.dumps(comando) + "\n")
        except Exception as e:
            self.fallito.emit(e)
            return
        try:
            dati = json.loads(risposta)
        except (TypeError, ValueError) as e:
            self.fallito.emit(e)
            return
        if dati.get("op") != "opened":
            self.fallito.emit(dati.get("motivo"))
            return
        self.pronto.emit(dati)

    def _invia(self, riga):
        if self._trasporto is not None:
            return self._trasporto(riga)
        return self._invia_al_processo(riga)

    def _invia_al_processo(self, riga):
        from PyQt5.QtCore import QProcess
        if self._processo is None or self._processo.state() == QProcess.NotRunning:
            self._avvia()
        self._processo.write(riga.encode("utf-8"))
        return self._leggi_una_riga_completa()

    def _leggi_una_riga_completa(self):
        """Aspetta che il buffer del processo contenga una riga intera,
        non solo dei byte.

        `waitForReadyRead` torna appena arrivano DEI byte, non
        necessariamente una riga completa -- e `readLine()` consegna
        quello che c'e' nel buffer anche senza il `\\n` finale. Una sola
        lettura dopo l'attesa (come faceva questo metodo prima) puo'
        quindi restituire meta' risposta e lasciare il resto nel buffer:
        alla richiesta successiva `waitForReadyRead` torna subito (i dati
        ci sono gia') e `readLine()` consegna la coda della risposta
        precedente spacciandola per quella nuova -- da li' in poi ogni
        richiesta e' sfasata di un messaggio rispetto a quella vera.
        `canReadLine()` dice se il buffer contiene davvero un `\\n`;
        finche' non e' cosi' si continua ad aspettare, rispettando il
        timeout complessivo invece di uno per ogni lettura parziale."""
        scadenza = time.monotonic() + TIMEOUT_MS / 1000.0
        while not self._processo.canReadLine():
            rimanente_ms = int((scadenza - time.monotonic()) * 1000)
            if rimanente_ms <= 0 or not self._processo.waitForReadyRead(rimanente_ms):
                raise OSError("il servizio di dettaglio non ha risposto")
        return bytes(self._processo.readLine()).decode("utf-8", "replace")

    def _avvia(self):
        from PyQt5.QtCore import QProcess
        from gui.faceset.avvio import comando_servizio
        programma, argomenti = comando_servizio(self._workdir)
        self._processo = QProcess(self)
        self._processo.setProcessChannelMode(QProcess.SeparateChannels)
        self._processo.start(programma, argomenti)
        self._processo.waitForStarted(TIMEOUT_MS)

    def ferma(self):
        if self._processo is not None:
            self._processo.kill()
            self._processo = None


def _punti_utilizzabili(landmarks):
    """Solo i punti che sono davvero due numeri consegnabili a Qt.

    Un NaN o un intero fuori scala attraverserebbe le somme senza
    sollevare per morire dentro l'int() di un paintEvent -- e un
    paintEvent che solleva si porta via il processo con dentro ogni altra
    scheda aperta.

    Il predicato e' `intero_qt_utilizzabile` e **non** il solo
    `numero_finito`, che era la forma di prima: `1e300` e' finito, quindi
    passava, e moriva un momento dopo dentro il `QPoint(int(x), int(y))`
    del paintEvent con l'`OverflowError` della firma a 32 bit. Il caso non
    e' teorico -- i landmark arrivano da `DFLJPG.get_landmarks()` di un file
    dell'utente, serializzati in JSON da un altro processo, e nessuno dei
    due passaggi li guarda.
    """
    buoni = []
    for punto in landmarks or ():
        try:
            x, y = punto
        except (TypeError, ValueError):
            continue
        if intero_qt_utilizzabile(x) and intero_qt_utilizzabile(y):
            buoni.append((float(x), float(y)))
    return buoni


def _poligoni_utilizzabili(polys):
    """[(tipo, [(x, y), ...]), ...] dal campo `polys` del protocollo.

    La forma vera e' quella di `SegIEPolys.dump()`: un dizionario con una
    lista di poligoni, ognuno col suo `type` (1 include, 0 esclude) e i
    suoi `pts`. Tutto il resto -- `None`, una lista al posto del
    dizionario, un poligono che non e' un dizionario, dei punti che non
    sono punti -- si scarta invece di sollevare: questi dati arrivano da
    un altro processo, e il posto dove morirebbero e' un paintEvent.
    """
    if not isinstance(polys, dict):
        return []
    fuori = []
    for poligono in polys.get("polys") or ():
        if not isinstance(poligono, dict):
            continue
        punti = _punti_utilizzabili(poligono.get("pts"))
        if len(punti) < 2:
            continue
        tipo = poligono.get("type")
        fuori.append((tipo if isinstance(tipo, int) else 1, punti))
    return fuori


class _Tela(QLabel):
    #Un poligono INCLUDE aggiunge alla maschera, un EXCLUDE la scava: due
    #colori, perche' a occhio la differenza non si deduce dalla forma.
    COLORE_INCLUDE = QColor(80, 220, 140)
    COLORE_EXCLUDE = QColor(240, 120, 90)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._punti = []
        self._poligoni = []
        self._mostra_punti = False
        self._maschera = None
        self._modo = MASCHERA_OFF
        # In alto a sinistra, e non e' estetica: landmark e poligoni
        # arrivano in coordinate dell'IMMAGINE, e questo e' cio' che le
        # rende coordinate del widget. Con l'allineamento predefinito
        # (centrato in verticale) una finestra piu' alta del volto li
        # spostava tutti in blocco.
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)

    def imposta_dati(self, punti, maschera, poligoni=()):
        self._punti = punti
        self._maschera = maschera
        self._poligoni = list(poligoni)
        self.update()

    def imposta_punti_visibili(self, valore):
        self._mostra_punti = bool(valore)
        self.update()

    def imposta_modo_maschera(self, modo):
        self._modo = modo
        self.update()

    def _area_del_volto(self):
        """Il rettangolo occupato dall'immagine, in coordinate del widget.

        Con l'allineamento in alto a sinistra e' l'origine piu' la
        dimensione del pixmap; senza pixmap (nessun volto ancora mostrato)
        si ripiega sull'intero widget, che e' quanto si sapeva prima.
        """
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return self.rect()
        return QRect(0, 0, pixmap.width(), pixmap.height())

    #override
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        if self._maschera is not None and self._modo != MASCHERA_OFF:
            painter.setOpacity(1.0 if self._modo == "only" else 0.45)
            # Sul VOLTO, non su tutta la finestra: `self.rect()` stirava la
            # maschera su ogni pixel del widget, quindi appena il layout
            # dava alla tela piu' spazio del volto la maschera non stava
            # piu' sopra il volto -- e a schermo si legge come una
            # segmentazione sbagliata.
            painter.drawImage(self._area_del_volto(), self._maschera)
            painter.setOpacity(1.0)
        for tipo, punti in self._poligoni:
            painter.setPen(self.COLORE_INCLUDE if tipo else self.COLORE_EXCLUDE)
            painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in punti]))
        if self._mostra_punti:
            painter.setPen(QColor(255, 90, 90))
            for x, y in self._punti:
                painter.drawPoint(QPoint(int(x), int(y)))


class FinestraDettaglio(QWidget):
    def __init__(self, workdir, parent=None):
        super().__init__(parent, Qt.Window)
        self._workdir = Path(workdir)
        self._percorso = None
        self._ordine = []
        self._landmark_visibili = False
        self.tela = _Tela()
        # Gli stessi due interruttori della pagina, sullo stesso volto a
        # piena risoluzione: la finestra si apre proprio per guardare la
        # maschera da vicino, e senza comandi mostrava il JPEG e basta.
        self.spunta_landmark = QCheckBox(testi.FACESET_LANDMARKS)
        self.spunta_landmark.setToolTip(testi.FACESET_LANDMARKS_TIP)
        self.spunta_landmark.toggled.connect(self.mostra_landmark)
        self.selettore_maschera = theme.tendina()
        for chiave, etichetta in MODI_MASCHERA:
            self.selettore_maschera.addItem(etichetta, chiave)
        self.selettore_maschera.currentIndexChanged.connect(self._su_modo_maschera)
        barra = QHBoxLayout()
        barra.addWidget(self.spunta_landmark)
        barra.addWidget(self.selettore_maschera)
        barra.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addLayout(barra)
        layout.addWidget(self.tela)
        self.setWindowTitle(testi.FACESET_DETAIL_TITLE)
        self._aggiorna_selettore_maschera()

    def percorso(self):
        return self._percorso

    def imposta_ordine(self, percorsi):
        self._ordine = list(percorsi)

    def mostra(self, percorso, dati):
        self._percorso = Path(percorso)
        self.tela.setPixmap(QPixmap(str(percorso)))
        punti = _punti_utilizzabili((dati or {}).get("landmarks"))
        maschera = None
        nome = (dati or {}).get("mask")
        if nome:
            candidata = QImage(str(self._workdir / nome))
            maschera = None if candidata.isNull() else candidata
        self.tela.imposta_dati(punti, maschera,
                               _poligoni_utilizzabili((dati or {}).get("polys")))
        self._aggiorna_selettore_maschera()
        # Stessa regola della maschera: un interruttore acceso su un volto
        # che non ha landmark promette dei punti che non esistono.
        self.spunta_landmark.setEnabled(bool(punti))
        self.spunta_landmark.setToolTip(
            testi.FACESET_LANDMARKS_TIP if punti else testi.FACESET_NO_LANDMARKS)

    def mostra_landmark(self, valore):
        self._landmark_visibili = bool(valore)
        self.tela.imposta_punti_visibili(valore)

    def landmark_visibili(self):
        return self._landmark_visibili

    def imposta_modo_maschera(self, modo):
        self.tela.imposta_modo_maschera(modo)

    def _su_modo_maschera(self, indice_voce):
        modo = self.selettore_maschera.itemData(indice_voce)
        if modo is not None:
            self.imposta_modo_maschera(modo)

    def _aggiorna_selettore_maschera(self):
        """Il selettore vale per il volto mostrato ORA.

        Navigando con le frecce si passa da un volto con maschera a uno
        senza: lasciare il modo acceso mostrerebbe la maschera del volto
        precedente -- la tela la tiene finche' non gliene si da' un'altra
        -- cioe' la cosa peggiore che questa finestra possa fare.
        """
        c_e = self.tela._maschera is not None
        self.selettore_maschera.setEnabled(c_e)
        self.selettore_maschera.setToolTip(
            testi.FACESET_MASK_TIP if c_e else testi.FACESET_NO_MASKS)
        if not c_e:
            self.selettore_maschera.setCurrentIndex(0)
            self.imposta_modo_maschera(MASCHERA_OFF)

    def _sposta(self, passo):
        if self._percorso is None or self._percorso not in self._ordine:
            return
        i = self._ordine.index(self._percorso) + passo
        if 0 <= i < len(self._ordine):
            self.mostra(self._ordine[i], None)

    def vai_avanti(self):
        self._sposta(1)

    def vai_indietro(self):
        self._sposta(-1)

    #override
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_Right:
            self.vai_avanti()
        elif event.key() == Qt.Key_Left:
            self.vai_indietro()
        else:
            super().keyPressEvent(event)
