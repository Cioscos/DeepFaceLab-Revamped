"""La tela dell'editor: il volto allineato 1:1, coi landmark sopra.

**1:1 e senza scala.** Le coordinate dell'immagine SONO quelle del
widget, quindi un trascinamento non attraversa nessuna conversione e un
errore di scala non puo' esistere. Un allineato sta fra 256 e 768 pixel di
lato: chi ha la finestra piccola scorre (la finestra la mette in una
QScrollArea).

**Un paintEvent che solleva chiama qFatal** e si porta via il processo con
dentro ogni altro training aperto. Ogni coordinata che arriva da fuori
passa da gui/numeri.py::intero_qt_utilizzabile, che comincia gia' da
numero_finito: 1e300 e' finito e muore dentro l'int() che lo consegna a
Qt, quindi e' quella chiamata sola a dover esserci.

**L'ordine dei livelli**, dal basso: pixmap, maschera, poligoni, spezzate,
punti, selezione, laccio. La maschera SOTTO i punti, o coprirebbe proprio
cio' che si sta posizionando. E' legato da un test sui pixel, non da
questo commento.
"""
from PyQt5 import QtCore, QtGui, QtWidgets

from gui.dettaglio import gruppi as gruppi_mod
from gui.dettaglio import selezione as sel_mod
from gui.faceset.griglia import MASCHERA_OFF
from gui.numeri import intero_qt_utilizzabile


def _punto_disegnabile(punto):
    """(x, y) come interi consegnabili a Qt, o None. Non solleva: di qui
    si passa dentro un paintEvent.

    `intero_qt_utilizzabile` gia' comincia da `numero_finito`: chiamarlo
    da solo basta, la difesa e' una sola."""
    try:
        x, y = punto
    except (TypeError, ValueError):
        return None
    for v in (x, y):
        if not intero_qt_utilizzabile(v):
            return None
    return int(x), int(y)


class Tela(QtWidgets.QWidget):
    COLORE_SELEZIONE = QtGui.QColor(255, 255, 255)
    COLORE_INCLUDE = QtGui.QColor(80, 220, 140)
    COLORE_EXCLUDE = QtGui.QColor(240, 120, 90)
    COLORE_SMORZATO = QtGui.QColor(110, 110, 110)
    RAGGIO_PUNTO = 3
    RAGGIO_PRESA = 8
    OPACITA_MASCHERA = 0.45

    selezione_cambiata = QtCore.pyqtSignal(object)
    punti_mossi = QtCore.pyqtSignal(object)
    trascinamento_finito = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._punti = []
        self._maschera = None
        self._poligoni = []
        self._face_type = None
        self._influenti = frozenset()
        self._selezione = frozenset()
        self._aree = ()  # vuota = tutto smorzato, vedi imposta_aree_attive
        self._colori = {}
        self._modo = MASCHERA_OFF
        self._visibili = False
        self._laccio = None
        self._spezzate = 0
        self._modificabile = True
        self._presa = None          # (x, y) dell'ultimo movimento
        self._origine_laccio = None
        self._ha_mosso = False
        self._collassa_su = None    # click su un punto gia' selezionato,
                                    # in attesa di sapere se e' un click o un trascinamento

    def imposta_volto(self, pixmap, punti, maschera, face_type, poligoni=()):
        self._pixmap = pixmap
        self._punti = list(punti or [])
        self._maschera = maschera
        self._poligoni = list(poligoni)
        self._face_type = face_type
        self._influenti = gruppi_mod.indici_influenti(face_type)
        self._selezione = frozenset()
        if pixmap is not None and not pixmap.isNull():
            self.setMinimumSize(pixmap.width(), pixmap.height())
        self.update()

    def punti(self):
        return [list(p) for p in self._punti]

    def imposta_punti(self, punti):
        """Solo i punti: il pixmap e la maschera restano quelli. Si usa
        durante il trascinamento, prima che l'anteprima riwarpata arrivi."""
        self._punti = list(punti or [])
        self.update()

    def imposta_selezione(self, indici):
        self._selezione = frozenset(indici)
        self.update()

    def selezione(self):
        return self._selezione

    def imposta_aree_attive(self, nomi):
        """Le aree attive: `self._aree` parte vuota (nessuna area accesa),
        quindi CHI MONTA LA FINESTRA deve chiamare questo prima del primo
        paintEvent, o ogni punto si disegna smorzato -- un volto tutto
        grigio, non un errore, solo un interruttore mai girato."""
        self._aree = tuple(nomi)
        # Un'area spenta perde i suoi punti dalla selezione VIVA: lasciarli
        # dentro li farebbe muovere a un trascinamento successivo, dopo che
        # l'utente ha detto di non volerli.
        ammessi = set()
        for nome in self._aree:
            ammessi |= gruppi_mod.indici_gruppo(nome)
        self._selezione &= frozenset(ammessi)
        self.update()

    def aree_attive(self):
        return self._aree

    def imposta_colori(self, mappa):
        self._colori = dict(mappa or {})
        self.update()

    def colore_di(self, nome):
        return self._colori.get(nome,
                                QtGui.QColor(*gruppi_mod.COLORI_PREDEFINITI[nome]))

    def imposta_modo_maschera(self, modo):
        self._modo = modo
        self.update()

    def imposta_landmark_visibili(self, valore):
        self._visibili = bool(valore)
        self.update()

    def imposta_modificabile(self, valore):
        """Spegne la MODIFICA, non la selezione: guardare non e'
        modificare, e la selezione serve anche solo a capire quale punto
        e' quale.

        Vive in un posto solo. La lezione del ciclo precedente e' che una
        regola applicata in due gestori finisce applicata in uno.
        """
        self._modificabile = bool(valore)
        if not self._modificabile:
            self._presa = None
            self._origine_laccio = None
            self._collassa_su = None
            self._laccio = None
            self.update()

    def modificabile(self):
        return self._modificabile

    def _ammessi(self):
        return sel_mod.indici_ammessi(self._aree)

    def _imposta_selezione_e_annuncia(self, indici):
        indici = frozenset(indici)
        if indici == self._selezione:
            return
        self._selezione = indici
        self.update()
        self.selezione_cambiata.emit(indici)

    #override
    def mousePressEvent(self, evento):
        if evento.button() != QtCore.Qt.LeftButton:
            super().mousePressEvent(evento)
            return
        x, y = evento.pos().x(), evento.pos().y()
        self._ha_mosso = False
        self._collassa_su = None
        indice = sel_mod.punto_vicino(self._punti, x, y, self.RAGGIO_PRESA,
                                      self._ammessi())
        if indice is None:
            self._origine_laccio = (x, y)
            self._presa = None
            return
        if evento.modifiers() & QtCore.Qt.ControlModifier:
            self._imposta_selezione_e_annuncia(
                sel_mod.commuta(self._selezione, indice))
        elif indice not in self._selezione:
            self._imposta_selezione_e_annuncia({indice})
        else:
            # Gia' selezionato: un trascinamento deve muovere l'intera
            # selezione, quindi non si restringe subito. Se al rilascio
            # risulta che non ci si e' mossi, e' un click e collassa qui.
            self._collassa_su = indice
        self._presa = (x, y)

    #override
    def mouseMoveEvent(self, evento):
        # Un rilascio sinistro perso -- un popup, un modale che ruba il
        # grab, la finestra che si disattiva -- non deve lasciare un
        # semplice passaggio del mouse a trascinare i landmark: qui si
        # guarda quali bottoni sono PREMUTI ORA, non quale ha generato
        # l'evento.
        if not (evento.buttons() & QtCore.Qt.LeftButton):
            return
        x, y = evento.pos().x(), evento.pos().y()
        if self._origine_laccio is not None:
            self._ha_mosso = True
            x0, y0 = self._origine_laccio
            self._laccio = QtCore.QRect(QtCore.QPoint(x0, y0),
                                        QtCore.QPoint(x, y)).normalized()
            self.update()
            return
        if self._presa is None or not self._modificabile:
            return
        dx, dy = x - self._presa[0], y - self._presa[1]
        self._presa = (x, y)
        if dx == 0 and dy == 0:
            # Un evento di movimento e' arrivato, ma niente si e' spostato:
            # non conta come trascinamento, o un gesto fermo chiederebbe lo
            # stesso il riallineamento sincrono al rilascio.
            return
        self._ha_mosso = True
        self._punti = sel_mod.trasla(self._punti, self._selezione, dx, dy)
        self.update()
        # Solo la tela: nessuna richiesta al servizio durante il gesto. Il
        # client e' sincrono, e un viaggio per mouseMoveEvent terrebbe la
        # GUI ferma fra un movimento e l'altro.
        self.punti_mossi.emit(self.punti())

    #override
    def mouseReleaseEvent(self, evento):
        if evento.button() != QtCore.Qt.LeftButton:
            super().mouseReleaseEvent(evento)
            return
        x, y = evento.pos().x(), evento.pos().y()
        if self._origine_laccio is not None:
            x0, y0 = self._origine_laccio
            self._origine_laccio = None
            self._laccio = None
            if self._ha_mosso:
                self._imposta_selezione_e_annuncia(
                    sel_mod.nel_laccio(self._punti, x0, y0, x, y, self._ammessi()))
            else:
                # Un click sul fondo senza trascinare svuota: e' il gesto
                # con cui si esce da una selezione senza doverne fare
                # un'altra.
                self._imposta_selezione_e_annuncia(frozenset())
            self.update()
            return
        mosso = self._ha_mosso and self._presa is not None and self._modificabile
        self._presa = None
        if not mosso and self._collassa_su is not None:
            # Click senza trascinare su un punto gia' selezionato: lo
            # seleziona da solo, come qualunque altro click. Vale anche se
            # il mancato movimento e' dovuto alla sola lettura: un
            # trascinamento che non sposta niente e' un click.
            self._imposta_selezione_e_annuncia({self._collassa_su})
        self._collassa_su = None
        if mosso:
            # QUI si chiede il riallineamento, una volta sola per gesto.
            self.trascinamento_finito.emit(self.punti())

    def spezzate_disegnate(self):
        """Quante spezzate ha disegnato l'ultimo paintEvent. Esiste per
        legare la regola dei 68 punti a un test: contare i pixel non la
        distinguerebbe da un colore sbagliato."""
        return self._spezzate

    def _disegna_maschera(self, pittore):
        if self._maschera is None or self._modo == MASCHERA_OFF:
            return
        pittore.setOpacity(1.0 if self._modo == "only" else self.OPACITA_MASCHERA)
        area = QtCore.QRect(0, 0, self._pixmap.width(), self._pixmap.height())
        pittore.drawImage(area, self._maschera)
        pittore.setOpacity(1.0)

    def _disegna_poligoni(self, pittore):
        for tipo, punti in self._poligoni:
            coppie = [_punto_disegnabile(p) for p in punti]
            coppie = [c for c in coppie if c is not None]
            if len(coppie) < 2:
                continue
            pittore.setPen(self.COLORE_INCLUDE if tipo else self.COLORE_EXCLUDE)
            pittore.drawPolygon(QtGui.QPolygon(
                [QtCore.QPoint(x, y) for x, y in coppie]))

    def _disegna_spezzate(self, pittore):
        self._spezzate = 0
        # Solo a 68: FAN puo' produrne 98, e collegare indici che non
        # significano quello disegnerebbe una faccia inventata. Meglio i
        # soli pallini, che restano veri qualunque sia il modello.
        if len(self._punti) != 68:
            return
        for nome, _indici, chiusa in gruppi_mod.GRUPPI_68:
            colore = QtGui.QColor(self.colore_di(nome))
            colore.setAlpha(140)
            pittore.setPen(colore)
            coppie = [_punto_disegnabile(self._punti[i])
                      for i in gruppi_mod.spezzata_gruppo(nome)]
            segmenti = list(zip(coppie, coppie[1:]))
            # Occhi e bocca sono spezzate CHIUSE: un segmento in piu' che
            # torna dall'ultimo punto al primo, come le due
            # cv2.polylines(closed=True) dell'originale.
            if chiusa and len(coppie) > 2:
                segmenti.append((coppie[-1], coppie[0]))
            disegnata = False
            for uno, altro in segmenti:
                # Un segmento si salta se un estremo non e' utilizzabile,
                # invece di ricucire il gruppo saltando il punto: un
                # collegamento falso che scavalca sarebbe peggio del buco.
                if uno is None or altro is None:
                    continue
                pittore.drawLine(QtCore.QPoint(*uno), QtCore.QPoint(*altro))
                disegnata = True
            if disegnata:
                self._spezzate += 1

    def _disegna_punti(self, pittore):
        r = self.RAGGIO_PUNTO
        for i, punto in enumerate(self._punti):
            coppia = _punto_disegnabile(punto)
            if coppia is None:
                continue
            x, y = coppia
            nome = gruppi_mod.gruppo_di(i)
            # Smorzato se il gruppo non esiste (indice oltre i 68) o se
            # la sua area e' spenta dall'interruttore -- §8.2: un punto
            # che non partecipa alla selezione non deve sembrare vivo.
            attivo = nome is not None and nome in self._aree
            colore = self.colore_di(nome) if attivo else self.COLORE_SMORZATO
            if i in self._selezione:
                colore = self.COLORE_SELEZIONE
            pittore.setPen(QtGui.QPen(colore))
            # Vuoto se il punto NON muove il ritaglio: serve a non far
            # sembrare rotto il riwarp quando si trascina la mascella di un
            # whole_face. L'insieme dipende dal face type: con `head`
            # estimate_averaged_yaw legge anche 0,1,2 e 14,15,16.
            pittore.setBrush(QtGui.QBrush(colore) if i in self._influenti
                             else QtCore.Qt.NoBrush)
            pittore.drawEllipse(QtCore.QPoint(x, y), r, r)

    def _disegna_laccio(self, pittore):
        if self._laccio is None:
            return
        pittore.setPen(QtGui.QPen(self.COLORE_SELEZIONE, 1, QtCore.Qt.DashLine))
        pittore.setBrush(QtCore.Qt.NoBrush)
        pittore.drawRect(self._laccio)

    #override
    def paintEvent(self, _evento):
        pittore = QtGui.QPainter(self)
        if self._pixmap is None or self._pixmap.isNull():
            return
        pittore.drawPixmap(0, 0, self._pixmap)
        self._disegna_maschera(pittore)
        self._disegna_poligoni(pittore)
        if self._visibili:
            self._disegna_spezzate(pittore)
            self._disegna_punti(pittore)
        self._disegna_laccio(pittore)
