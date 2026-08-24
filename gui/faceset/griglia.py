"""La griglia dei volti: vista virtualizzata, delegato che dipinge da se'.

Il delegato NON chiama super().paint(). Nel ciclo v4 l'approccio che il
piano stesso suggeriva -- super().paint() con displayText svuotato --
segfaultava per davvero sotto il tema scuro reale, invisibile ai test
sintetici perche' nessuno di loro applicava il tema. Qui si dipinge tutto:
sfondo, immagine, cornice di selezione.
"""
from collections import OrderedDict

from PyQt5.QtCore import QRect, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPen
from PyQt5.QtWidgets import (QAbstractItemView, QListView, QStyle,
                             QStyledItemDelegate)

from gui import testi
from gui.faceset.decodifica import LATI, Decodificatore, peso_immagine
from gui.faceset.modello import RUOLO_PERCORSO, RUOLO_VOCE

MASCHERA_OFF = "off"
MASCHERA_OVERLAY = "overlay"
MASCHERA_ONLY = "only"

# I tre stati e i loro testi, nell'ordine in cui stanno in una tendina. Una
# sola volta e qui, accanto alle chiavi: la griglia e la finestra di
# dettaglio offrono lo STESSO controllo, e due elenchi paralleli
# divergerebbero -- un utente che legge «Mask only» in due posti e ne vede
# uno che fa altro non ha modo di capire quale dei due sta mentendo. Le
# chiavi sono dato di dispatch, i testi nascono in gui/testi.py.
MODI_MASCHERA = (
    (MASCHERA_OFF, testi.FACESET_MASK_OFF),
    (MASCHERA_OVERLAY, testi.FACESET_MASK_OVERLAY),
    (MASCHERA_ONLY, testi.FACESET_MASK_ONLY),
)

MARGINE = 6

# Lo stesso verde del poligono INCLUDE della finestra di dettaglio
# (gui/dettaglio/tela.py): due superfici che dicono «questo appartiene
# all'insieme» lo dicono con lo stesso colore.
COLORE_FRATELLO = QColor(80, 220, 140)


def _lato_ammesso(valore):
    return min(LATI, key=lambda l: abs(l - valore))


class DelegatoVolti(QStyledItemDelegate):
    def __init__(self, griglia):
        super().__init__(griglia)
        self._griglia = griglia

    #override
    def sizeHint(self, option, index):
        lato = self._griglia.lato()
        return QSize(lato + MARGINE * 2, lato + MARGINE * 2)

    #override
    def paint(self, painter, option, index):
        lato = self._griglia.lato()
        rect = option.rect
        painter.save()
        painter.fillRect(rect, option.palette.base())

        percorso = index.data(RUOLO_PERCORSO)
        immagine = self._griglia.immagine_per(percorso, lato)
        cella = QRect(rect.left() + MARGINE, rect.top() + MARGINE, lato, lato)
        modo = self._griglia.modo_maschera()

        if modo != MASCHERA_ONLY and immagine is not None:
            painter.drawImage(cella, immagine)
        elif modo != MASCHERA_ONLY:
            painter.fillRect(cella, option.palette.window())

        if modo != MASCHERA_OFF:
            maschera = self._griglia.maschera_per(index.data(RUOLO_VOCE))
            if maschera is not None:
                if modo == MASCHERA_ONLY:
                    painter.drawImage(cella, maschera)
                else:
                    # La MASCHERA sopra il volto, non una tinta sulla
                    # cella. Prima qui c'era un fillRect dell'intera cella:
                    # ogni volto con una maschera veniva verde uguale, e
                    # accendere l'overlay diceva soltanto «una maschera
                    # c'e'» -- che chi ha appena chiesto le maschere sa
                    # gia'. La forma e' l'unica cosa per cui si guarda una
                    # segmentazione. Stessa resa della tela della finestra
                    # di dettaglio (gui/dettaglio/tela.py): due
                    # superfici che mostrano la stessa cosa allo stesso
                    # modo. Il prezzo e' il colore -- la maschera e' in
                    # scala di grigi e schiarisce invece di tingere.
                    painter.setOpacity(0.45)
                    painter.drawImage(cella, maschera)
                    painter.setOpacity(1.0)
            elif modo == MASCHERA_ONLY:
                # Tratteggio: «nessuna maschera» deve essere distinguibile
                # da «maschera tutta nera», che e' esattamente la
                # differenza che si sta cercando accendendo questa vista.
                #
                # Il ritaglio non e' un dettaglio: le diagonali scendono
                # verso SINISTRA, e senza clip la coda di ognuna finiva
                # sulla cella accanto -- che il delegato ha gia' dipinto,
                # perche' si dipinge da sinistra a destra. Nello scatto si
                # vedevano le righe del tratteggio sopra la maschera del
                # vicino, cioe' una maschera buona che sembrava rotta.
                # L'intervallo parte piu' a sinistra e finisce piu' a
                # destra del necessario apposta: il clip taglia il resto, e
                # cosi' la cella e' tratteggiata fino ai suoi bordi.
                painter.save()
                painter.setClipRect(cella)
                painter.setPen(QColor(150, 150, 150))
                for x in range(cella.left(), cella.right() + cella.height(), 8):
                    painter.drawLine(x, cella.top(), x - cella.height(), cella.bottom())
                painter.restore()

        if option.state & QStyle.State_Selected:
            painter.setPen(option.palette.highlight().color())
            painter.drawRect(cella.adjusted(0, 0, -1, -1))

        if self._griglia.e_fratello(percorso):
            # DENTRO la cornice di selezione, non al suo posto: «e' dello
            # stesso frame» e «e' selezionato» sono due fatti diversi, e un
            # volto puo' essere entrambi. Sovrascriverne uno con l'altro
            # toglierebbe dallo schermo proprio l'informazione che si e'
            # appena chiesta.
            painter.setPen(QPen(COLORE_FRATELLO, 2))
            painter.drawRect(cella.adjusted(1, 1, -2, -2))
        painter.restore()


#Lo stesso peso della cache delle immagini, dalla stessa funzione: due copie
#della stessa misura sono il modo in cui una delle due smette di misurare la
#stessa cosa. Il nome locale resta `_peso` perche' e' quello che i punti di
#chiamata leggono meglio qui dentro.
_peso = peso_immagine


class Griglia(QListView):
    volto_aperto = pyqtSignal(object)
    selezione_cambiata = pyqtSignal(list)
    corrente_cambiato = pyqtSignal(object)
    menu_richiesto = pyqtSignal(object, object)

    # Il tetto e' in BYTE, non in voci, e la ragione e' misurata: le
    # maschere del faceset di prova pesano 64,0 KiB l'una, ma sono
    # 256x256 in scala di grigi -- una cartella a 512 ne peserebbe
    # quattro volte tanto e
    # sfonderebbe lo stesso tetto a numero senza che nessuna costante sia
    # cambiata. Prima non c'era nessun tetto: 50 000 volti
    # scorsi facevano ~3,2 GiB che non tornavano indietro finche' la
    # pagina restava aperta.
    #
    # 64 MiB sono 1024 maschere da 256 (misurate) o ~256 da 512: molto
    # piu' di una schermata di griglia (~400 celle al lato piu' piccolo su
    # uno schermo 1080p), che e' cio' che serve perche' scorrere avanti e
    # indietro non ridecodifichi.
    TETTO_MASCHERE_BYTE = 64 * 1024 * 1024

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lato = 128
        self._modo = MASCHERA_OFF
        self._fornitore_maschere = lambda voce: None
        self._maschere = OrderedDict()
        self._peso_maschere = 0
        self.decodificatore = Decodificatore(self)
        self.decodificatore.pronta.connect(self._su_immagine_pronta)

        self.setViewMode(QListView.IconMode)
        self.setResizeMode(QListView.Adjust)
        self.setUniformItemSizes(True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setItemDelegate(DelegatoVolti(self))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.doubleClicked.connect(self._su_doppio_click)
        self._fratelli = frozenset()
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._su_menu_richiesto)
        self._aggiorna_griglia()

    def lato(self):
        return self._lato

    def imposta_lato(self, valore):
        self._lato = _lato_ammesso(valore)
        self.decodificatore.dimentica_le_richieste()
        self._aggiorna_griglia()

    def _aggiorna_griglia(self):
        self.setGridSize(QSize(self._lato + MARGINE * 2, self._lato + MARGINE * 2))
        self.setIconSize(QSize(self._lato, self._lato))
        self.viewport().update()

    def modo_maschera(self):
        return self._modo

    def imposta_modo_maschera(self, modo):
        self._modo = modo
        self.viewport().update()

    def imposta_fornitore_maschere(self, fornitore):
        self._fornitore_maschere = fornitore
        self._maschere.clear()
        self._peso_maschere = 0

    def immagine_per(self, percorso, lato):
        if percorso is None:
            return None
        immagine = self.decodificatore.in_cache(percorso, lato)
        if immagine is None:
            self.decodificatore.richiedi(percorso, lato)
        return immagine

    def maschera_per(self, voce):
        if voce is None:
            return None
        chiave = (voce.nome, voce.mtime)
        if chiave in self._maschere:
            # Anche il colpo di cache aggiorna la recenza, come in
            # `Decodificatore.in_cache`: la cella che l'utente sta
            # guardando adesso e' ridisegnata a ogni repaint, e senza
            # questa riga sarebbe la prima a cadere.
            self._maschere.move_to_end(chiave)
            return self._maschere[chiave]
        byte = self._fornitore_maschere(voce)
        immagine = None
        if byte:
            candidata = QImage()
            if candidata.loadFromData(byte):
                immagine = candidata
        self._maschere[chiave] = immagine
        self._peso_maschere += _peso(immagine)
        while self._peso_maschere > self.TETTO_MASCHERE_BYTE and self._maschere:
            # Una voce senza maschera pesa zero e ne libera zero: il ciclo
            # va avanti e ne pota un'altra, non gira a vuoto. Le voci a
            # None restano comunque limitate dal numero di volti della
            # cartella, che la pagina tiene gia' tutti in `_abbinati`.
            _chiave, caduta = self._maschere.popitem(last=False)
            self._peso_maschere -= _peso(caduta)
        return immagine

    def _su_immagine_pronta(self, *_args):
        self.viewport().update()

    def _su_doppio_click(self, index):
        percorso = index.data(RUOLO_PERCORSO)
        if percorso is not None:
            self.volto_aperto.emit(percorso)

    def imposta_fratelli(self, percorsi):
        """I volti che vengono dallo stesso frame di quello corrente."""
        self._fratelli = frozenset(percorsi or ())
        self.viewport().update()

    def e_fratello(self, percorso):
        """Interrogato dal delegato una volta per cella dipinta: un
        `in` su un frozenset, non la costruzione di un insieme."""
        return percorso in self._fratelli

    def _su_menu_richiesto(self, punto):
        """Il percorso viene da `indexAt`, non dalla selezione: il menu
        deve agire su cio' che l'utente ha premuto. Su uno spazio vuoto si
        emette comunque, con None -- chi costruisce il menu decide cosa
        offrire, e non deve dedurlo dal silenzio."""
        index = self.indexAt(punto)
        percorso = index.data(RUOLO_PERCORSO) if index.isValid() else None
        self.menu_richiesto.emit(percorso, self.viewport().mapToGlobal(punto))

    #override
    def selectionChanged(self, selected, deselected):
        super().selectionChanged(selected, deselected)
        self.selezione_cambiata.emit(self.percorsi_selezionati())

    #override
    def currentChanged(self, current, previous):
        super().currentChanged(current, previous)
        self.corrente_cambiato.emit(
            current.data(RUOLO_PERCORSO) if current.isValid() else None)

    def percorsi_selezionati(self):
        return [i.data(RUOLO_PERCORSO) for i in self.selectionModel().selectedIndexes()]

    #override
    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            passo = 1 if event.angleDelta().y() > 0 else -1
            corrente = LATI.index(self._lato)
            self.imposta_lato(LATI[max(0, min(len(LATI) - 1, corrente + passo))])
            event.accept()
            return
        super().wheelEvent(event)
