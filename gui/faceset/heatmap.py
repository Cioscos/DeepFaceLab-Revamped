"""L'istogramma imbardata x beccheggio: il conto, senza widget.

Domini FISSI e non ricavati dai dati: se il dominio seguisse i dati,
cancellare i volti estremi cambierebbe la forma della griglia sotto le
mani dell'utente, e due cartelle non sarebbero confrontabili a occhio --
che e' meta' del motivo per cui la si apre.

Scala logaritmica per la stessa ragione per cui la heatmap esiste: un
faceset e' violentemente concentrato sul frontale, e in scala lineare
tutto cio' che non e' il centro diventa indistinguibile da zero. Cioe'
sparisce esattamente l'informazione che si sta cercando.

Il rollio non c'e': voce 1.21 del registro, satura al clip nel 97-99% dei
volti.
"""
import math

from gui.numeri import numero_finito

LIMITE_RAD = math.pi / 2
BIN_AMMESSI = (8, 12, 16)


def _indice(valore, bins):
    # Chi chiama garantisce solo "finito" (vedi numero_finito), non "dentro
    # il dominio": un valore enorme (es. 1e308) farebbe traboccare il float
    # PRIMA del clip di min/max sotto, perche' la divisione stessa diventa
    # infinita. Il clip qui e' innocuo per i valori gia' nel dominio (lo
    # stesso caso di 10.0 oltre [-pi/2, pi/2] gia' coperto) e rende il resto
    # della funzione sicuro per costruzione.
    valore = max(-LIMITE_RAD, min(LIMITE_RAD, valore))
    quota = (valore + LIMITE_RAD) / (2 * LIMITE_RAD)
    return max(0, min(bins - 1, int(quota * bins)))


def bin_di(yaw, pitch, bins):
    """(colonna, riga) oppure None se la posa non c'e' o non e' un numero."""
    if not numero_finito(yaw) or not numero_finito(pitch):
        return None
    colonna = _indice(float(yaw), bins)
    # riga 0 = beccheggio MASSIMO, perche' e' cosi' che sta sullo schermo
    riga = bins - 1 - _indice(float(pitch), bins)
    return (colonna, riga)


def istogramma(voci, bins):
    matrice = [[0] * bins for _ in range(bins)]
    senza_posa = 0
    for v in voci:
        posizione = bin_di(v.yaw, v.pitch, bins)
        if posizione is None:
            senza_posa += 1
            continue
        colonna, riga = posizione
        matrice[riga][colonna] += 1
    return matrice, senza_posa


def intensita(conteggio, massimo):
    if not conteggio or not massimo or massimo <= 0:
        return 0.0
    return math.log1p(conteggio) / math.log1p(massimo)


def etichette_gradi(bins):
    passo = 2 * LIMITE_RAD / bins
    etichette = []
    for i in range(bins):
        centro = math.degrees(-LIMITE_RAD + passo * (i + 0.5))
        etichette.append("%d°" % round(centro))
    return etichette


def passo_etichette(passo_cella, ingombro):
    """Ogni quante celle scrivere un'etichetta perche' non si tocchino.

    L'ingombro non e' una costante: cambia col font, e il font cambia con
    `View > Text size`. A sedici bin in una finestra stretta le celle sono
    piu' strette di «-79°», e scriverle tutte le farebbe sovrapporre --
    cioe' renderebbe illeggibile proprio cio' che si e' aggiunto per
    leggere.
    """
    if passo_cella <= 0:
        return 1
    return max(1, -(-ingombro // passo_cella))


from PyQt5.QtCore import QRect, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QWidget

from gui import testi

CHIAVE_COLLASSATA = "faceset/heatmap_collassata"
CHIAVE_BINS = "faceset/heatmap_bins"
LATO_CELLA_MINIMO = 25
_SFONDO_VUOTO = QColor(70, 70, 70)
_PIENO = QColor(120, 200, 160)


def bin_di_percorsi(abbinati, accesi, bins):
    """I percorsi dei volti che cadono nei bin accesi. None = nessun filtro."""
    if not accesi:
        return None
    scelti = set()
    for percorso, voce in abbinati.items():
        posizione = bin_di(voce.yaw, voce.pitch, bins)
        if posizione is not None and posizione in accesi:
            scelti.add(percorso)
    return scelti


class WidgetHeatmap(QWidget):
    selezione_bin_cambiata = pyqtSignal(set)

    def __init__(self, impostazioni, parent=None):
        super().__init__(parent)
        self._impostazioni = impostazioni
        self._bins = self._bins_ricordati()
        self._matrice = [[0] * self._bins for _ in range(self._bins)]
        self._massimo = 0
        self._senza_posa = 0
        self._accesi = set()
        self._voci = []
        self._collassata = self._impostazioni.value(CHIAVE_COLLASSATA, False, type=bool)
        self._adegua_altezza()

    def _banda_inferiore(self):
        """Lo spazio sotto la mappa per i gradi di imbardata."""
        return self.fontMetrics().height() + 2

    def _ingombro_etichette(self):
        """La larghezza dell'etichetta piu' larga, con un po' d'aria.

        Si misura, non si fissa: «-79°» e «11°» sono larghi diversi, e a
        scala `xlarge` entrambi sono larghi il doppio.
        """
        metriche = self.fontMetrics()
        return max(metriche.horizontalAdvance(t)
                   for t in etichette_gradi(self._bins)) + 6

    def _banda_sinistra(self):
        """Lo spazio a sinistra per i gradi di beccheggio."""
        return self._ingombro_etichette()

    def _adegua_altezza(self):
        """L'altezza segue i bin, perche' cio' che deve restare costante e'
        la CELLA.

        A 120 px e otto bin la cella e' 15 px, e il bin acceso -- un
        rettangolo di due pixel di bordo -- non si vede: e' la ragione per
        cui l'altezza minima era stata fissata a 200. Ma 200 e' 8 x 25, e
        tenerla fissa mentre i bin salgono a 16 riporta la cella a 12 px,
        cioe' peggio del caso che quella riga aveva chiuso (visto in uno
        scatto a 16x16: due quadratini di ~11 px, quasi solo bordo). Il
        default non cambia -- 8 x 25 fa esattamente i 200 di prima, piu' la
        fascia dei gradi che sta sotto la mappa: se non
        entrasse nel minimo, le celle si rimpicciolirebbero per farle
        posto, cioe' il contrario di quel che questa riga difende.
        """
        self.setMinimumHeight(LATO_CELLA_MINIMO * self._bins
                              + self._banda_inferiore())

    def _bins_ricordati(self):
        """Il numero di bin dell'ultima sessione, o 8.

        Stessa regola di `ScalaTesto`: un valore illeggibile o non ammesso
        vale come il default -- un 7 scritto a mano nel file darebbe una
        griglia che il resto del codice non si aspetta, e la pagina deve
        aprirsi comunque.
        """
        try:
            valore = self._impostazioni.value(CHIAVE_BINS, BIN_AMMESSI[0], type=int)
        except (TypeError, ValueError):
            # Un QSettings vero SOLLEVA su `type=int` con una stringa che
            # non e' un numero; il doppio dei test torna il default. Le due
            # vie portano allo stesso posto, e nessuna delle due impedisce
            # alla pagina di aprirsi.
            return BIN_AMMESSI[0]
        return valore if valore in BIN_AMMESSI else BIN_AMMESSI[0]

    def bins(self):
        return self._bins

    def imposta_bins(self, n):
        if n not in BIN_AMMESSI:
            return
        self._bins = n
        self._impostazioni.setValue(CHIAVE_BINS, n)
        self._adegua_altezza()
        # `pulisci_selezione` e non `_accesi.clear()`: chi tiene il filtro
        # e' la pagina, e lo tiene come insieme di PERCORSI. Svuotando in
        # silenzio, la griglia resterebbe filtrata sui percorsi dei bin di
        # prima mentre la mappa non ne mostra piu' nessuno acceso -- cioe'
        # una fetta senza piu' niente a schermo che lo dica.
        self.pulisci_selezione()
        self.aggiorna(self._voci)

    def aggiorna(self, voci):
        self._voci = list(voci)
        self._matrice, self._senza_posa = istogramma(self._voci, self._bins)
        self._massimo = max((max(r) for r in self._matrice), default=0)
        self.update()

    def senza_posa(self):
        return self._senza_posa

    def estremi(self):
        """(minimo, massimo) dei bin NON vuoti: gli estremi della scala.

        Il minimo non e' zero: un bin vuoto e' disegnato a parte (fondo
        grigio e croce), non sta sulla rampa di colori, e dichiararlo come
        estremo della legenda direbbe che il colore piu' chiaro vale zero
        volti quando vale uno.
        """
        pieni = [c for riga in self._matrice for c in riga if c]
        return (min(pieni), max(pieni)) if pieni else (0, 0)

    def bin_accesi(self):
        return set(self._accesi)

    def pulisci_selezione(self):
        self._accesi.clear()
        self.selezione_bin_cambiata.emit(set())
        self.update()

    def collassata(self):
        return self._collassata

    def imposta_collassata(self, valore):
        self._collassata = bool(valore)
        self._impostazioni.setValue(CHIAVE_COLLASSATA, self._collassata)
        self.setVisible(not self._collassata)

    def _commuta_bin(self, posizione, aggiungi):
        if aggiungi:
            if posizione in self._accesi:
                self._accesi.discard(posizione)
            else:
                self._accesi.add(posizione)
        elif self._accesi == {posizione}:
            self._accesi.clear()
        else:
            self._accesi = {posizione}
        self.selezione_bin_cambiata.emit(set(self._accesi))
        self.update()

    def _geometria(self):
        """(origine_x, lato_di_una_cella) della mappa disegnata.

        Le celle sono QUADRATE: imbardata e beccheggio hanno lo stesso
        dominio, e lo stesso passo angolare deve occupare gli stessi
        pixel sui due assi. Prendendo tutta la larghezza e l'altezza
        minima le celle venivano 197x15 -- una striscia in cui il buco
        che si va a cercare non si distingue da niente, e i due bin
        accesi erano invisibili.

        Una funzione sola perche' disegno e click DEVONO leggere la
        stessa geometria: due copie che divergono accendono un bin
        diverso da quello cliccato, ed e' la classe di difetto piu'
        cara di questa GUI. Anche le fasce delle etichette passano
        di qui per la stessa ragione: sono spazio RISERVATO, e un bin che
        finisse sotto le scritte sarebbe cliccabile dove non si vede.
        """
        sinistra, sotto = self._banda_sinistra(), self._banda_inferiore()
        larghezza = max(0, self.width() - sinistra)
        altezza = max(0, self.height() - sotto)
        passo = max(1, min(larghezza // self._bins, altezza // self._bins))
        return (sinistra + (larghezza - passo * self._bins) // 2, passo)

    def _cella_sotto(self, punto):
        """(colonna, riga), oppure None fuori dalla mappa disegnata."""
        origine_x, passo = self._geometria()
        colonna = (punto.x() - origine_x) // passo
        riga = punto.y() // passo
        if not (0 <= colonna < self._bins and 0 <= riga < self._bins):
            return None
        return (int(colonna), int(riga))

    #override
    def mousePressEvent(self, event):
        cella = self._cella_sotto(event.pos())
        # Un click sulla fascia vuota accanto alla mappa non e' un click
        # su un bin: prima ne accendeva uno di bordo, cioe' filtrava la
        # griglia senza che l'utente avesse puntato niente.
        if cella is None:
            return
        self._commuta_bin(cella, bool(event.modifiers() & Qt.ControlModifier))

    def _etichette_asse(self, passo):
        """[(indice_di_cella, testo)]: quelle che ci stanno, in gradi.

        Lo stesso elenco serve i due assi -- imbardata e beccheggio hanno
        lo stesso dominio -- e sull'asse verticale l'indice i vale la riga
        `bins - 1 - i`, perche' la riga 0 e' il beccheggio MASSIMO.
        """
        etichette = etichette_gradi(self._bins)
        ogni = passo_etichette(passo, self._ingombro_etichette())
        return [(i, etichette[i]) for i in range(self._bins) if i % ogni == 0]

    def _disegna_etichette(self, painter, origine_x, passo):
        """I gradi sugli assi: senza, la mappa non dice a quale posa
        corrisponda una cella, ed e' l'unica cosa che una distribuzione
        deve dire."""
        painter.setPen(self.palette().windowText().color())
        ingombro = self._ingombro_etichette()
        fondo = passo * self._bins
        sotto = self._banda_inferiore()
        sinistra = self._banda_sinistra()
        for i, testo in self._etichette_asse(passo):
            centro = origine_x + i * passo + passo // 2
            painter.drawText(QRect(centro - ingombro // 2, fondo, ingombro, sotto),
                             Qt.AlignHCenter | Qt.AlignVCenter, testo)
            riga = self._bins - 1 - i
            # Accanto alla MAPPA, non al bordo della finestra: la mappa e'
            # centrata in orizzontale, e nel primo scatto i gradi del
            # beccheggio stavano mezzo schermo piu' a sinistra della
            # griglia che etichettavano -- una colonna di numeri che non
            # sembrava appartenere a niente.
            painter.drawText(QRect(origine_x - sinistra, riga * passo,
                                   sinistra - 4, passo),
                             Qt.AlignRight | Qt.AlignVCenter, testo)

    #override
    def paintEvent(self, event):
        painter = QPainter(self)
        origine_x, passo = self._geometria()
        for riga in range(self._bins):
            for colonna in range(self._bins):
                rect = QRect(origine_x + colonna * passo, riga * passo, passo, passo)
                conteggio = self._matrice[riga][colonna]
                if conteggio == 0:
                    # Un bin vuoto e' graficamente diverso da uno quasi
                    # vuoto: il buco e' il risultato che si va a cercare.
                    painter.fillRect(rect, _SFONDO_VUOTO)
                    painter.setPen(QColor(100, 100, 100))
                    painter.drawLine(rect.topLeft(), rect.bottomRight())
                else:
                    colore = QColor(_PIENO)
                    colore.setAlphaF(0.15 + 0.85 * intensita(conteggio, self._massimo))
                    painter.fillRect(rect, colore)
                if (colonna, riga) in self._accesi:
                    # Due pixel, non uno: un bin acceso e' l'unica cosa
                    # che dice che la griglia sotto e' filtrata, e un
                    # filo di un pixel su un fondo scuro non si vede.
                    painter.setPen(QPen(self.palette().highlight().color(), 2))
                    painter.drawRect(rect.adjusted(1, 1, -1, -1))
        self._disegna_etichette(painter, origine_x, passo)
