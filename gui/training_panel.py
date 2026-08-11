"""La scheda di un training: anteprime, grafico, stato, comandi.

E' una vista di un job, non il suo proprietario: il chiamante la costruisce
al primo Start e le passa gli eventi del canale; lo stato vive qui dentro e
sopravvive alla chiusura della scheda perche' l'oggetto non viene distrutto.

Due modi, uno solo alla volta. In *diretta* segue il training: ogni evento
`preview` sostituisce le immagini, ogni evento `iter` allunga la curva. In
*storico* e' fermo a un'iterazione, e allora **si spostano insieme** -- le
immagini vengono dalla cartella <modello>_history/ e la curva si ferma alla
stessa iterazione. Il numero d'iterazione e' la chiave che unisce le due
sorgenti: il nome del file da una parte, la colonna `iter` del CSV
dall'altra.
"""
import threading
import time
from pathlib import Path

from PyQt5.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QSlider, QVBoxLayout, QWidget)

from gui import testi
from gui.finestra_immagine import FinestraImmagine
from gui.loss_plot import COLORI, INTERVALLI, LossPlot
from gui.loss_source import LossSource
from gui.preview_grid import celle, etichetta, normalizza, righe_effettive
from gui.preview_history import StoricoAnteprime
from gui.rimozione import svuota
from gui.status_line import (
    RitmoIterazioni, iterazione_utilizzabile, obiettivo_valido, valori_di_stato)
from gui.tessere_stato import TessereStato

RITARDO_CURSORE_MS = 120     # coalizza un trascinamento veloce
LATO_CONTORNO = 96
LATO_MINIATURA = 64

#Nomi tenuti cosi' -- li importano i test -- ma il valore viene da gui.testi:
#ogni testo che l'utente legge nasce li', non qui.
SENZA_STORICO = testi.SLIDER_NO_HISTORY
CON_STORICO = testi.SLIDER_WITH_HISTORY


def _scalato(immagine, dimensione):
    return QPixmap.fromImage(immagine).scaled(
        dimensione, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def _separatore():
    """Una riga verticale, per raggruppare i bottoni dei comandi in basso."""
    linea = QFrame()
    linea.setFrameShape(QFrame.VLine)
    linea.setFrameShadow(QFrame.Sunken)
    return linea


class _CellaLaterale(QWidget):
    """Un riquadro della colonna a destra: l'immagine, il nome sotto -- come
    la finestra cv2 le etichettava -- e adesso anche i due gesti dell'utente.

    Click promuove questa cella nel riquadro grande; doppio click apre la
    finestra a dimensione naturale. Entrambi passano dal pannello, che ha la
    rete di ognuno dei due (`mostra_cella` tramite `_ridisegna`,
    `apri_a_dimensione_naturale` con la propria) -- qui non c'e' nessun
    `try`, perche' non ne serve uno: se il pannello sparisse sotto i piedi
    di un click in corso lo si vedrebbe come ogni altro guasto del pannello,
    non come uno di questo widget.
    """

    def __init__(self, immagine, lato, testo, riga, colonna, pannello, parent=None):
        super().__init__(parent)
        self.setProperty("cella", (riga, colonna))
        self._riga = riga
        self._colonna = colonna
        self._pannello = pannello

        contenuto = QVBoxLayout(self)
        contenuto.setContentsMargins(0, 0, 0, 0)
        quadro = QLabel()
        quadro.setAlignment(Qt.AlignCenter)
        quadro.setPixmap(_scalato(immagine, QSize(lato, lato)))
        contenuto.addWidget(quadro)
        nome = QLabel(testo)
        nome.setAlignment(Qt.AlignCenter)
        contenuto.addWidget(nome)
        #Sul contenitore, non sull'immagine o sull'etichetta: i due gesti si
        #fanno su tutta la cella, quindi il suggerimento che li nomina deve
        #comparire su tutta la cella. Senza, niente a schermo dice che una
        #cella si puo' cliccare.
        self.setToolTip(testi.cell_tip(testo))

    #override
    def mousePressEvent(self, event):
        self._pannello.mostra_cella(self._colonna)

    #override
    def mouseDoubleClickEvent(self, event):
        self._pannello.apri_a_dimensione_naturale(self._riga, self._colonna)


def _righe_dei_nomi(nomi_file):
    """Le righe "src: ... / dst: ..." che la finestra cv2 metteva in testa.

    Il canale le manda come una lista per lato, ognuna con un nome per
    campione. Qualunque altra forma vale come assente: un nome sbagliato
    sotto un volto mente su cio' che si sta guardando.

    La regola vale ai **due** livelli, e per un po' e' stata applicata solo a
    quello interno: `nomi_file` stesso arriva dal canale come tutto il resto,
    e un `5` al suo posto faceva saltare lo `zip` prima ancora di guardare i
    lati. Mezzo predicato -- che difende il dentro e non il fuori -- e' il
    modo in cui questa famiglia di guasti e' arrivata fino a un click.
    """
    if isinstance(nomi_file, str) or not hasattr(nomi_file, "__iter__"):
        nomi_file = []
    righe = []
    for lato, nomi in zip(("src", "dst"), nomi_file or []):
        if isinstance(nomi, str) or not hasattr(nomi, "__iter__"):
            continue
        righe.append("%s: %s" % (lato, " ".join(str(n) for n in nomi)))
    return righe


def _svuota(layout):
    """Toglie i widget di un layout: il contorno si ricostruisce ogni volta.

    Nome tenuto -- lo importano i test -- ma il come sta in `gui.rimozione`,
    insieme al motivo per cui nascondere prima di staccare non e' un
    dettaglio: senza, ogni cella tolta diventa una finestrella a se'.
    """
    svuota(layout)


class _Riquadro(QLabel):
    """Il riquadro grande: tiene l'immagine sorgente e si riscala da solo.

    Scalare al momento del disegno non basta. Il pannello ridisegna quando
    arriva roba nuova, e in diretta la roba nuova e' l'anteprima successiva:
    col salvataggio di default sono venticinque minuti, durante i quali la
    finestra si ridimensiona e l'immagine resterebbe della misura di prima.
    L'etichetta se ne accorge da se', dalla sorgente che ha in mano.

    La politica di dimensionamento e' Ignored in entrambe le direzioni, ed e'
    una precauzione dichiarata come tale: in teoria setPixmap porta la
    dimensione del pixmap nel suggerimento dell'etichetta, il layout lo
    asseconda e una riscalatura ne provoca un'altra -- verso il basso,
    perche' KeepAspectRatio lascia sempre un margine su un lato. Provato a
    riprodurlo coi due bracci su una finestra di primo livello e su tre
    forme d'immagine (alta, larga, quadrata): identici cifra per cifra, e
    fermi al primo giro. Qui il minimo e i fattori di allungamento decidono
    tutto, quindi la politica non cambia niente di osservabile -- resta
    perche' costa una riga e toglie di mezzo la classe di problema, non
    perche' un rimbalzo sia stato visto.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sorgente = None
        #Impostato dal pannello dopo la costruzione: senza immagine il
        #doppio click non deve fare niente, quindi None e' un valore
        #legittimo, non un buco da riempire a forza.
        self.doppio_click = None

    def imposta_immagine(self, immagine):
        self._sorgente = immagine
        self._disegna()

    def pulisci(self):
        self._sorgente = None
        self.clear()

    #override
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._disegna()

    #override
    def mouseDoubleClickEvent(self, event):
        if self.doppio_click is not None:
            self.doppio_click()

    def _disegna(self):
        if self._sorgente is not None:
            self.setPixmap(_scalato(self._sorgente, self.size()))


class _BottoneMiniatura(QPushButton):
    """Una miniatura della striscia dei campioni: click sceglie il
    campione, come oggi -- doppio click apre la sua cella a dimensione
    naturale, il gesto nuovo di questo widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.doppio_click = None
        #Di default un QPushButton e' orizzontalmente Minimum: il
        #sizeHint() e' un minimo, non un massimo, e cresce per riempire lo
        #spazio che il layout gli offre -- in una striscia con poche
        #miniature quello spazio e' tanto, e il risultato era un riquadro
        #quasi vuoto con l'icona persa al centro invece di una miniatura
        #della sua misura naturale.
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    #override
    def mouseDoubleClickEvent(self, event):
        super().mouseDoubleClickEvent(event)
        if self.doppio_click is not None:
            self.doppio_click()


class TrainingPanel(QWidget):
    comando = pyqtSignal(str)
    #La sorgente della loss, letta altrove e consegnata qui gia' pronta,
    #con il numero della lettura che l'ha prodotta e l'avviso se e' andata
    #storta. Emesso dal thread che l'ha costruita: la connessione automatica
    #lo fa diventare un evento in coda, quindi lo slot gira sul thread
    #dell'interfaccia e nessuno stato e' toccato da due thread insieme.
    _sorgente_pronta = pyqtSignal(object, int, str)

    def __init__(self, previews_dir, parent=None):
        super().__init__(parent)
        self.previews_dir = Path(previews_dir)
        self._immagini = {}          # nome anteprima -> QImage intera
        self._descrittori = {}       # nome anteprima -> descrittore o None
        self._nomi_file = None
        self.anteprima_selezionata = None
        self.campione_selezionato = 0
        self.iterazione_mostrata = None
        self.modo = "diretta"
        #Una sorgente che punta a un file che non esistera' mai: i punti vivi
        #degli eventi `iter` funzionano da subito, e l'evento `hello` la
        #rimpiazza con quella vera appena si sa come si chiama il modello.
        self.loss = LossSource(self.previews_dir / "loss_history.csv")
        self._vivi = []                  # i punti degli eventi iter, per il travaso
        self._generazione = 0            # cresce a ogni lettura del CSV
        self._consegnata = 0             # l'ultima presa in consegna: se e'
                                         # indietro, una lettura e' in volo
        self._sorgente_pronta.connect(self._prendi_in_consegna)
        self.storico = None
        self._model = {"name": None, "dir": None, "target_iter": 0}
        self._ritmo = RitmoIterazioni()
        #Le coppie dell'ultimo evento `iter`, tenute perche' le tessere si
        #ridisegnano anche quando non arriva nessun evento -- entrare e
        #uscire dallo storico ne cambia una senza che il training abbia
        #detto niente di nuovo.
        self._coppie_stato = ()
        self._iterazione_viva = None     # l'ultima annunciata da un evento preview
        self._immagine_storico = None
        self._iterazioni = []            # gli scatti che lo storico ha su disco
        self._cella_risultato = None
        #Solo la colonna e' uno stato della vista che l'utente puo' promuovere
        #con un click: None finche' nessun click l'ha mai toccata. La riga
        #resta quella che era gia' -- campione_selezionato o il risultato
        #dichiarato -- apposta, cosi' scegliere un altro campione muove
        #ancora il riquadro grande anche dopo aver promosso una colonna,
        #invece di restare congelato sulla riga di quando si e' cliccato.
        #Un descrittore nuovo che non ha piu' quella colonna (meno colonne di
        #prima) la riporta al risultato dichiarato invece di indicizzare
        #fuori.
        self._colonna_mostrata = None
        #Il pubblico: la coppia (riga, colonna) davvero mostrata nel
        #riquadro grande adesso, ricalcolata a ogni disegno -- None solo
        #prima che una griglia sia mai stata disegnata. E' anche cio' che il
        #doppio click sul riquadro grande apre.
        self.cella_mostrata = None
        #Un campo per produttore, non una casella sola. Chi scrive qui non
        #si parla: la consegna del CSV arriva da un thread quando gli pare,
        #l'anteprima da un evento del figlio, l'esito quando il figlio
        #muore. Con una stringa condivisa l'ultimo che scrive cancella
        #l'altro -- un training che muore subito perdeva il codice d'uscita,
        #sostituito dalla consegna vuota di una lettura riuscita. Ognuno
        #possiede il proprio e la riga si compone al momento di scriverla.
        self._avviso_anteprima = ""      # le anteprime annunciate
        self._avviso_loss = ""           # la lettura del CSV
        self._avviso_disegno = ""        # il disegno delle immagini
        self._avviso_evento = ""         # un evento che non si e' potuto applicare
        self._esito_job = ""             # il figlio che non c'e' piu'

        layout = QVBoxLayout(self)

        #Il selettore dell'anteprima -- quale delle immagini che il modello
        #manda -- e, quando il descrittore dice righe_sono_campioni, un
        #secondo modo di scegliere il campione: la stessa cosa che un click
        #sulla striscia in basso fa gia', non un terzo stato che vive da
        #solo (vedi _sincronizza_selettore_campione).
        alto = QHBoxLayout()
        alto.addWidget(QLabel(testi.PREVIEW_LABEL))
        self.selettore = QComboBox()
        self.selettore.setToolTip(testi.PREVIEW_SELECTOR_TIP)
        self.selettore.currentTextChanged.connect(self._su_cambio_anteprima)
        alto.addWidget(self.selettore, 1)
        self.selettore_campione = QComboBox()
        self.selettore_campione.setToolTip(testi.SAMPLE_SELECTOR_TIP)
        self.selettore_campione.setVisible(False)     # niente da scegliere finche' non c'e' una griglia a campioni
        self.selettore_campione.currentIndexChanged.connect(self.seleziona_campione)
        alto.addWidget(self.selettore_campione)
        layout.addLayout(alto)

        #Le tessere, sopra il centro: il progresso si legge da lontano,
        #prima ancora di guardare le immagini o il grafico.
        self.tessere = TessereStato()
        layout.addWidget(self.tessere)

        #I due allunghi ai lati raccolgono cio' che ne' il riquadro grande ne'
        #la colonna possono usare: l'anteprima e' quadra e limitata
        #dall'altezza, quindi la larghezza avanzata esiste sempre -- l'unica
        #scelta e' dove metterla. Ai lati il gruppo anteprima+celle resta
        #unito e centrato; dentro il riquadro grande, come prima, diventava
        #una banda di niente proprio fra le due cose da confrontare.
        centro = QHBoxLayout()
        centro.addStretch(1)
        self.risultato = _Riquadro()
        self.risultato.setAlignment(Qt.AlignCenter)
        self.risultato.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.risultato.setMinimumSize(256, 256)
        self.risultato.doppio_click = self._apri_grande
        centro.addWidget(self.risultato, 1)
        self.contorno = QVBoxLayout()      # le altre celle della riga
        #Fattore di allungamento 0, e la colonna si prende la larghezza che
        #le serve: le celle sono a lato fisso (LATO_CONTORNO), quindi con un
        #fattore >0 lo spazio in piu' finiva dentro la colonna e restava
        #vuoto -- una banda di centinaia di pixel di niente fra il riquadro
        #grande e le celle, a ogni scala. Adesso tutto cio' che avanza va al
        #riquadro grande, che e' l'unica cosa che sa cosa farsene.
        self.contorno_widget = QWidget()
        self.contorno_widget.setLayout(self.contorno)
        centro.addWidget(self.contorno_widget, 0)
        centro.addStretch(1)
        layout.addLayout(centro, 1)

        self.striscia = QHBoxLayout()      # miniature dei campioni
        striscia_widget = QWidget()
        striscia_widget.setLayout(self.striscia)
        layout.addWidget(striscia_widget)

        #La barra del grafico: nome, l'intervallo che lo comanda -- prima
        #era in cima alla scheda, lontanissimo da cio' che comanda, ed era
        #tutta la ragione per cui non si capiva -- e la legenda dei colori,
        #accanto perche' e' li' che serve leggerla.
        self.barra_grafico = QWidget()
        riga_grafico = QHBoxLayout(self.barra_grafico)
        riga_grafico.setContentsMargins(0, 0, 0, 0)
        riga_grafico.addWidget(QLabel(testi.LOSS_CHART))
        riga_grafico.addWidget(QLabel(testi.RANGE_LABEL))
        self.intervallo = QComboBox()
        self.intervallo.setToolTip(testi.RANGE_TIP)
        for n in INTERVALLI:
            testo = testi.RANGE_ALL if n == 0 else testi.range_last_label(n)
            suggerimento = testi.RANGE_ALL_TIP if n == 0 else testi.RANGE_LAST_TIP
            self.intervallo.addItem(testo, n)
            self.intervallo.setItemData(self.intervallo.count() - 1, suggerimento, Qt.ToolTipRole)
        self.intervallo.currentIndexChanged.connect(
            lambda _i: self.plot.imposta_intervallo(self.intervallo.currentData()))
        riga_grafico.addWidget(self.intervallo)
        self.legenda = QLabel(testi.loss_legend_html(COLORI[0], COLORI[1]))
        riga_grafico.addWidget(self.legenda)
        riga_grafico.addStretch(1)
        layout.addWidget(self.barra_grafico)

        self.plot = LossPlot()
        layout.addWidget(self.plot)

        riga_cursore = QHBoxLayout()
        self.cursore = QSlider(Qt.Horizontal)
        self.cursore.setEnabled(False)
        self.cursore.setToolTip(SENZA_STORICO)
        self.cursore.valueChanged.connect(self._su_cursore)
        riga_cursore.addWidget(self.cursore, 1)
        #Dice dov'e' fermo il pannello: "Live" mentre segue il training,
        #l'iterazione quando e' fermo nello storico -- aggiornata in vai_a
        #e in torna_in_diretta.
        self.etichetta_cursore = QLabel(testi.cursor_live())
        riga_cursore.addWidget(self.etichetta_cursore)
        layout.addLayout(riga_cursore)

        self.stato = QLabel("")
        self.stato.setWordWrap(True)
        self.stato.setVisible(False)     # niente da dire finche' non arriva un avviso
        layout.addWidget(self.stato)

        #I comandi, raggruppati con due separatori: seguire/aggiornare
        #l'anteprima, poi salvare/backuppare senza fermarsi, poi lo Stop --
        #che ha il proprio ruolo per il foglio di stile (il rosso viene da
        #li', non da un colore impostato a mano) e resta l'ultimo, isolato
        #dagli altri quattro perche' e' l'unico che ferma il training.
        #
        #Stanno in un contenitore proprio, non direttamente nel layout del
        #pannello, perche' a job finito sparisce la riga *intera*: nasconderli
        #per nome lasciava indietro i due separatori, che un nome non ce
        #l'hanno, e ogni comando aggiunto qui domani avrebbe lo stesso
        #destino.
        self.comandi = QWidget()
        basso = QHBoxLayout(self.comandi)
        basso.setContentsMargins(0, 0, 0, 0)
        self.diretta_button = QPushButton(testi.LIVE)
        self.diretta_button.setToolTip(testi.LIVE_TIP)
        self.diretta_button.clicked.connect(self.torna_in_diretta)
        self.aggiorna_button = QPushButton(testi.REFRESH_PREVIEW)
        self.aggiorna_button.setToolTip(testi.REFRESH_PREVIEW_TIP)
        self.aggiorna_button.clicked.connect(lambda: self.comando.emit("preview"))
        basso.addWidget(self.diretta_button)
        basso.addWidget(self.aggiorna_button)
        basso.addWidget(_separatore())
        self.save_button = QPushButton(testi.SAVE)
        self.save_button.setToolTip(testi.SAVE_TIP)
        self.save_button.clicked.connect(lambda: self.comando.emit("save"))
        self.backup_button = QPushButton(testi.BACKUP)
        self.backup_button.setToolTip(testi.BACKUP_TIP)
        self.backup_button.clicked.connect(lambda: self.comando.emit("backup"))
        basso.addWidget(self.save_button)
        basso.addWidget(self.backup_button)
        basso.addWidget(_separatore())
        self.stop_button = QPushButton(testi.STOP)
        self.stop_button.setProperty("ruolo", "stop")
        self.stop_button.setToolTip(testi.STOP_TIP)
        self.stop_button.clicked.connect(lambda: self.comando.emit("close"))
        basso.addWidget(self.stop_button)
        layout.addWidget(self.comandi)

        self._timer_cursore = QTimer(self)
        self._timer_cursore.setSingleShot(True)
        self._timer_cursore.timeout.connect(self._carica_dallo_storico)

    # -- gli eventi del canale ---------------------------------------------

    def applica_evento(self, event):
        """Un evento del canale eventi del figlio, di qualunque tipo.

        **Non solleva mai**, ed e' la parte del contratto che vale la pena
        dichiarare: un'iterazione inutilizzabile ferma l'evento che la porta
        -- con un `ValueError` che pero' non esce di qui -- e finisce nella
        riga di stato, dove l'utente la legge.

        Prima erano due politiche per lo stesso valore storto: qui si
        sollevava e negli altri quattro punti che lo incontrano si degradava
        in silenzio, con la differenza che questo e' il metodo *pubblico* e
        funzionava soltanto perche' chi lo chiama ha una rete. Una rete del
        chiamante non e' un contratto: e' una proprieta' di un chiamante, e
        il prossimo consumatore l'avrebbe scoperta come si scoprono queste
        cose, con il processo che se ne va. La rete sta dove sta la superficie
        che puo' dirlo -- questo oggetto -- non dove capita che ci sia.

        Quella del chiamante resta, e non e' ridondanza: copre *qualunque*
        pannello, anche uno che questa promessa non la mantenesse, e protegge
        il secondo lettore dello stesso evento, che e' la striscia.
        """
        try:
            self._applica_evento(event)
        except Exception as errore:
            self.evento_non_applicato(errore)

    def _applica_evento(self, event):
        tipo = event.get("type")
        if tipo == "hello":
            self._su_hello(event)
        elif tipo == "preview":
            self._su_preview(event)
        elif tipo == "iter":
            self._su_iter(event)
        elif tipo == "save":
            #Il CSV appena riscritto rimpiazza i punti vivi con quelli veri,
            #fin dove il salvataggio dice di essere arrivato -- ma non
            #mentre una lettura e' in volo: in quella finestra self._vivi e'
            #l'unico posto dove stanno i punti che la sorgente in arrivo non
            #ha, e l'iterazione dichiarata qui non dice niente su dove quella
            #lettura si sia fermata.
            self.loss.ricarica()
            if self._consegnata == self._generazione:
                self._dimentica_i_vivi(event.get("iter"))
            self._aggiorna_grafico()

    def _su_hello(self, event):
        nome, cartella = event.get("model_name"), event.get("model_dir")
        #L'obiettivo entra validato, per la stessa ragione dell'iterazione:
        #viene confrontato con lei a ogni evento, e ricordarlo storto
        #romperebbe ogni evento buono da qui alla fine della corsa.
        self._model = {"name": nome, "dir": cartella,
                       "target_iter": obiettivo_valido(event.get("target_iter"))}
        if nome and cartella:
            self._leggi_il_csv(Path(cartella) / ("%s_loss_history.csv" % nome))
            self.storico = StoricoAnteprime(cartella, nome)
        self._aggiorna_cursore()
        self._aggiorna_grafico()

    def _leggi_il_csv(self, percorso):
        """La prima lettura del CSV, fuori dal thread dell'interfaccia.

        Su un modello da mezzo milione di iterazioni sono decine di
        megabyte, e letti qui bloccherebbero la finestra all'apertura della
        scheda. Il lavoratore costruisce una sorgente tutta sua, la carica e
        la consegna gia' pronta: niente stato condiviso da mutare in due, un
        solo passaggio di consegne. Intanto il pannello disegna i punti vivi
        che gli eventi `iter` gli portano.
        """
        #Un hello e' una vita nuova: i punti vivi di prima appartengono alla
        #corsa che si e' chiusa, e dopo un rollback a un checkpoint piu'
        #vecchio sarebbero iterazioni che sul disco non esistono piu'.
        self._vivi = []
        self._generazione += 1
        generazione = self._generazione

        def lavora():
            sorgente = LossSource(percorso)
            sorgente.ricarica()
            avviso = ("" if sorgente.errore is None else
                      testi.loss_history_unreadable(sorgente.errore))
            try:
                self._sorgente_pronta.emit(sorgente, generazione, avviso)
            except RuntimeError:
                #Il pannello e' stato distrutto mentre leggevamo -- la
                #finestra che si chiude, non la scheda, che invece lascia
                #vivo l'oggetto. Non c'e' piu' nessuno a cui consegnare.
                pass

        threading.Thread(target=lavora, name="loss-csv", daemon=True).start()

    def _prendi_in_consegna(self, sorgente, generazione, avviso):
        if generazione != self._generazione:
            return      # una lettura vecchia, arrivata dopo quella che l'ha sostituita
        self._consegnata = generazione
        if sorgente.errore is None:
            #Quali punti siano vivi lo sa il pannello, non la sorgente di
            #prima: dedurlo interrogando quella significava reiniettare come
            #vivi anche le righe del suo CSV. Qui la sorgente nuova ha solo
            #il suo file, quindi ultima_iterazione() e' esattamente fin dove
            #il file arriva.
            self._dimentica_i_vivi(sorgente.ultima_iterazione())
            for iterazione, losses in self._vivi:
                sorgente.aggiungi_vivo(iterazione, losses)
            self.loss = sorgente
        #Anche vuoto: una lettura riuscita deve cancellare l'avviso di
        #quella fallita, o la riga di stato resta a dire una cosa vecchia.
        #Cancella il proprio, pero': l'esito del job e l'anteprima non
        #leggibile hanno il loro campo e non si toccano da qui.
        self._avviso_loss = avviso
        self._aggiorna_stato()
        self._aggiorna_grafico()

    def _dimentica_i_vivi(self, fino_a):
        """I punti che il CSV ha raggiunto: adesso li ha lui, non servono piu'.

        L'iterazione arriva dall'evento `save`, cioe' dal canale: storta
        vale come "non dichiarata". Un NaN qui e' il caso silenzioso di
        sempre -- `v[0] > nan` e' falso per ogni punto, quindi *tutti*
        verrebbero dimenticati, e sono la sola copia di cio' che la sorgente
        in arrivo non ha.
        """
        if iterazione_utilizzabile(fino_a):
            self._vivi = [v for v in self._vivi if v[0] > fino_a]

    def _dentro_la_cartella(self, file_):
        """Il percorso annunciato, o None se non e' nella cartella del job.

        E' l'unico punto del pannello che compone un percorso a partire da un
        dato che arriva da fuori. Il produttore e' il figlio che la GUI stessa
        ha lanciato, quindi non e' una superficie d'attacco -- e' proprio per
        questo che va chiuso qui e adesso, mentre il costo e' una riga: un
        `"file": "../fuori.png"` viene caricato da fuori la cartella senza
        che niente lo dica, e chi verificasse la cosa fra un anno partirebbe
        dal presupposto che qualcuno ci avesse pensato.
        """
        try:
            percorso = (self.previews_dir / file_).resolve()
            return percorso if percorso.parent == self.previews_dir.resolve() else None
        except (OSError, ValueError, TypeError):
            #Un `file` che non e' un pezzo di percorso (byte nulli, un tipo
            #sbagliato) vale come annuncio che non si puo' seguire.
            return None

    def _su_preview(self, event):
        """Le anteprime annunciate, tenendo le vecchie di quelle illeggibili.

        Un'anteprima che non si carica non deve cancellare quella che c'era,
        ne' far avanzare l'iterazione mostrata: il file annunciato puo'
        essere sparito sotto i piedi (cartella ripulita, corsa interrotta) e
        l'ultima immagine buona resta la cosa piu' vera che si possa
        mostrare.
        """
        immagini = dict(self._immagini)
        descrittori = dict(self._descrittori)
        arrivate = 0
        for voce in event.get("immagini") or []:
            nome, file_ = voce.get("nome"), voce.get("file")
            #Il nome e' una chiave e finisce in un `QComboBox`: un `5` la'
            #dentro fa saltare l'evento intero -- dopo che le immagini sono
            #gia' state scritte in memoria -- e lascia il selettore a
            #raccontare un elenco che non e' piu' quello vero.
            if not isinstance(nome, str) or not nome or not file_:
                continue
            percorso = self._dentro_la_cartella(file_)
            if percorso is None:
                continue
            immagine = QImage(str(percorso))
            if immagine.isNull():
                continue
            immagini[nome] = immagine
            descrittori[nome] = normalizza(voce)
            arrivate += 1
        if not arrivate:
            self._avviso_anteprima = testi.preview_unreadable()
            self._aggiorna_stato()
            return
        self._immagini, self._descrittori = immagini, descrittori
        self._nomi_file = event.get("nomi_file")
        #Anche qui, e per lo stesso motivo: l'iterazione annunciata diventa
        #quella mostrata, e quella mostrata finisce nel cursore e nel taglio
        #del grafico. Un valore inutilizzabile vale come "non annunciata".
        iterazione = event.get("iter")
        self._iterazione_viva = iterazione if iterazione_utilizzabile(iterazione) else None
        self._avviso_anteprima = ""
        if self.modo == "diretta":
            self.iterazione_mostrata = self._iterazione_viva
        self._sincronizza_selettore()
        self._aggiorna_cursore()
        self._aggiorna_stato()
        self._ridisegna()

    def _su_iter(self, event):
        iterazione = event.get("iter", 0)
        #Un'iterazione con cui non si puo' contare si ferma qui, prima di
        #entrare in qualunque memoria. Se entrasse, ogni evento buono che
        #viene dopo si romperebbe nel confrontarsi con lei -- la storia
        #della loss e il ritmo tengono l'ultimo punto, ed e' proprio quello
        #che un valore storto avvelena, per tutto il resto della corsa.
        if not iterazione_utilizzabile(iterazione):
            raise ValueError(testi.iteration_not_usable(iterazione))
        losses = event.get("losses") or []
        #Solo se la sorgente l'ha accettato. La lista qui e' la copia che
        #serve al travaso nella sorgente che la lettura del CSV consegnera',
        #e una copia che accetta cio' che l'originale rifiuta -- un punto
        #fuori ordine -- se lo riporta dentro al primo `hello`.
        if self.loss.aggiungi_vivo(iterazione, losses):
            self._vivi.append((iterazione, list(losses)))
            #Un punto in coda, non tutta la storia da capo. A settantacinquemila
            #iterazioni ripubblicarla costava 12,6 ms per evento, con gli
            #eventi che arrivano fino a due al secondo -- e cresceva con
            #l'allenamento. I valori li da' la sorgente, gia' passati dalla
            #sua regola su cosa e' disegnabile: rileggerli da `losses`
            #sarebbe una seconda regola libera di divergere.
            self.plot.aggiungi_punto(*self.loss.ultimo_punto())

        ritmo = self._ritmo.aggiorna(iterazione, time.monotonic())

        self._coppie_stato = valori_di_stato(
            iterazione, losses, ritmo, self._model["target_iter"],
            event.get("vram_usata_gib"), event.get("vram_totale_gib"))
        self._disegna_tessere()
        #Il progresso e' nelle tessere adesso, ma la riga di avvisi va
        #ricomposta lo stesso: e' lei a dire quanti valori di questa loss
        #non erano disegnabili (`_avviso_scarti`, contati dalla sorgente).
        self._aggiorna_stato()

    # -- lo stato che si legge da fuori ------------------------------------

    def anteprime_disponibili(self):
        return list(self._immagini)

    def descrittore_corrente(self):
        return self._descrittori.get(self.anteprima_selezionata)

    def immagine_intera(self):
        """L'immagine mostrata adesso, intera: dallo storico o dal vivo."""
        if self.modo == "storico" and self._immagine_storico is not None:
            return self._immagine_storico
        return self._immagini.get(self.anteprima_selezionata)

    def cella_risultato(self):
        """La cella nel riquadro grande, o None senza descrittore."""
        return self._cella_risultato

    def stato_testo(self):
        return self.stato.text()

    # -- i comandi dell'utente ---------------------------------------------

    def seleziona_campione(self, indice):
        self.campione_selezionato = max(0, min(int(indice), self._righe_correnti() - 1))
        self._ridisegna()

    def mostra_cella(self, colonna):
        """Porta una cella nel riquadro grande. Le altre restano di lato.

        Solo la colonna: le celle laterali stanno tutte nella riga di
        adesso, quindi promuoverne una significa solo cambiare colonna. La
        riga resta quella che era gia' -- campione_selezionato o il
        risultato dichiarato -- cosi' un click successivo su una miniatura
        continua a spostare il riquadro grande invece di restare congelato
        sulla riga di quando si e' promossa la colonna.

        Passa da `_ridisegna`, che ha gia' la sua rete: e' il motivo per cui
        questo metodo puo' essere chiamato da un click senza avere una rete
        propria.
        """
        self._colonna_mostrata = colonna
        self._ridisegna()

    def apri_a_dimensione_naturale(self, riga, colonna):
        """Uno scatto: apre una finestra con quella cella, cosi' com'e' ora.

        Non passa da `_ridisegna` -- non tocca i widget del pannello, ne
        costruisce uno nuovo che vive per conto suo -- quindi ha bisogno
        della propria rete: chiamato da un doppio click, cioe' da uno slot
        Qt senza rete propria, dove un'eccezione non risale a nessuno e
        porta via il processo con dentro ogni altro training aperto (la
        stessa classe di guasto della voce 3.23).
        """
        try:
            immagine = self._cella_a(riga, colonna)
            if immagine is None:
                return None
            titolo = testi.preview_window_title(
                etichetta(self.descrittore_corrente(), riga, colonna), self.iterazione_mostrata)
            finestra = FinestraImmagine(immagine, titolo, parent=self)
            finestra.show()
            return finestra
        except Exception as errore:
            self._avviso_disegno = testi.preview_not_drawable(
                type(errore).__name__, errore)
            self._aggiorna_stato()
            return None

    def _apri_grande(self):
        """Il doppio click sul riquadro grande: la cella che ha in mano ora."""
        if self.cella_mostrata is not None:
            self.apri_a_dimensione_naturale(*self.cella_mostrata)

    def _cella_a(self, riga, colonna):
        """La QImage di una cella qualunque della griglia di adesso, o None
        se l'immagine, il descrittore o la coordinata non ci sono."""
        immagine = self.immagine_intera()
        descrittore = self.descrittore_corrente()
        if immagine is None or descrittore is None:
            return None
        colonne = descrittore["colonne"]
        righe = righe_effettive(immagine, colonne)
        if not (0 <= riga < righe and 0 <= colonna < colonne):
            return None
        return celle(immagine, colonne, righe)[riga][colonna]

    def vai_a(self, iterazione):
        """Ferma il pannello a un'iterazione: immagine e curva insieme."""
        self.modo = "storico"
        self.iterazione_mostrata = iterazione
        self.plot.imposta_cursore(iterazione)
        self.etichetta_cursore.setText(testi.cursor_at_iteration(iterazione))
        self._disegna_tessere()
        if iterazione in self._iterazioni:
            self._muovi_cursore(self._iterazioni.index(iterazione))
        #Riavviabile: un secondo movimento prima della scadenza sostituisce
        #il caricamento in attesa invece di accodarne un altro, cosi' un
        #trascinamento non apre trecento file.
        self._timer_cursore.start(RITARDO_CURSORE_MS)

    def torna_in_diretta(self):
        self.modo = "diretta"
        self._timer_cursore.stop()
        self._immagine_storico = None
        self.iterazione_mostrata = self._iterazione_viva
        self.plot.imposta_cursore(None)
        self.etichetta_cursore.setText(testi.cursor_live())
        self._disegna_tessere()
        if self._iterazioni:
            self._muovi_cursore(len(self._iterazioni) - 1)
        self._ridisegna()

    def evento_non_applicato(self, errore):
        """Un evento che non si e' potuto applicare, detto a schermo.

        Lo chiamano in due, e non e' un caso: `applica_evento` quando e' lui
        stesso a non farcela, e chi instrada gli eventi quando a non farcela
        e' il pannello nel suo insieme. Entrambi girano da uno slot Qt, dove
        un'eccezione non risale a nessuno: PyQt5 chiama qFatal e il processo
        muore di colpo, portandosi via anche gli altri training vivi. Qui non
        si cattura niente: si mostra soltanto cosa non e' stato applicato, in
        un campo proprio come ogni altro produttore della riga di stato.
        """
        self._avviso_evento = testi.event_not_applied(type(errore).__name__, errore)
        self._aggiorna_stato()

    def job_finito(self, codice):
        """Il figlio non c'e' piu': via i comandi, il contenuto resta.

        La riga se ne va **intera**, separatori compresi, e Live con gli
        altri quattro: senza un figlio a cui parlare nessuno dei cinque ha
        piu' un interlocutore, e la via che Live apriva -- tornare
        all'ultimo fotogramma dopo aver guardato indietro -- la apre gia' il
        cursore portato in fondo (`_su_cursore`).

        Nascondere il contenitore invece dei figli uno per uno non e' una
        scorciatoia: e' la ragione per cui `self.comandi` esiste. I due
        separatori restavano a schermo -- due righe bianche in fondo a
        destra, col bottone superstite allargato a tutta la larghezza --
        proprio perche' questo elenco li nominava per nome e loro un nome
        non ce l'hanno.
        """
        self.comandi.setVisible(False)
        self._esito_job = testi.job_finished(codice)
        self._aggiorna_stato()

    # -- il resto --------------------------------------------------------

    def _su_cambio_anteprima(self, nome):
        self.anteprima_selezionata = nome or None
        self._immagine_storico = None
        self._aggiorna_cursore()
        if self.modo == "storico":
            self._timer_cursore.start(RITARDO_CURSORE_MS)
        self._ridisegna()

    def _sincronizza_selettore(self):
        nomi = self.anteprime_disponibili()
        if [self.selettore.itemText(i) for i in range(self.selettore.count())] == nomi:
            return
        precedente = self.anteprima_selezionata
        bloccato = self.selettore.blockSignals(True)
        self.selettore.clear()
        self.selettore.addItems(nomi)
        if precedente in nomi:
            self.selettore.setCurrentText(precedente)
        self.selettore.blockSignals(bloccato)
        self.anteprima_selezionata = self.selettore.currentText() or None

    def _sincronizza_selettore_campione(self, righe_sono_campioni, righe, riga):
        """Il selettore del campione, allineato alla griglia di adesso.

        Seconda via per lo stesso stato della striscia sotto, non un terzo
        stato: le voci si ricostruiscono solo quando il numero di campioni
        cambia, e l'indice corrente segue sempre `riga` -- con i segnali
        bloccati, cosi' un giro di sincronizzazione non ne apre un altro.
        E' lo stesso schema di `_sincronizza_selettore` qui sopra, per lo
        stesso motivo.
        """
        if not righe_sono_campioni:
            self.selettore_campione.setVisible(False)
            return
        self.selettore_campione.setVisible(True)
        etichette = [testi.sample_label(indice) for indice in range(righe)]
        bloccato = self.selettore_campione.blockSignals(True)
        attuali = [self.selettore_campione.itemText(i)
                  for i in range(self.selettore_campione.count())]
        if attuali != etichette:
            self.selettore_campione.clear()
            self.selettore_campione.addItems(etichette)
        self.selettore_campione.setCurrentIndex(riga)
        self.selettore_campione.blockSignals(bloccato)

    def _su_cursore(self, indice):
        if not self._iterazioni:
            return
        indice = max(0, min(indice, len(self._iterazioni) - 1))
        if indice == len(self._iterazioni) - 1:
            self.torna_in_diretta()     # il cursore in fondo torna a seguire
        else:
            self.vai_a(self._iterazioni[indice])

    def _muovi_cursore(self, indice):
        bloccato = self.cursore.blockSignals(True)
        self.cursore.setRange(0, max(0, len(self._iterazioni) - 1))
        self.cursore.setValue(indice)
        self.cursore.blockSignals(bloccato)

    def _aggiorna_cursore(self):
        """Il cursore copre gli scatti che lo storico ha davvero su disco.

        Riletti a ogni anteprima nuova: la cartella cresce mentre il
        pannello e' aperto.
        """
        nome = self.anteprima_selezionata
        if self.storico is not None and self.storico.disponibile():
            if nome is None or nome not in self.storico.anteprime():
                anteprime = self.storico.anteprime()
                nome = anteprime[0] if anteprime else None
            self._iterazioni = self.storico.iterazioni(nome) if nome else []
        else:
            self._iterazioni = []
        self.cursore.setEnabled(bool(self._iterazioni))
        self.cursore.setToolTip(CON_STORICO if self._iterazioni else SENZA_STORICO)
        if self._iterazioni and self.modo == "diretta":
            self._muovi_cursore(len(self._iterazioni) - 1)

    def _disegna_tessere(self):
        """Le tessere dell'ultimo evento `iter`, con l'iterazione di cio' che
        si sta davvero guardando.

        In diretta le due coincidono e non succede niente. Nello storico no:
        anteprima, curva e cursore sono fermi a un'iterazione, e la tessera
        continuava a scrivere quella viva -- il numero piu' grande dello
        schermo che dice la cosa sbagliata proprio mentre tutto il resto ne
        dice un'altra. Qui cambia **chiave**, non solo valore, cosi' la
        didascalia diventa "Iteration (history)" e `valori()` continua a
        raccontare cosa la tessera sta mostrando davvero.

        Gli altri numeri restano quelli della corsa viva ed e' corretto che
        lo siano: velocita', ETA e VRAM sono misure di adesso, non di
        allora, e lo storico non ne conserva nessuna.
        """
        coppie = self._coppie_stato
        if self.modo == "storico" and iterazione_utilizzabile(self.iterazione_mostrata):
            coppie = tuple(
                (testi.TILE_KEY_ITERATION_HISTORY, "%d" % self.iterazione_mostrata)
                if chiave == testi.TILE_KEY_ITERATION else (chiave, valore)
                for chiave, valore in coppie)
        self.tessere.aggiorna(coppie)

    def _aggiorna_grafico(self):
        """Ripubblica la storia intera: solo quando cambia davvero.

        Sono tre momenti e sono tutti rari -- un `hello`, il CSV riletto, il
        CSV riconsegnato dal lettore -- e da qui in poi il grafico si
        allunga da solo, un punto per evento.

        Il taglio allo storico non passa piu' di qui: era `punti(fino_a=...)`
        a ogni scatto del cursore, cioe' una copia dell'intera storia per
        ogni pixel di trascinamento, e disegnava **esattamente** la stessa
        curva -- il grafico non guarda comunque niente oltre la finestra, e
        la finestra si ferma al cursore. Adesso lo dice solo al cursore.
        """
        self.plot.imposta_dati(*self.loss.punti())

    def _carica_dallo_storico(self):
        """Lo scatto dell'iterazione ferma, o il piu' vicino che si legga."""
        if self.modo != "storico" or self.storico is None:
            return
        nome = self.anteprima_selezionata
        if nome is None:
            return
        immagine = self.storico.immagine(nome, self.iterazione_mostrata)
        if immagine is None:
            #Jpg troncato o sparito: si va allo scatto valido piu' vicino
            #invece di lasciare l'utente davanti a un buco.
            for iterazione in sorted(self._iterazioni,
                                     key=lambda v: abs(v - self.iterazione_mostrata)):
                immagine = self.storico.immagine(nome, iterazione)
                if immagine is not None:
                    self.iterazione_mostrata = iterazione
                    self.plot.imposta_cursore(iterazione)
                    self._aggiorna_grafico()
                    break
        if immagine is None:
            return
        self._immagine_storico = immagine
        self._ridisegna()

    def _aggiorna_stato(self):
        """La riga di avvisi sotto le tessere, composta dai soli produttori
        che restano testo: il progresso adesso vive in `self.tessere`.

        L'ordine e' fisso e i pezzi vuoti spariscono, cosi' un avviso che va
        e viene non sposta gli altri di posto: prima cosa non si e' potuto
        mostrare, poi cosa non si e' potuto leggere, poi quali valori non si
        sono potuti disegnare, poi quale immagine non si e' potuta
        disegnare, poi cosa non si e' potuto applicare, e in fondo l'esito
        -- l'ultima parola, quando c'e'.

        Il pezzo dei valori scartati non ha un campo suo: e' la sorgente a
        contarli, e la sorgente viene sostituita a ogni consegna del CSV.
        Chiederglielo al momento di scrivere e' l'unico modo perche' il
        numero racconti la curva che si sta guardando adesso.

        Nessun avviso da dire: la riga si nasconde, invece di restare a
        schermo vuota fra le tessere e il grafico.
        """
        pezzi = [self._avviso_anteprima, self._avviso_loss,
                self._avviso_scarti(), self._avviso_disegno,
                self._avviso_evento, self._esito_job]
        testo = " | ".join(p for p in pezzi if p)
        self.stato.setText(testo)
        self.stato.setVisible(bool(testo))

    def _avviso_scarti(self):
        """Quanti valori di loss non erano disegnabili, quando ce ne sono.

        Una curva che si interrompe senza spiegazione e' peggio di nessuna
        curva: senza questa riga il buco si confonde con un salvataggio
        saltato, e la causa vera -- un training che diverge -- resta
        invisibile proprio nel momento in cui va vista.
        """
        quanti = self.loss.scartati
        if not quanti:
            return ""
        return testi.loss_values_dropped(quanti)

    def _righe_correnti(self):
        immagine, descrittore = self.immagine_intera(), self.descrittore_corrente()
        if immagine is None or descrittore is None:
            return 1
        return righe_effettive(immagine, descrittore["colonne"])

    def _ridisegna(self):
        """Il disegno, con la rete che il gestore degli eventi ha gia'.

        Ci si arriva da tre slot che rete non ne hanno -- il cambio di
        anteprima dal selettore, il click su una miniatura, la scadenza del
        timer del cursore -- e da li' un'eccezione non risale a nessuno:
        PyQt5 chiama qFatal e la finestra se ne va con dentro ogni altro
        training aperto. La rete attorno al gestore degli eventi non copre
        niente di tutto questo: quando l'utente clicca, l'evento che ha
        portato il dato storto e' passato da un pezzo, ed e' stato *assorbito*
        -- ma cio' che ha lasciato in memoria e' ancora qui.
        """
        try:
            self._disegna()
        except Exception as errore:
            self._avviso_disegno = testi.preview_not_drawable(
                type(errore).__name__, errore)
            self._aggiorna_stato()

    def _disegna(self):
        """L'unico punto che tocca i widget delle immagini.

        Le righe le conta l'immagine, non il descrittore: uno scatto dello
        storico puo' venire da una corsa con un altro batch size, e quindi
        avere meno campioni di quanti il modello ne dichiari adesso.
        """
        self._avviso_disegno = ""
        _svuota(self.contorno)
        _svuota(self.striscia)
        self._cella_risultato = None
        immagine = self.immagine_intera()
        descrittore = self.descrittore_corrente()
        if immagine is None:
            self.risultato.pulisci()
            self.selettore_campione.setVisible(False)
            return
        risultato = descrittore["risultato"] if descrittore is not None else None
        if risultato is None:
            #Senza descrittore, o senza la cella che dichiara il risultato,
            #l'immagine intera: meglio nessuna etichetta che una sbagliata.
            self.risultato.imposta_immagine(immagine)
            self.risultato.setToolTip("\n".join(
                [testi.BIG_FRAME_WHOLE_TIP] + _righe_dei_nomi(self._nomi_file)))
            self.selettore_campione.setVisible(False)
            return
        colonne = descrittore["colonne"]
        righe = righe_effettive(immagine, colonne)
        #Con righe_sono_campioni la riga la sceglie l'utente; senza, la
        #dichiara il modello -- le righe sono viste diverse, non campioni.
        #Questo calcolo non cambia mai per via di una colonna promossa: e'
        #cosi' che scegliere un altro campione continua a spostare il
        #riquadro grande anche dopo un click su una cella laterale, invece
        #di restare congelato sulla riga di quando si e' cliccato.
        riga = self.campione_selezionato if descrittore["righe_sono_campioni"] else risultato[0]
        riga = max(0, min(riga, righe - 1))
        self._sincronizza_selettore_campione(descrittore["righe_sono_campioni"], righe, riga)
        griglia = celle(immagine, colonne, righe)

        #La colonna mostrata e' uno stato della vista, non del descrittore:
        #un click la promuove, ma non deve mai indicizzare fuori dalla
        #griglia di adesso -- un descrittore nuovo che non la contiene piu'
        #(meno colonne di quante ne avesse quello di prima) la riporta alla
        #colonna del risultato dichiarato, che e' sempre valida quanto lo e'
        #il descrittore stesso (normalizza() la valida contro le stesse
        #`colonne` usate qui, mai ricalcolate dall'immagine come le righe).
        colonna_grande = self._colonna_mostrata
        if colonna_grande is not None and not (0 <= colonna_grande < colonne):
            colonna_grande = None
            self._colonna_mostrata = None
        if colonna_grande is None:
            colonna_grande = risultato[1]

        self._cella_risultato = griglia[riga][colonna_grande]
        self.cella_mostrata = (riga, colonna_grande)
        self.risultato.imposta_immagine(self._cella_risultato)
        self.risultato.setToolTip("\n".join(
            [testi.big_frame_tip(etichetta(descrittore, riga, colonna_grande))]
            + _righe_dei_nomi(self._nomi_file)))
        for colonna in range(colonne):
            if colonna != colonna_grande:
                laterale = _CellaLaterale(griglia[riga][colonna], LATO_CONTORNO,
                                          etichetta(descrittore, riga, colonna),
                                          riga, colonna, self)
                self.contorno.addWidget(laterale)
        if descrittore["righe_sono_campioni"]:
            #La colonna mostrata, non risultato[1]: sotto al volto che si
            #sta guardando ci sono gli altri campioni della stessa cosa,
            #anche dopo aver promosso una colonna diversa da quella del
            #risultato dichiarato.
            for indice in range(righe):
                self.striscia.addWidget(self._miniatura(griglia[indice][colonna_grande],
                                                        indice, indice == riga,
                                                        indice, colonna_grande))
        else:
            for nome in self.anteprime_disponibili():
                self.striscia.addWidget(QLabel(nome))
        #Senza, le miniature (gia' a misura fissa) restano allineate a
        #sinistra ma lo spazio avanzato si spalma comunque nella riga
        #attraverso il layout stesso; con lo stretch in coda finisce tutto
        #alla fine della riga, dove non sposta niente.
        self.striscia.addStretch(1)

    def _miniatura(self, immagine, indice, scelto, riga, colonna):
        bottone = _BottoneMiniatura()
        bottone.setIcon(QIcon(QPixmap.fromImage(immagine)))
        bottone.setIconSize(QSize(LATO_MINIATURA, LATO_MINIATURA))
        bottone.setCheckable(True)
        bottone.setChecked(scelto)
        bottone.setToolTip(testi.thumbnail_tip(indice))
        bottone.clicked.connect(lambda _c, i=indice: self.seleziona_campione(i))
        bottone.doppio_click = lambda r=riga, c=colonna: self.apri_a_dimensione_naturale(r, c)
        return bottone
