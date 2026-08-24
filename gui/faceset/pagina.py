"""La pagina di cura del faceset: composizione, niente logica propria.

Segue il PROGETTO, non il job -- e' l'unica differenza strutturale
rispetto a TrainingPanel, che e' indicizzato per job. La scheda e' una
sola, e cambiare progetto cambia cio' che mostra.
"""
import tempfile
from pathlib import Path

from PyQt5.QtCore import QUrl, Qt, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QKeySequence
from PyQt5.QtWidgets import (QAction, QApplication, QHBoxLayout, QLabel, QMenu,
                             QMessageBox, QPushButton, QShortcut, QSlider,
                             QToolButton, QVBoxLayout, QWidget)

from gui import testi
from gui import theme
from gui.catalog import step_by_name
from gui.execution.jobs import StepConflict
from gui.faceset import azioni as azioni_mod
from gui.faceset import cache as cache_mod
from gui.faceset import cestino as cestino_mod
from gui.faceset import conflitti as conflitti_mod
from gui.faceset import fratelli as fratelli_mod
from gui.faceset import indice as indice_mod
from gui.faceset.decodifica import LATI
from gui.faceset.griglia import MODI_MASCHERA, Griglia
from gui.faceset.heatmap import BIN_AMMESSI, WidgetHeatmap, bin_di_percorsi
from gui.faceset.modello import RUOLO_PERCORSO, ModelloVolti
from gui.faceset.progresso import PilaProgresso
from gui.faceset.tacche import StrisciaTacche

ESTENSIONI = (".jpg", ".jpeg", ".png")

# L'unica operazione con un tasto proprio invece che una voce di Tools: e'
# quella che l'utente lancia venti volte per faceset. Una costante e non un
# letterale ripetuto perche' e' una CHIAVE di dispatch, non un testo -- il
# testo e' `Operazione.etichetta`, e nasce in gui/faceset/azioni.py.
CHIAVE_SORT = "sort"


def _operazione(chiave):
    for op in azioni_mod.OPERAZIONI:
        if op.chiave == chiave:
            return op
    raise KeyError(chiave)


def _predefinita(cartelle):
    """La cartella su cui aprire un dataset: `aligned` quando c'e'.

    R23: `cartelle()` torna `[radice] + sottocartelle`, e aprire sulla
    radice significa aprire sui fotogrammi grezzi con ogni operazione di
    cura giustamente grigia -- la prima impressione sbagliata su una
    pagina che di cura si occupa. La radice resta il ripiego, per un
    dataset non ancora estratto.
    """
    for c in cartelle:
        if c.name == "aligned":
            return c
    return cartelle[0]


def _ha_il_passo(operazione, dataset):
    """Se questa operazione esiste per questo dataset.

    R24: si decide sui DATI (`Operazione.passo_dst`), non su un `None` di
    `passo_per` -- che non lo produce mai, perche' ricade su `passo_src`.
    Quel ripiego e' giusto per chi chiama `passo_per` sapendo cosa chiede;
    qui sarebbe un job che dichiara `modifies=('faceset_src',)` mentre
    riscrive `data_dst/aligned`, cioe' la contesa silenziosa che questa
    pagina esiste per chiudere.
    """
    return not (dataset == "dst" and not operazione.passo_dst)


class PaginaCuraFaceset(QWidget):
    richiesta_frame_originale = pyqtSignal(str, str)

    def __init__(self, radice_e, impostazioni, parent=None):
        super().__init__(parent)
        self._radice_e = Path(radice_e)
        self._workspace = None
        self._dataset = "src"
        self._cartella = None
        self._indice = indice_mod.Indice([])
        self._abbinati = {}
        self._mappa_frame = {}
        self._fratelli = []
        self._stato = indice_mod.STATO_ASSENTE
        self._cartella_dettaglio = None
        self._cliente = None
        self._finestra_dettaglio = None
        self._gestore = None
        self._ultima_mossa = None
        self._job_corrente = None
        self._nome_job_corrente = ""
        self._bin_accesi = set()
        # Mutuamente esclusivo con `_bin_accesi`: un filtro alla volta, una
        # pastiglia sola da leggere. Vive sul NOME DEL FRAME e non su un
        # insieme di percorsi -- un sort rinomina i file, e un filtro fatto
        # di percorsi si svuoterebbe (la stessa trappola gia' pagata dal
        # filtro della heatmap, vedi `_riapplica_filtro`).
        self._frame_filtrato = None

        self.modello = ModelloVolti()
        self.griglia = Griglia()
        self.griglia.setModel(self.modello)
        self.heatmap = WidgetHeatmap(impostazioni)
        self.pila = PilaProgresso()
        self.etichetta_stato = QLabel("")
        self.selettore_cartella = theme.tendina()
        self.bottone_src = QPushButton(testi.FACESET_SRC)
        self.bottone_src.setCheckable(True)
        self.bottone_dst = QPushButton(testi.FACESET_DST)
        self.bottone_dst.setCheckable(True)
        self.cursore_zoom = QSlider(Qt.Horizontal)
        self.cursore_zoom.setRange(0, len(LATI) - 1)
        self.cursore_zoom.setValue(LATI.index(self.griglia.lato()))
        # Una larghezza dichiarata, non quel che avanza: in una barra sola
        # il cursore era l'unico elastico fra dieci widget e a scala
        # `xlarge` si riduceva a ~50 px, misurato su uno scatto.
        # Il massimo c'e' perche' l'estremo opposto e' altrettanto
        # sbagliato: un cursore lungo mezza finestra per tre soli valori
        # promette una precisione che non ha.
        self.cursore_zoom.setMinimumWidth(160)
        self.cursore_zoom.setMaximumWidth(280)
        # L'unico controllo della barra senza nessuna parola addosso: da
        # solo su una riga dice ancora meno di quando stava in mezzo agli
        # altri.
        self.cursore_zoom.setToolTip(testi.FACESET_SIZE_TIP)
        self.selettore_maschera = theme.tendina()
        for chiave, etichetta in MODI_MASCHERA:
            self.selettore_maschera.addItem(etichetta, chiave)
        self.bottone_apri_cartella = QPushButton(testi.FACESET_OPEN_FOLDER)
        self.bottone_apri_cartella.setToolTip(testi.FACESET_OPEN_FOLDER_TIP)
        self.bottone_sort = QPushButton(_operazione(CHIAVE_SORT).etichetta)
        self.bottone_strumenti = QPushButton(testi.FACESET_TOOLS)
        self.bottone_strumenti.setToolTip(testi.FACESET_TOOLS_TIP)
        self.menu_strumenti = QMenu(self.bottone_strumenti)
        self.bottone_strumenti.setMenu(self.menu_strumenti)
        self.bottone_cancella = QPushButton(testi.FACESET_DELETE)
        self.bottone_cancella.setToolTip(testi.FACESET_DELETE_TIP)
        self.bottone_annulla = QPushButton(testi.FACESET_UNDO_DELETE)
        self.bottone_annulla.setToolTip(testi.FACESET_UNDO_DELETE_TIP)
        self.bottone_indice = QPushButton(testi.FACESET_INDEX_NOW)
        # La heatmap nuda non dice cosa sia ne' quanto valga un colore: la
        # scala e' logaritmica, e un colore senza numeri mente su quanto
        # vale. La pastiglia c'e' solo col filtro acceso -- «3 di 3» a
        # filtro spento sarebbe rumore, e a filtro acceso e' l'unica riga
        # che dice che la griglia sotto e' una fetta.
        # Il titolo E' il comando («▾ Pose
        # distribution»): un triangolo separato accanto a un'etichetta
        # sarebbe lo stesso disegno con un bersaglio da cliccare dieci
        # volte piu' piccolo.
        self.bottone_collassa = QToolButton()
        self.bottone_collassa.setText(testi.HEATMAP_TITLE)
        self.bottone_collassa.setToolTip(testi.HEATMAP_COLLAPSE_TIP)
        self.bottone_collassa.setCheckable(True)
        self.bottone_collassa.setAutoRaise(True)
        self.bottone_collassa.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.bottone_collassa.setArrowType(Qt.DownArrow)
        self.bottone_collassa.setProperty("ruolo", "sezione")
        self.selettore_bin = theme.tendina()
        self.selettore_bin.setToolTip(testi.HEATMAP_BINS_TIP)
        for n in BIN_AMMESSI:
            self.selettore_bin.addItem(testi.heatmap_bins_label(n), n)
        # Prima di agganciare il segnale: il valore ricordato e' gia' dentro
        # la heatmap, e rimandarglielo di qui spegnerebbe un filtro che
        # nessuno ha ancora acceso.
        self.selettore_bin.setCurrentIndex(BIN_AMMESSI.index(self.heatmap.bins()))
        self.legenda_heatmap = QLabel("")
        self.legenda_heatmap.setProperty("ruolo", "minore")
        self.pastiglia_filtro = QLabel("")
        self.pastiglia_filtro.setProperty("ruolo", "pastiglia")
        # Il `[Clear]` sta DENTRO la pastiglia, non accanto:
        # e' il posto in cui l'utente sta gia' guardando quando si chiede
        # perche' la griglia mostri meno volti di quanti la cartella ne
        # abbia. Un collegamento e non un bottone perche' la pastiglia e'
        # una riga di testo, e un bottone accanto sarebbe un secondo
        # elemento in barra da spiegare.
        self.pastiglia_filtro.linkActivated.connect(lambda _href: self.pulisci_filtro())
        self.pastiglia_filtro.hide()

        # La fascia dei fratelli compare solo quando ce ne sono: una riga
        # vuota sopra ogni griglia sarebbe una promessa che non mantiene.
        self.bottone_fratello_prec = QToolButton()
        self.bottone_fratello_prec.setText(testi.FACESET_SIBLING_PREV)
        self.bottone_fratello_prec.setToolTip(testi.FACESET_SIBLING_TIP)
        self.bottone_fratello_succ = QToolButton()
        self.bottone_fratello_succ.setText(testi.FACESET_SIBLING_NEXT)
        self.bottone_fratello_succ.setToolTip(testi.FACESET_SIBLING_TIP)
        self.etichetta_fratelli = QLabel("")
        self.etichetta_fratelli.setProperty("ruolo", "minore")
        self.fascia_fratelli = QWidget()
        barra_fratelli = QHBoxLayout(self.fascia_fratelli)
        barra_fratelli.setContentsMargins(0, 0, 0, 0)
        for w in (self.bottone_fratello_prec, self.etichetta_fratelli,
                  self.bottone_fratello_succ):
            barra_fratelli.addWidget(w)
        barra_fratelli.addStretch(1)
        self.fascia_fratelli.hide()
        self.striscia = StrisciaTacche()

        # Due righe: cio' che SCEGLIE e cio' che
        # AGISCE sopra, cio' che regola la VISTA sotto. Dieci widget su una
        # riga sola lasciavano al cursore dello zoom quel che avanzava.
        barra = QHBoxLayout()
        for w in (self.bottone_src, self.bottone_dst, self.selettore_cartella,
                  self.bottone_apri_cartella, self.bottone_sort,
                  self.bottone_strumenti, self.bottone_cancella,
                  self.bottone_annulla):
            barra.addWidget(w)
        barra.addStretch(1)
        barra_vista = QHBoxLayout()
        barra_vista.addWidget(self.cursore_zoom)
        barra_vista.addWidget(self.selettore_maschera)
        barra_vista.addStretch(1)
        fascia_indice = QHBoxLayout()
        fascia_indice.addWidget(self.etichetta_stato, 1)
        fascia_indice.addWidget(self.bottone_indice)
        barra_heatmap = QHBoxLayout()
        barra_heatmap.addWidget(self.bottone_collassa)
        barra_heatmap.addWidget(self.selettore_bin)
        barra_heatmap.addWidget(self.legenda_heatmap)
        barra_heatmap.addWidget(self.pastiglia_filtro)
        barra_heatmap.addStretch(1)
        centro = QHBoxLayout()
        centro.addWidget(self.griglia, 1)
        centro.addWidget(self.striscia)
        radice = QVBoxLayout(self)
        radice.addLayout(barra)
        radice.addLayout(barra_vista)
        radice.addLayout(barra_heatmap)
        radice.addWidget(self.heatmap)
        radice.addWidget(self.fascia_fratelli)
        radice.addLayout(centro, 1)
        radice.addWidget(self.pila)
        radice.addLayout(fascia_indice)

        self.bottone_src.clicked.connect(lambda: self.imposta_dataset("src"))
        self.bottone_dst.clicked.connect(lambda: self.imposta_dataset("dst"))
        self.selettore_cartella.activated.connect(self._su_cartella_scelta)
        self.heatmap.selezione_bin_cambiata.connect(self._su_filtro_bin)
        self.cursore_zoom.valueChanged.connect(
            lambda i: self.griglia.imposta_lato(LATI[i]))
        # currentIndexChanged e non activated: quando la cartella corrente
        # non ha maschere il controllo torna su «off» da solo, e quel
        # ritorno deve raggiungere la griglia esattamente come il click
        # dell'utente -- se no la griglia resta in «sola maschera» sotto un
        # controllo grigio che dice «off».
        self.selettore_maschera.currentIndexChanged.connect(
            self._su_modo_maschera)
        self.griglia.volto_aperto.connect(self._su_volto_aperto)
        self.griglia.corrente_cambiato.connect(self._su_corrente_cambiato)
        self.griglia.menu_richiesto.connect(self._su_menu_griglia)
        self.bottone_fratello_prec.clicked.connect(lambda: self.vai_al_fratello(-1))
        self.bottone_fratello_succ.clicked.connect(lambda: self.vai_al_fratello(1))
        self.striscia.riga_scelta.connect(self._su_tacca_scelta)
        self.griglia.verticalScrollBar().valueChanged.connect(
            lambda _v: self._aggiorna_banda())
        self.bottone_apri_cartella.clicked.connect(self.apri_nel_gestore_file)
        self.bottone_sort.clicked.connect(lambda: self.avvia_operazione(CHIAVE_SORT))
        self.bottone_cancella.clicked.connect(self.cancella_selezione)
        self.bottone_annulla.clicked.connect(self.annulla_ultima_cancellazione)
        self.bottone_indice.clicked.connect(self.avvia_indicizzazione)
        # I due tasti soliti, sulla pagina e non sulla griglia: la
        # selezione e' della griglia, ma la decisione se cancellare e' della
        # pagina -- e' lei a sapere se un job tiene la cartella.
        QShortcut(QKeySequence.Delete, self, self.cancella_selezione)
        QShortcut(QKeySequence.Undo, self, self.annulla_ultima_cancellazione)
        self.bottone_collassa.toggled.connect(self._su_collasso)
        self.selettore_bin.currentIndexChanged.connect(self._su_bin_scelti)
        # Lo stato ricordato va APPLICATO, non solo letto: il widget lo
        # tiene dalla sessione scorsa, ma finche' nessuno chiama
        # `imposta_collassata` la heatmap resta visibile e il bottone dice
        # «espansa» sopra una heatmap espansa che l'utente aveva chiuso.
        self.bottone_collassa.setChecked(self.heatmap.collassata())
        self._su_collasso(self.heatmap.collassata())
        self._rigenera_comandi()

    # -- workspace e cartelle ------------------------------------------------

    def imposta_workspace(self, workspace):
        self._workspace = Path(workspace)
        self.imposta_dataset(self._dataset)

    def dataset(self):
        return self._dataset

    def imposta_dataset(self, dataset):
        self._dataset = dataset
        self._aggiorna_dataset_bottoni()
        cartelle = self.cartelle()
        self.selettore_cartella.clear()
        for c in cartelle:
            self.selettore_cartella.addItem(c.name, c)
        if cartelle:
            self.imposta_cartella(_predefinita(cartelle))
        else:
            # Senza questo, heatmap, riga di
            # stato e barra restano con i dati del dataset precedente --
            # cioe' mentono su cio' che la griglia, ormai vuota, mostra.
            self._cartella = None
            self._ultima_mossa = None
            self._abbinati = {}
            self._fratelli = []
            self._mappa_frame = {}
            self._frame_filtrato = None
            self.griglia.imposta_fratelli([])
            self._indice = indice_mod.Indice([])
            self._stato = indice_mod.STATO_ASSENTE
            self.modello.imposta([], {})
            self.heatmap.pulisci_selezione()
            self.heatmap.aggiorna([])
            self._aggiorna_fascia_heatmap()
            # Senza questa riga la fascia dei fratelli resterebbe accesa col
            # gruppo del dataset precedente: `_abbinati` e' gia' vuoto, ma
            # nessuno lo aveva ancora detto alla fascia.
            self._aggiorna_fascia_fratelli()
            self.etichetta_stato.setText(testi.faceset_index_state(self._stato, 0))
            self._rigenera_comandi()

    def _aggiorna_dataset_bottoni(self):
        """Lo stesso mestiere di PaginaEstrazione._aggiorna_lato_bottoni.
        Chiamato anche quando il dataset NON cambia: il bottone si e' gia'
        scommutato da solo al click, e senza questa chiamata la barra
        resterebbe senza nessun lato acceso."""
        self.bottone_src.setChecked(self._dataset == "src")
        self.bottone_dst.setChecked(self._dataset == "dst")

    def cartelle(self):
        if self._workspace is None:
            return []
        radice = self._workspace / ("data_%s" % self._dataset)
        try:
            voci = sorted(p for p in radice.iterdir() if p.is_dir())
        except OSError:
            return []
        return [radice] + voci if radice.exists() else []

    def cartella(self):
        return self._cartella

    def imposta_cartella(self, cartella):
        self._cartella = Path(cartella)
        # Il cestino appartiene alla CARTELLA, non alla pagina: i percorsi
        # della Mossa sono assoluti, quindi un Undo premuto dopo un cambio
        # riporterebbe davvero un file dell'altra cartella -- fuori dallo
        # schermo che si sta guardando. Nessun dato si perde (cestino.annulla
        # salta se l'origine esiste gia'), ma il bottone mentirebbe su cosa
        # sta per annullare. Stessa riga, e stessa ragione, di
        # PaginaEstrazione.apri con `_ultima_mossa_debug`.
        self._ultima_mossa = None
        self._fratelli = []
        self._mappa_frame = {}
        self._frame_filtrato = None
        self.griglia.imposta_fratelli([])
        self._sincronizza_selettore()
        # La cache della decodifica e' indicizzata per percorso: le voci
        # della cartella di prima non saranno mai piu' richieste, e senza
        # questa riga restavano dentro fino alla chiusura della pagina --
        # fino ai 64 MiB del tetto, tolti all'unica cosa che serve davvero,
        # cioe' la cartella che si sta guardando adesso.
        self.griglia.decodificatore.svuota()
        # Il filtro della heatmap e' un insieme di PERCORSI, e i percorsi
        # sono quelli della cartella su cui e' stato acceso: senza questa
        # riga nessun file della cartella nuova ci cade dentro, la griglia
        # resta vuota, e i bin accesi -- l'unica cosa a schermo che spiega
        # una griglia filtrata -- si riferiscono ormai a un'altra
        # cartella. Si guardava una cartella piena e si vedeva il vuoto.
        self.heatmap.pulisci_selezione()
        self.ricalcola()

    def _sincronizza_selettore(self):
        """Il selettore deve NOMINARE la cartella mostrata.

        Senza questo, `_predefinita` apriva su `aligned` (R23) mentre il
        menu restava sulla prima voce, la radice del dataset: la griglia
        mostrava i volti di `aligned` sotto la scritta «data_src», e ogni
        operazione lanciata da li' sembrava puntare la cartella
        sbagliata. Un solo posto lo fa, quello in cui la cartella cambia
        davvero -- il segnale agganciato e' `activated`, che scatta solo
        per il click dell'utente, quindi non c'e' ricorsione.
        """
        for i in range(self.selettore_cartella.count()):
            if self.selettore_cartella.itemData(i) == self._cartella:
                self.selettore_cartella.setCurrentIndex(i)
                return

    def _su_cartella_scelta(self, indice_voce):
        cartella = self.selettore_cartella.itemData(indice_voce)
        if cartella is not None:
            self.imposta_cartella(cartella)

    # -- dati ----------------------------------------------------------------

    def _elenco(self):
        """(percorso, dimensione, mtime_ns) per volto, UNA lettura sola.

        La cartella si legge una volta e da quella lettura escono sia i
        percorsi da mostrare sia i campi della chiave d'indice: prima erano
        due passate di `stat()` per file, una per filtrare e una per
        riconciliare, e su drvfs quelle due passate erano l'intera attesa
        all'apertura.
        """
        return indice_mod.elenca(self._cartella, ESTENSIONI)

    def ricalcola(self):
        letti = self._elenco()
        percorsi = [p for p, _dimensione, _mtime in letti]
        cartella_cache = cache_mod.percorso_cache(self._radice_e, self._cartella)
        self._indice = indice_mod.Indice(indice_mod.leggi(cartella_cache))
        self._abbinati, mancanti = self._indice.abbina_letti(letti)
        self._mappa_frame = fratelli_mod.mappa_per_frame(self._abbinati)
        self._stato = indice_mod.stato(self._abbinati, mancanti)
        self.modello.imposta(percorsi, self._abbinati)
        self.griglia.imposta_fornitore_maschere(
            lambda voce: indice_mod.maschera(cartella_cache, voce))
        self.heatmap.aggiorna(list(self._abbinati.values()))
        self._riapplica_filtro()
        self._aggiorna_fascia_heatmap()
        self.etichetta_stato.setText(
            testi.faceset_index_state(self._stato, len(mancanti)))
        # Come per il filtro della heatmap: se la cartella cambia mentre la
        # finestra di dettaglio e' aperta, le frecce non devono continuare
        # a navigare un elenco di una cartella che la griglia non elenca
        # piu'.
        if self._finestra_dettaglio is not None:
            self._finestra_dettaglio.imposta_ordine(self.modello.percorsi_visibili())
            # Senza cartella non c'e' nemmeno una genitrice: si passa None,
            # e la finestra si mette in sola lettura da se'. Sollevare qui
            # costerebbe l'intero processo, con dentro ogni training aperto.
            self._finestra_dettaglio.imposta_frame_dir(
                None if self._cartella is None else self._cartella.parent)
        # Cambiare cartella cambia cosa e' applicabile, e finire un job
        # cambia cosa e' libero: senza questa riga un'azione resta grigia
        # dopo che il job che teneva la cartella e' finito.
        self._rigenera_comandi()
        self._aggiorna_fratelli()

    def ricarica(self):
        self.ricalcola()

    def stato_indice(self):
        return self._stato

    def maschere_disponibili(self):
        """Se l'indice di QUESTA cartella porta almeno una maschera.

        Si guarda l'indice, non il disco: la griglia disegna le maschere
        dal blob della cache, quindi una cartella piena di volti con XSeg
        ma non ancora indicizzata non ha niente da mostrare -- e il
        controllo deve dirlo invece di accendere una vista vuota.
        """
        return any(v.mask_len for v in self._abbinati.values())

    def _su_modo_maschera(self, indice_voce):
        modo = self.selettore_maschera.itemData(indice_voce)
        if modo is not None:
            self.griglia.imposta_modo_maschera(modo)

    def _aggiorna_fascia_heatmap(self):
        """Legenda e pastiglia: i due testi che dicono cosa vale un colore
        e quanto della cartella si sta guardando."""
        minimo, massimo = self.heatmap.estremi()
        senza_posa = self.heatmap.senza_posa()
        self.legenda_heatmap.setText(
            testi.heatmap_legend(minimo, massimo, senza_posa))
        # Niente scala, niente legenda: su una cartella non indicizzata
        # «0 to 0 faces per cell» e' vero e non dice niente, e la riga di
        # stato in fondo gia' spiega perche' la heatmap e' vuota.
        # Collassata, la legenda descrive una mappa che non si vede.
        self.legenda_heatmap.setVisible(bool(massimo or senza_posa)
                                        and not self.heatmap.collassata())
        if self._frame_filtrato is not None:
            self.pastiglia_filtro.setText(testi.faceset_frame_filter_pill_html(
                self.modello.rowCount(), self.modello.totali(),
                self._frame_filtrato))
            self.pastiglia_filtro.show()
        elif self._bin_accesi:
            self.pastiglia_filtro.setText(testi.heatmap_filter_pill_html(
                self.modello.rowCount(), self.modello.totali(),
                len(self._bin_accesi)))
            self.pastiglia_filtro.show()
        else:
            self.pastiglia_filtro.hide()

    def _riapplica_filtro(self):
        """I bin accesi restano, i percorsi si rifanno.

        `ricalcola()` gira anche a fine job, e un sort RINOMINA i file: il
        filtro e' un insieme di percorsi, e dopo la rinomina non ne
        contiene piu' nessuno di quelli sul disco -- la griglia si
        svuoterebbe come al cambio di cartella, misurato `0 == 3`. Il
        filtro vive sui BIN, che al sort sopravvivono; i percorsi sono solo
        la forma in cui il modello lo consuma.

        Al cambio di cartella o di dataset i bin NON devono sopravvivere, e
        infatti non e' qui che si decide: li spegne `pulisci_selezione()`
        prima del ricalcolo, perche' si riferivano a un'altra
        distribuzione e riapplicarli sarebbe indovinare.

        Col filtro frame acceso i percorsi vengono dalla mappa dei frame,
        per la stessa ragione: e' il nome del frame a sopravvivere al sort.
        """
        if self._frame_filtrato is not None:
            self.modello.imposta_filtro(
                fratelli_mod.percorsi_del_frame(self._mappa_frame, self._frame_filtrato))
            return
        scelti = bin_di_percorsi(self._abbinati, self._bin_accesi,
                                 self.heatmap.bins())
        self.modello.imposta_filtro(scelti)

    def _su_filtro_bin(self, accesi):
        # Un filtro alla volta: accendere un bin esce dal filtro frame.
        if accesi:
            self._frame_filtrato = None
        self._bin_accesi = set(accesi)
        self._riapplica_filtro()
        self._aggiorna_fascia_heatmap()
        # Senza questa riga la fascia dei fratelli resta quella di PRIMA del
        # filtro -- gli altri due cammini (`filtra_per_frame`,
        # `pulisci_filtro`) la chiamano gia'; un bin acceso e' il terzo
        # modo di cambiare cio' che la griglia mostra, e deve aggiornarla
        # come loro.
        self._aggiorna_fratelli()
        # L'ordine mostrato dalla finestra di dettaglio e' quello filtrato
        # della griglia: se il filtro cambia mentre la finestra e' aperta,
        # la navigazione non deve restare legata a un elenco superato.
        if self._finestra_dettaglio is not None:
            self._finestra_dettaglio.imposta_ordine(self.modello.percorsi_visibili())

    def _su_collasso(self, collassata):
        """Il comando del titolo. La pastiglia resta: e' l'unica cosa a
        schermo che dice che la griglia sotto e' una fetta, e serve
        soprattutto quando la mappa coi bin accesi non si vede."""
        self.heatmap.imposta_collassata(collassata)
        self.bottone_collassa.setArrowType(
            Qt.RightArrow if collassata else Qt.DownArrow)
        self.selettore_bin.setVisible(not collassata)
        self._aggiorna_fascia_heatmap()

    def _su_bin_scelti(self, indice_voce):
        n = self.selettore_bin.itemData(indice_voce)
        if n is not None:
            self.heatmap.imposta_bins(n)

    # -- il filtro «stesso frame» ---------------------------------------------

    def frame_filtrato(self):
        return self._frame_filtrato

    def filtra_per_frame(self, nome_frame):
        """Mostra i soli volti di quel frame. Torna False -- senza toccare
        niente -- se il frame non ha volti in questa cartella: una griglia
        vuota si legge come una cartella vuota, e sarebbe la risposta
        sbagliata a un comando che ha appena promesso dei volti."""
        if not fratelli_mod.percorsi_del_frame(self._mappa_frame, nome_frame):
            return False
        # Prima i bin, poi il frame: `pulisci_selezione` fa scattare
        # `_su_filtro_bin`, che azzererebbe il frame appena impostato.
        self.heatmap.pulisci_selezione()
        self._bin_accesi = set()
        self._frame_filtrato = nome_frame
        self._riapplica_filtro()
        self._aggiorna_fascia_heatmap()
        self._aggiorna_fratelli()
        if self._finestra_dettaglio is not None:
            self._finestra_dettaglio.imposta_ordine(self.modello.percorsi_visibili())
        return True

    def pulisci_filtro(self):
        """Il `[Clear]` della pastiglia: spegne quello dei due che e'
        acceso, senza che chi lo preme debba sapere quale."""
        self._frame_filtrato = None
        self.heatmap.pulisci_selezione()
        self._riapplica_filtro()
        self._aggiorna_fascia_heatmap()
        self._aggiorna_fratelli()

    # -- il menu del tasto destro ---------------------------------------------

    def costruisci_menu_volto(self, percorso):
        """Il menu del tasto destro su un volto, o None sullo spazio vuoto.

        Costruito e restituito invece di essere aperto qui dentro: un menu
        che si apre da solo si prova soltanto simulando un clic, e le due
        voci hanno da dire piu' di quanto un clic possa verificare.
        """
        if percorso is None:
            return None
        nome = fratelli_mod.nome_frame_di(self._abbinati, percorso)
        menu = QMenu(self)
        # Senza questa riga Qt non disegna MAI il suggerimento delle voci
        # disabilitate (il default e' False): le due voci spente
        # comparivano senza dire perche', il dato di `toolTip()` restava
        # scritto ma invisibile.
        menu.setToolTipsVisible(True)
        stesso = QAction(testi.FACESET_MENU_SAME_FRAME, menu)
        stesso.setToolTip(testi.FACESET_MENU_SAME_FRAME_TIP if nome
                          else testi.FACESET_NO_INDEX_FOR_SIBLINGS)
        stesso.setEnabled(nome is not None)
        stesso.triggered.connect(lambda _c=False, n=nome: self.filtra_per_frame(n))
        originale = QAction(testi.FACESET_MENU_ORIGINAL_FRAME, menu)
        originale.setToolTip(testi.FACESET_MENU_ORIGINAL_FRAME_TIP if nome
                             else testi.FACESET_NO_INDEX_FOR_SIBLINGS)
        originale.setEnabled(nome is not None)
        originale.triggered.connect(
            lambda _c=False, n=nome: self.richiesta_frame_originale.emit(self._dataset, n))
        menu.addAction(stesso)
        menu.addAction(originale)
        return menu

    def _su_menu_griglia(self, percorso, punto):
        menu = self.costruisci_menu_volto(percorso)
        if menu is not None:
            menu.exec_(punto)

    def mostra_messaggio(self, testo):
        """La riga di stato della pagina, scritta da fuori: e' dove finisce
        il motivo per cui una navigazione incrociata si e' fermata, e deve
        stare sulla pagina che l'utente sta guardando."""
        self.etichetta_stato.setText(testo)

    def _frame_presente_in(self, cartella, nome_frame):
        """Sonda se `cartella` porta volti di `nome_frame`, SENZA renderla
        corrente: le stesse primitive di `ricalcola()` (`indice_mod.elenca`,
        la cache, `abbina_letti`, `mappa_per_frame`), usate su una lettura a
        parte invece che su `self._cartella`. Regge una cartella che non si
        elenca e una cache assente: entrambe tornano gia' liste vuote, senza
        sollevare.
        """
        letti = indice_mod.elenca(cartella, ESTENSIONI)
        cartella_cache = cache_mod.percorso_cache(self._radice_e, cartella)
        indice = indice_mod.Indice(indice_mod.leggi(cartella_cache))
        abbinati, _mancanti = indice.abbina_letti(letti)
        mappa = fratelli_mod.mappa_per_frame(abbinati)
        return bool(fratelli_mod.percorsi_del_frame(mappa, nome_frame))

    def mostra_solo_frame(self, lato, nome_frame):
        """L'ingresso dalla pagina di estrazione: porta il dataset su
        `lato`, la cartella su data_<lato>/aligned e filtra su quel frame.

        Torna False se non c'e' niente da mostrare -- cartella inesistente,
        o nessun volto di quel frame nella cartella bersaglio -- SENZA
        cambiare stato: la sonda (`_frame_presente_in`) legge la cartella
        bersaglio PRIMA di spostare dataset o cartella, cosi' un rifiuto non
        lascia la pagina spostata su un'altra cartella con la griglia non
        filtrata -- proprio quello che chi chiama deve poter escludere.
        """
        if self._workspace is None:
            return False
        cartella = self._workspace / ("data_%s" % lato) / "aligned"
        if not cartella.is_dir():
            return False
        if not self._frame_presente_in(cartella, nome_frame):
            return False
        if lato != self._dataset:
            self.imposta_dataset(lato)
        if self._cartella != cartella:
            self.imposta_cartella(cartella)
        return self.filtra_per_frame(nome_frame)

    # -- i fratelli dello stesso frame ---------------------------------------

    def fratelli(self):
        """I volti evidenziati ora: gli ALTRI dello stesso frame."""
        return list(self._fratelli)

    def _volto_corrente(self):
        index = self.griglia.currentIndex()
        return index.data(RUOLO_PERCORSO) if index.isValid() else None

    def _su_corrente_cambiato(self, _percorso):
        self._aggiorna_fratelli()

    def _aggiorna_fratelli(self):
        """Chi e' fratello di chi si ricalcola a ogni cambio di volto
        corrente, di cartella e di filtro: e' un giro di dizionario, non una
        passata su disco, e tenerlo aggiornato costa meno che spiegare
        perche' e' vecchio."""
        percorso = self._volto_corrente()
        self._fratelli = fratelli_mod.fratelli_di(
            self._mappa_frame, self._abbinati, percorso) if percorso else []
        self.griglia.imposta_fratelli(self._fratelli)
        self._aggiorna_fascia_fratelli()

    def _gruppo_corrente(self):
        """Il gruppo intero, volto corrente compreso, e il nome del frame."""
        percorso = self._volto_corrente()
        nome = fratelli_mod.nome_frame_di(self._abbinati, percorso) if percorso else None
        if nome is None:
            return [], None
        return fratelli_mod.percorsi_del_frame(self._mappa_frame, nome), nome

    def _gruppo_visibile_corrente(self):
        """Il gruppo ristretto a cio' che il filtro corrente lascia sulla
        griglia: con un filtro acceso la striscia puo' segnare solo righe
        VISIBILI, quindi contatore e frecce devono contare e camminare lo
        stesso sottoinsieme -- un fratello nascosto non deve ne' pesare sul
        conteggio ne' essere raggiungibile in silenzio da una freccia.
        Senza filtro coincide con `_gruppo_corrente`: ogni membro del
        gruppo e' visibile."""
        gruppo, nome = self._gruppo_corrente()
        visibili = set(self.modello.percorsi_visibili())
        return [p for p in gruppo if p in visibili], nome

    def _aggiorna_fascia_fratelli(self):
        gruppo, nome = self._gruppo_visibile_corrente()
        mostra = len(gruppo) > 1 and self._frame_filtrato is None
        self.fascia_fratelli.setVisible(mostra)
        self.striscia.setVisible(mostra)
        if not mostra:
            self.striscia.imposta(0, [])
            return
        corrente = self._volto_corrente()
        posizione = gruppo.index(corrente) + 1 if corrente in gruppo else 1
        self.etichetta_fratelli.setText(
            testi.faceset_sibling_counter(posizione, len(gruppo), nome))
        self.striscia.imposta(self.modello.rowCount(),
                              self.modello.righe_di(gruppo))
        self._aggiorna_banda()

    def _aggiorna_banda(self):
        """La porzione a schermo, letta dalla vista e non calcolata: con
        celle a dimensione uniforme `indexAt` sui due angoli e' la risposta
        esatta, e non va rifatta quando cambia lo zoom."""
        viewport = self.griglia.viewport().rect()
        prima = self.griglia.indexAt(viewport.topLeft())
        ultima = self.griglia.indexAt(viewport.bottomLeft())
        if not prima.isValid():
            return
        fine = ultima.row() if ultima.isValid() else self.modello.rowCount() - 1
        self.striscia.imposta_banda(prima.row(), fine)

    def vai_al_fratello(self, passo):
        """Rende corrente il fratello precedente o successivo dentro il
        gruppo VISIBILE, e ce lo porta. Non esce dal gruppo: agli estremi
        non fa niente, invece di saltare a un volto di un altro frame -- e
        con un filtro acceso non salta nemmeno su un fratello che il
        filtro nasconde, o "non farebbe niente in silenzio" su di lui."""
        gruppo, _nome = self._gruppo_visibile_corrente()
        corrente = self._volto_corrente()
        if corrente not in gruppo:
            return
        i = gruppo.index(corrente) + passo
        if not (0 <= i < len(gruppo)):
            return
        self._porta_a(gruppo[i])

    def _su_tacca_scelta(self, riga):
        percorsi = self.modello.percorsi_visibili()
        if 0 <= riga < len(percorsi):
            self._porta_a(percorsi[riga])

    def _porta_a(self, percorso):
        righe = self.modello.righe_di([percorso])
        if not righe:
            return
        index = self.modello.index(righe[0], 0)
        self.griglia.setCurrentIndex(index)
        self.griglia.scrollTo(index)

    # -- cosa e' permesso ora ------------------------------------------------

    def imposta_job_manager(self, gestore):
        if gestore is self._gestore:
            # Una seconda chiamata col medesimo gestore riaggancerebbe le
            # lambda una seconda volta, e ogni job rigenererebbe la barra
            # due volte.
            return
        self._gestore = gestore
        # Un job avviato dalla scheda dei passi occupa questa cartella
        # esattamente come uno avviato da qui: senza queste due connessioni
        # la barra resterebbe verde finche' non si cambia cartella. Il
        # doppio finto dei test non ha segnali, e non deve averne.
        for nome in ("job_started", "job_finished"):
            segnale = getattr(gestore, nome, None)
            if segnale is not None and hasattr(segnale, "connect"):
                segnale.connect(lambda *_a: self._rigenera_comandi())
        self._rigenera_comandi()

    def _occupata(self):
        if self._gestore is None or self._cartella is None:
            return None
        return conflitti_mod.chi_occupa(self._gestore, self._workspace,
                                        self._cartella, self._dataset)

    def azioni_disponibili(self):
        """chiave -> (ammessa, motivo). Il motivo e' gia' un testo da mostrare.

        Le operazioni senza gemello dst restano nel dizionario anche sul
        dataset dst, rifiutate col loro motivo: chi interroga questa
        funzione per una chiave deve trovarla sempre. Sparire dal menu e'
        una scelta di `_rigenera_comandi`, non di qui.
        """
        occupata = self._occupata()
        disponibili = {}
        for op in azioni_mod.OPERAZIONI:
            if self._cartella is None:
                disponibili[op.chiave] = (False, testi.FACESET_NO_FOLDER)
                continue
            if not _ha_il_passo(op, self._dataset):
                disponibili[op.chiave] = (
                    False, testi.action_src_only(op.etichetta))
                continue
            if occupata is not None:
                disponibili[op.chiave] = (False, testi.job_holds(occupata[0], occupata[1]))
                continue
            ammessa, _motivo = azioni_mod.applicabile(op, self._cartella)
            disponibili[op.chiave] = (
                ammessa,
                "" if ammessa else testi.action_not_applicable(op.etichetta,
                                                               self._cartella.name))
        return disponibili

    def cancellazione_disponibile(self):
        # Il controllo su _cartella e' lo stesso che fa `cancella`, e deve
        # restare lo stesso: quando i due si contraddicevano, la
        # QShortcut della Canc -- che ignora il tasto disabilitato --
        # arrivava a Path(None) dentro uno slot Qt, cioe' a qFatal e alla
        # finestra intera con dentro ogni training aperto.
        if self._cartella is None:
            return (False, testi.FACESET_NO_FOLDER)
        occupata = self._occupata()
        if occupata is not None:
            return (False, testi.job_holds(occupata[0], occupata[1]))
        return (True, "")

    # -- lanciare ------------------------------------------------------------

    def avvia_operazione(self, chiave, risposte=None):
        if not self.azioni_disponibili().get(chiave, (False, ""))[0]:
            return None
        op = _operazione(chiave)
        passo = step_by_name(azioni_mod.passo_per(op, self._dataset))
        if risposte is None:
            risposte = self._chiedi_risposte(passo)
            if risposte is None:      # annullato
                return None
        return self._lancia(passo, risposte, self._cartella)

    def avvia_indicizzazione(self):
        """Permessa anche dove le operazioni sono grigie per la cartella:
        la cache vive fuori dal progetto, quindi indicizzare non contende
        nessun artefatto. Non permessa mentre un job di questa pagina
        gira -- vedi `_lancia`."""
        if self._gestore is None or self._cartella is None:
            return None
        cartella_cache = cache_mod.percorso_cache(self._radice_e, self._cartella)
        return self._lancia(azioni_mod.PASSO_INDICE, {}, self._cartella,
                            extra_args=("--cache-dir", str(cartella_cache),
                                        "--only-missing"))

    def job_corrente(self):
        return self._job_corrente

    def _lancia(self, passo, risposte, cartella, extra_args=()):
        # R22: un job alla volta da questa pagina, e non e' una
        # mitigazione della pila condivisa -- e' la cosa giusta comunque.
        # Ogni figlio numera le sue barre da 1, quindi due job che
        # scrivono sullo stesso canale si pilotano le barre a vicenda; e
        # indicizzare MENTRE si ordina e' sbagliato di suo, perche' il
        # sort rinomina i file sotto l'indice. Con un job solo,
        # `pila.pulisci()` qui sotto resta corretta: non c'e' piu' niente
        # di vivo da cancellare.
        if self._job_corrente is not None:
            QMessageBox.warning(self, testi.TITLE_FACESET_ONE_AT_A_TIME,
                                testi.faceset_one_job_at_a_time(
                                    self._nome_job_corrente))
            return None
        try:
            job = self._gestore.try_start(passo, risposte, self._workspace,
                                          extra_args=extra_args, input_dir=cartella)
        except StepConflict as exc:
            # azioni_disponibili() e' la prima rete, non l'unica: fra la
            # costruzione del menu e il click ci sta una corsa vera (un job
            # avviato da un'altra scheda nel frattempo). Uno slot Qt non e'
            # un posto da cui lasciar scappare un'eccezione.
            QMessageBox.warning(self, testi.TITLE_STEP_BUSY, str(exc))
            self._rigenera_comandi()
            return None
        self.pila.pulisci()
        if job is not None:
            self._job_corrente = job
            self._nome_job_corrente = passo.name
            job.progress.connect(self._su_progresso)
            job.finished.connect(self._su_job_finito)
            # Il figlio parte con spawn e paga l'import a freddo di torch
            # (mainscripts/Sorter.py lo importa a livello di modulo;
            # FacesetIndex passa da facelib, che tira dentro S3FD e FAN),
            # e per l'index anche la costruzione del pool: sono secondi in
            # cui l'unico segnale era un bottone che si spegneva. La spegne
            # da sola PilaProgresso._apri alla prima riga `open`, quindi
            # qui non c'e' niente da disfare.
            self.pila.mostra_avvio(passo.name)
        self._rigenera_comandi()
        return job

    def _chiedi_risposte(self, passo):
        """Il form del catalogo in un dialogo. La semantica _touched resta:
        solo i campi toccati vengono spediti (l'invariante della voce 3.14)."""
        from gui.faceset.dialogo import DialogoOperazione
        dialogo = DialogoOperazione(passo, self)
        if not dialogo.exec_():
            return None
        return dialogo.risposte()

    def _su_progresso(self, riga):
        self.pila.applica(riga)

    def _su_job_finito(self, _codice):
        self._job_corrente = None
        self._nome_job_corrente = ""
        self.pila.pulisci()
        self.ricarica()

    # -- cancellare ----------------------------------------------------------

    def cancella(self, percorsi):
        # Prima di cancellazione_disponibile(), che gia' lo controlla: i
        # due devono concordare, e questo e' il punto in cui `None`
        # arriverebbe a Path().
        if self._cartella is None:
            return 0
        if not self.cancellazione_disponibile()[0]:
            return 0
        percorsi = list(percorsi)
        if not percorsi:
            # Senza questo, la Canc senza selezione fa comunque il mkdir
            # del cestino dentro sposta_nel_cestino: un aligned_trash
            # fantasma che poi compare nel selettore di cartella, e un
            # _ultima_mossa vuoto che rende Undo un no-op silenzioso.
            return 0
        # elimina_definitivamente non ha nessuna
        # rete propria, e' qui che si sceglie fra spostare e cancellare per
        # sempre. Dentro un cestino un secondo livello sarebbe solo un
        # posto in piu' in cui dimenticare dei file.
        if cestino_mod.e_un_cestino(self._cartella):
            quanti = cestino_mod.elimina_definitivamente(percorsi)
            self._ultima_mossa = None
        else:
            mossa = cestino_mod.sposta_nel_cestino(percorsi, self._cartella)
            self._ultima_mossa = mossa
            quanti = len(mossa.coppie)
        self.ricarica()
        return quanti

    def cancella_selezione(self):
        return self.cancella(self.griglia.percorsi_selezionati())

    def annulla_ultima_cancellazione(self):
        if self._ultima_mossa is None:
            return 0
        riportati = cestino_mod.annulla(self._ultima_mossa)
        self._ultima_mossa = None
        self.ricarica()
        return riportati

    # -- la barra ------------------------------------------------------------

    def apri_nel_gestore_file(self):
        if self._cartella is None:
            return False
        return QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._cartella)))

    def _rigenera_comandi(self):
        """Ricostruisce il menu Tools e lo stato dei tasti.

        Chiamata anche da ricalcola(), o un'azione resterebbe grigia dopo
        che il job che teneva la cartella e' finito.
        """
        disponibili = self.azioni_disponibili()
        libera = self._job_corrente is None
        ammessa, motivo = disponibili.get(CHIAVE_SORT, (False, ""))
        self.bottone_sort.setEnabled(ammessa and libera)
        self.bottone_sort.setToolTip(motivo)
        self.menu_strumenti.clear()
        for op in azioni_mod.OPERAZIONI:
            if op.chiave == CHIAVE_SORT:
                continue
            # R24: le quattro operazioni senza gemello dst spariscono dal
            # menu sul dataset dst invece di lanciare il passo src, che
            # riscriverebbe data_dst/aligned dichiarando di modificare
            # faceset_src -- e lascerebbe la pagina verde sopra la
            # cartella che si sta riscrivendo.
            if not _ha_il_passo(op, self._dataset):
                continue
            ammessa, motivo = disponibili[op.chiave]
            azione = self.menu_strumenti.addAction(op.etichetta)
            azione.setEnabled(ammessa and libera)
            azione.setToolTip(motivo)
            azione.triggered.connect(
                lambda _spuntata=False, chiave=op.chiave: self.avvia_operazione(chiave))
        self.bottone_strumenti.setEnabled(not self.menu_strumenti.isEmpty())
        cancellabile, motivo_canc = self.cancellazione_disponibile()
        self.bottone_cancella.setEnabled(cancellabile)
        self.bottone_cancella.setToolTip(motivo_canc or testi.FACESET_DELETE_TIP)
        self.bottone_annulla.setEnabled(cancellabile and self._ultima_mossa is not None)
        self.bottone_annulla.setToolTip(motivo_canc or testi.FACESET_UNDO_DELETE_TIP)
        self.bottone_apri_cartella.setEnabled(self._cartella is not None)
        con_maschere = self.maschere_disponibili()
        self.selettore_maschera.setEnabled(con_maschere)
        self.selettore_maschera.setToolTip(
            testi.FACESET_MASK_TIP if con_maschere else testi.FACESET_NO_MASKS)
        if not con_maschere:
            # Non basta ingrigire: il modo scelto su una cartella con le
            # maschere resterebbe acceso su una che non ne ha, e la griglia
            # direbbe «nessuna maschera qui» (tutta tratteggiata) dove la
            # verita' e' «questa cartella non e' indicizzata».
            self.selettore_maschera.setCurrentIndex(0)
        self.bottone_indice.setEnabled(self._gestore is not None
                                       and self._cartella is not None
                                       and libera)
        self.bottone_indice.setText(
            testi.FACESET_INDEX_NOW if self._stato == indice_mod.STATO_ASSENTE
            else testi.FACESET_UPDATE_INDEX)

    # -- finestra di dettaglio -----------------------------------------------

    def _workdir_dettaglio(self):
        """Una cartella temporanea, creata alla prima apertura, dove il
        servizio scrive le maschere annunciate."""
        if self._cartella_dettaglio is None:
            self._cartella_dettaglio = Path(tempfile.mkdtemp(prefix="dfl_dettaglio_"))
        return self._cartella_dettaglio

    def _risolvi_fratelli(self, nome_frame):
        """Il risolutore che la finestra di dettaglio interroga: i
        percorsi degli allineati di un fotogramma, se stesso compreso,
        letti dalla mappa che `ricalcola` tiene aggiornata."""
        return fratelli_mod.percorsi_del_frame(self._mappa_frame, nome_frame)

    def _su_volto_aperto(self, percorso):
        if self._cliente is None:
            from gui.dettaglio.finestra import FinestraDettaglio
            from gui.faceset.dettaglio import ClienteDettaglio
            # Il client PRIMA della finestra, e passato a lei: costruirne
            # un secondo vorrebbe dire un secondo processo figlio che
            # importa torch mentre il primo e' vivo.
            self._cliente = ClienteDettaglio(self._workdir_dettaglio())
            self._finestra_dettaglio = FinestraDettaglio(
                self._workdir_dettaglio(), cliente=self._cliente)
            # `pronto` NON si collega qui: la finestra ci si e' gia'
            # collegata da se' alla costruzione, e un secondo ascoltatore
            # che richiama `mostra` raddoppia il giro dei fratelli, e
            # ogni giro rilegge dal disco i dati DFL di tutti.
            self._cliente.fallito.connect(self._su_dettaglio_fallito)
            # Il METODO, non `self._mappa_frame` gia' calcolata: la mappa
            # si sostituisce a ogni `ricalcola()`, e un dizionario passato
            # una volta sola resterebbe quello del ricalcolo di allora.
            self._finestra_dettaglio.imposta_risolutore_fratelli(self._risolvi_fratelli)
        self._finestra_dettaglio.imposta_ordine(self.modello.percorsi_visibili())
        # I fotogrammi stanno un livello sopra la cartella che si guarda: si
        # lavora su `aligned`, `aligned_resized` o `aligned_enhanced`. Si
        # passa anche se non esiste -- chi decide se la modifica e'
        # possibile e' la finestra, in un posto solo.
        self._finestra_dettaglio.imposta_frame_dir(self._cartella.parent)
        self._finestra_dettaglio.show()
        self._finestra_dettaglio.raise_()
        self._finestra_dettaglio.activateWindow()
        # Il cambio di volto sta tutto dentro `apri_volto`: la domanda
        # sulle modifiche vive, il disegno e la richiesta al servizio. Da
        # qui si disegnava e si chiedeva in due passi, e la domanda non la
        # faceva nessuno. La finestra si mostra comunque: se rifiuta, e'
        # proprio quella che tiene il lavoro appena salvato dall'utente.
        # Il client e' asincrono e la chiamata torna subito: la clessidra
        # qui copre solo l'istante del click, un lampo. La vera attesa --
        # fino a 6 s per l'import al primo doppio click della sessione --
        # e' coperta dall'indicatore della finestra stessa
        # (`FinestraDettaglio.indicatore_attesa`, sopra la tela), che si
        # spegne alla consegna vera.
        # Il volto che la finestra mostra ORA: e' li' che la griglia deve
        # tornare se l'abbandono viene rifiutato.
        rimasto = self._finestra_dettaglio.percorso()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            aperto = self._finestra_dettaglio.apri_volto(percorso)
        finally:
            QApplication.restoreOverrideCursor()
        if not aperto and rimasto is not None:
            # Qt sposta `currentIndex` gia' sul mousePress, PRIMA che
            # `doubleClicked` parta: senza questo la griglia evidenzierebbe
            # il volto B -- e ne mostrerebbe i fratelli -- mentre la
            # finestra mostra A. La striscia della finestra si difende gia'
            # da se', dentro `apri_volto`.
            self._porta_a(rimasto)

    def _su_dettaglio_fallito(self, motivo, codice=None):
        """Il guasto del servizio, detto in inglese come tutto il resto.

        Qui si scriveva «questo file non porta dati DFL» per OGNI guasto --
        un `.npy` che manca, il fotogramma assente, il servizio morto per
        inattivita' -- cioe' una diagnosi falsa su un file sano, e la
        reazione naturale a quella frase e' cancellare il volto. Poi si e'
        mostrato il motivo del servizio tale e quale, che quella diagnosi
        la corregge ma e' italiano d'implementazione dentro una finestra
        inglese. La frase la sceglie il CODICE che il guasto porta con
        se'; il motivo resta il ripiego per i guasti senza codice.

        Stessa chiamata della finestra, che ascolta lo stesso segnale: una
        diagnosi sola, non due che si contraddicono.
        """
        self.etichetta_stato.setText(testi.dettaglio_guasto(codice, motivo))
