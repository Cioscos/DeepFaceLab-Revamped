"""La pagina di cura del faceset: composizione, niente logica propria.

Segue il PROGETTO, non il job -- e' l'unica differenza strutturale
rispetto a TrainingPanel, che e' indicizzato per job. La scheda e' una
sola, e cambiare progetto cambia cio' che mostra.
"""
import tempfile
from pathlib import Path

from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtGui import QDesktopServices, QKeySequence
from PyQt5.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel, QMenu,
                             QMessageBox, QPushButton, QShortcut, QSlider,
                             QToolButton, QVBoxLayout, QWidget)

from gui import testi
from gui.catalog import step_by_name
from gui.execution.jobs import StepConflict
from gui.faceset import azioni as azioni_mod
from gui.faceset import cache as cache_mod
from gui.faceset import cestino as cestino_mod
from gui.faceset import conflitti as conflitti_mod
from gui.faceset import indice as indice_mod
from gui.faceset.decodifica import LATI
from gui.faceset.griglia import MODI_MASCHERA, Griglia
from gui.faceset.heatmap import BIN_AMMESSI, WidgetHeatmap, bin_di_percorsi
from gui.faceset.modello import ModelloVolti
from gui.faceset.progresso import PilaProgresso

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
    def __init__(self, radice_e, impostazioni, parent=None):
        super().__init__(parent)
        self._radice_e = Path(radice_e)
        self._workspace = None
        self._dataset = "src"
        self._cartella = None
        self._indice = indice_mod.Indice([])
        self._abbinati = {}
        self._stato = indice_mod.STATO_ASSENTE
        self._cartella_dettaglio = None
        self._cliente = None
        self._finestra_dettaglio = None
        self._gestore = None
        self._ultima_mossa = None
        self._job_corrente = None
        self._nome_job_corrente = ""
        self._bin_accesi = set()

        self.modello = ModelloVolti()
        self.griglia = Griglia()
        self.griglia.setModel(self.modello)
        self.heatmap = WidgetHeatmap(impostazioni)
        self.pila = PilaProgresso()
        self.etichetta_stato = QLabel("")
        self.selettore_cartella = QComboBox()
        self.bottone_src = QPushButton(testi.FACESET_SRC)
        self.bottone_dst = QPushButton(testi.FACESET_DST)
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
        self.selettore_maschera = QComboBox()
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
        self.selettore_bin = QComboBox()
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
        self.pastiglia_filtro.linkActivated.connect(
            lambda _href: self.heatmap.pulisci_selezione())
        self.pastiglia_filtro.hide()

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
        radice = QVBoxLayout(self)
        radice.addLayout(barra)
        radice.addLayout(barra_vista)
        radice.addLayout(barra_heatmap)
        radice.addWidget(self.heatmap)
        radice.addWidget(self.griglia, 1)
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
            self._abbinati = {}
            self._indice = indice_mod.Indice([])
            self._stato = indice_mod.STATO_ASSENTE
            self.modello.imposta([], {})
            self.heatmap.pulisci_selezione()
            self.heatmap.aggiorna([])
            self._aggiorna_fascia_heatmap()
            self.etichetta_stato.setText(testi.faceset_index_state(self._stato, 0))
            self._rigenera_comandi()

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
        # Cambiare cartella cambia cosa e' applicabile, e finire un job
        # cambia cosa e' libero: senza questa riga un'azione resta grigia
        # dopo che il job che teneva la cartella e' finito.
        self._rigenera_comandi()

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
        if self._bin_accesi:
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
        """
        scelti = bin_di_percorsi(self._abbinati, self._bin_accesi,
                                 self.heatmap.bins())
        self.modello.imposta_filtro(scelti)

    def _su_filtro_bin(self, accesi):
        self._bin_accesi = set(accesi)
        self._riapplica_filtro()
        self._aggiorna_fascia_heatmap()
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

    def _su_volto_aperto(self, percorso):
        if self._cliente is None:
            from gui.faceset.dettaglio import ClienteDettaglio, FinestraDettaglio
            self._cliente = ClienteDettaglio(self._workdir_dettaglio())
            self._finestra_dettaglio = FinestraDettaglio(self._workdir_dettaglio())
            self._cliente.pronto.connect(self._su_dettaglio_pronto)
            self._cliente.fallito.connect(self._su_dettaglio_fallito)
        self._finestra_dettaglio.imposta_ordine(self.modello.percorsi_visibili())
        self._finestra_dettaglio.mostra(percorso, None)
        self._finestra_dettaglio.show()
        self._finestra_dettaglio.raise_()
        self._finestra_dettaglio.activateWindow()
        # L'attesa e' sincrona (il client blocca sul primo scambio col
        # servizio, fino a 6 s per l'import al primo doppio click della
        # sessione): il cursore a clessidra e' l'unico modo di dire
        # all'utente che la finestra non e' morta. Il `finally` serve
        # perche' `apri()` emette i suoi segnali in modo sincrono --
        # un gestore agganciato a `pronto`/`fallito` che sollevasse
        # lascerebbe altrimenti il cursore a clessidra per sempre.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._cliente.apri(percorso)
        finally:
            QApplication.restoreOverrideCursor()

    def _su_dettaglio_pronto(self, dati):
        if self._finestra_dettaglio is not None:
            self._finestra_dettaglio.mostra(self._finestra_dettaglio.percorso(), dati)

    def _su_dettaglio_fallito(self, motivo):
        self.etichetta_stato.setText(testi.FACESET_DETAIL_NO_DFL)
