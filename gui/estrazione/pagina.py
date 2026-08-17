"""La pagina di estrazione: composizione, niente logica propria.

Segue il PROGETTO, non il job -- come gui/faceset/pagina.py, e per la
stessa ragione: la scheda e' una sola, e cambiare progetto o lato cambia
cio' che mostra.

In alto la barra (lato, operazione automatica, parametri, sessione
manuale, ri-estrazione della selezione), al centro la `Tela`, in basso i
sei filtri del rapporto e la `Pellicola`.
"""
import json
import tempfile
import time
from pathlib import Path

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QButtonGroup, QComboBox, QHBoxLayout, QLabel,
                             QMessageBox, QPushButton, QVBoxLayout, QWidget)

from gui import testi
from gui.estrazione import avvio as avvio_mod
from gui.estrazione import azioni as azioni_mod
from gui.estrazione import indice as indice_mod
from gui.estrazione import servizio as servizio_mod
from gui.estrazione.modello import RUOLO_PERCORSO, ModelloFrame
from gui.estrazione.pellicola import Pellicola
from gui.estrazione.tela import Tela
from gui.execution.conflicts import libera_occupante, registra_occupante
from gui.execution.jobs import StepConflict
from gui.faceset import cache as cache_mod
from gui.faceset import cestino as cestino_mod
from gui.faceset.conflitti import PassoFittizio, artefatto_di, chi_occupa
from gui.faceset.dialogo import DialogoOperazione
from gui.faceset.indice import elenca as elenca_cartella
from gui.faceset.progresso import PilaProgresso
from gui.progetti import identita_workspace, ricorda_risposte, risposte_ricordate

ESTENSIONI = (".png", ".jpg", ".jpeg")

# Dove vive la cache del rapporto, fuori dal progetto: <radice_e>/extract-cache/<id>.
# id_cartella() e' l'unica funzione che lo sa (gui/faceset/cache.py) -- si
# riusa, non si ricalcola.
CACHE_SOTTOCARTELLA = "extract-cache"

CHIAVE_AUTO = "auto"
CHIAVE_MANUALE = "manuale"
CHIAVE_AUTO_CORREZIONE = "auto-con-correzione"
CHIAVE_RIESTRAI = "riestrai-selezione"

# Solo le operazioni che sono JOB vanno nel selettore: "manuale" ha il suo
# bottone dedicato (la sessione nativa, non il vecchio job che apre la
# finestra cv2), "riestrai-selezione" pure.
_OPERAZIONI_JOB = (CHIAVE_AUTO, CHIAVE_AUTO_CORREZIONE)

# _op_salva (mainscripts/ExtractManual.py) chiama FaceType.fromString, che
# vuole il nome lungo ("whole_face"). Il codice corto ("f"/"wf"/"head") e'
# quello dei PROMPT interattivi di Extractor.main -- un'altra via, mai
# percorsa dal servizio persistente, che il campo del catalogo condivide
# solo per riuso del form. La tabella traduce fra le due.
_FACE_TYPE_LUNGO = {"f": "full_face", "wf": "whole_face", "head": "head"}


def _radice_e_predefinita():
    """<pacchetto>/_internal/_e, calcolata dalla posizione di questo file
    -- lo stesso posto che MainWindow passa esplicitamente
    (`self._dfl_root.parent / "_e"`), qui ricavato per chi costruisce la
    pagina senza passare radice_e (i test)."""
    return Path(__file__).resolve().parent.parent.parent.parent / "_e"


def _valori_con_default(passo, risposte):
    """I campi del passo coi loro default, sovrascritti dalle risposte
    dell'utente. Il protocollo del servizio vuole un valore per campo,
    sempre -- a differenza di un job CLI, dove un campo non toccato
    semplicemente non compare in riga di comando e il valore di bordo lo
    fornisce lo script."""
    valori = dict((f.key, f.default) for f in passo.fields)
    valori.update(risposte)
    return valori


class _TrasportoProcesso:
    """Il trasporto reale del servizio: un QProcess, protocollo a righe
    JSON su stdin/stdout -- stessa forma di
    gui/faceset/dettaglio.py::ClienteDettaglio, con una differenza:
    Servizio parla in dizionari, non in righe, quindi la codifica JSON
    vive qui e non nel chiamante.

    `workdir` e' lo stesso passato a `avvio.comando_servizio` (--workdir)
    e quello in cui il figlio scrive il raster annunciato: un solo posto,
    come in FacesetDetail.
    """

    def __init__(self, workdir):
        self.workdir = Path(workdir)
        self._processo = None

    def invia(self, comando):
        from PyQt5.QtCore import QProcess
        if self._processo is None or self._processo.state() == QProcess.NotRunning:
            self._avvia()
        try:
            self._processo.write((json.dumps(comando) + "\n").encode("utf-8"))
            riga = self._leggi_una_riga_completa()
        except OSError:
            return None
        try:
            return json.loads(riga)
        except (TypeError, ValueError):
            return None

    def _leggi_una_riga_completa(self):
        """Vedi ClienteDettaglio._leggi_una_riga_completa: `canReadLine()`,
        non una singola `waitForReadyRead`, o una risposta a meta' si
        spaccia per la prossima."""
        scadenza = time.monotonic() + servizio_mod.TIMEOUT_MS / 1000.0
        while not self._processo.canReadLine():
            rimanente_ms = int((scadenza - time.monotonic()) * 1000)
            if rimanente_ms <= 0 or not self._processo.waitForReadyRead(rimanente_ms):
                raise OSError("il servizio di estrazione non ha risposto")
        return bytes(self._processo.readLine()).decode("utf-8", "replace")

    def _avvia(self):
        from PyQt5.QtCore import QProcess
        programma, argomenti = avvio_mod.comando_servizio(self.workdir)
        self._processo = QProcess()
        self._processo.setProcessChannelMode(QProcess.SeparateChannels)
        self._processo.start(programma, argomenti)
        self._processo.waitForStarted(servizio_mod.TIMEOUT_MS)

    def chiudi(self):
        if self._processo is not None:
            self._processo.kill()
            self._processo = None


class PaginaEstrazione(QWidget):
    def __init__(self, radice_e=None, parent=None):
        super().__init__(parent)
        self._radice_e = Path(radice_e) if radice_e is not None else _radice_e_predefinita()
        self._progetto = None
        self._lato = "src"
        self._cartella = None
        self._voci = []
        self._gestore = None
        self._job_corrente = None
        self._nome_job_corrente = ""
        self._trasporto = None
        self._frame_corrente = None
        self._pixmap_corrente = None
        self._rect_corrente = None
        self._landmarks_correnti = None
        self._parametri_manuale = {}
        self._salvati_per_frame = {}
        self._ultima_mossa_debug = None
        self._occupante = None            # PassoFittizio registrato mentre il servizio e' vivo
        self._identita_occupante = None

        self.modello = ModelloFrame()
        self.tela = Tela()
        self.pellicola = Pellicola()
        self.pellicola.setModel(self.modello)
        self.pila = PilaProgresso()
        self.servizio = None

        self.bottone_src = QPushButton(testi.FACESET_SRC)
        self.bottone_dst = QPushButton(testi.FACESET_DST)
        self.bottone_src.setCheckable(True)
        self.bottone_dst.setCheckable(True)
        self.selettore_operazione = QComboBox()
        self.bottone_avvia = QPushButton(testi.ESTRAZIONE_AVVIA)
        self.bottone_avvia.setToolTip(testi.ESTRAZIONE_AVVIA_TIP)
        self.bottone_manuale = QPushButton(testi.ESTRAZIONE_MANUALE)
        self.bottone_manuale.setToolTip(testi.ESTRAZIONE_MANUALE_TIP)
        self.bottone_manuale.setCheckable(True)
        self.bottone_riestrai = QPushButton(testi.ESTRAZIONE_RIESTRAI)
        self.bottone_riestrai.setToolTip(testi.ESTRAZIONE_RIESTRAI_TIP)
        self.bottone_annulla_riestrai = QPushButton(testi.ESTRAZIONE_ANNULLA_RIESTRAI)
        self.bottone_annulla_riestrai.setToolTip(testi.ESTRAZIONE_ANNULLA_RIESTRAI_TIP)
        self.bottone_indice = QPushButton(testi.ESTRAZIONE_INDICIZZA)
        self.bottone_indice.setToolTip(testi.ESTRAZIONE_INDICIZZA_TIP)
        self.etichetta_stato = QLabel("")
        self.etichetta_stato.setProperty("ruolo", "minore")

        self.gruppo_filtri = QButtonGroup(self)
        self.gruppo_filtri.setExclusive(True)
        self._bottoni_filtro = {}
        riga_filtri = QHBoxLayout()
        for chiave, etichetta, _predicato in indice_mod.FILTRI:
            bottone = QPushButton(etichetta)
            bottone.setCheckable(True)
            bottone.setChecked(chiave == "tutti")
            self.gruppo_filtri.addButton(bottone)
            self._bottoni_filtro[chiave] = bottone
            riga_filtri.addWidget(bottone)
        riga_filtri.addStretch(1)

        barra = QHBoxLayout()
        for w in (self.bottone_src, self.bottone_dst, self.selettore_operazione,
                  self.bottone_avvia, self.bottone_manuale, self.bottone_riestrai,
                  self.bottone_annulla_riestrai, self.bottone_indice):
            barra.addWidget(w)
        barra.addStretch(1)
        barra.addWidget(self.etichetta_stato)

        radice = QVBoxLayout(self)
        radice.addLayout(barra)
        radice.addWidget(self.pila)
        radice.addWidget(self.tela, 1)
        radice.addLayout(riga_filtri)
        radice.addWidget(self.pellicola)

        self.bottone_src.clicked.connect(lambda: self._cambia_lato("src"))
        self.bottone_dst.clicked.connect(lambda: self._cambia_lato("dst"))
        self.bottone_avvia.clicked.connect(lambda: self.avvia_operazione())
        self.bottone_manuale.toggled.connect(self._su_manuale_toggled)
        self.bottone_riestrai.clicked.connect(lambda: self.riestrai_selezione())
        self.bottone_annulla_riestrai.clicked.connect(lambda: self.annulla_riestrazione())
        self.bottone_indice.clicked.connect(lambda: self.aggiorna_indice())
        for chiave, bottone in self._bottoni_filtro.items():
            bottone.clicked.connect(lambda _c=False, chiave=chiave: self._su_filtro(chiave))
        self.pellicola.frame_scelto.connect(self._su_frame_scelto)
        self.tela.vettore_tracciato.connect(self._su_vettore_tracciato)
        self.tela.confermato.connect(self._su_confermato)

        self._aggiorna_lato_bottoni()
        self._rigenera_comandi()

    # -- progetto e lato -------------------------------------------------

    def apri(self, progetto, lato):
        """Elenca i frame di <progetto>/data_<lato> e legge il rapporto
        dalla cache di quella cartella. Non avvia il servizio -- vedi
        `_entra_modalita_manuale`."""
        self.ferma_servizio()
        self._progetto = Path(progetto)
        self._lato = lato
        self._cartella = self._progetto / ("data_%s" % lato)
        # La cache della decodifica e' indicizzata per percorso: le
        # miniature della cartella di prima non serviranno mai piu'.
        self.pellicola.decodificatore.svuota()
        letti = elenca_cartella(self._cartella, ESTENSIONI)
        percorsi = [p for p, _dimensione, _mtime in letti]
        self._voci = indice_mod.leggi(self._cache_dir())
        self.modello.imposta(percorsi, self._voci)
        self._salvati_per_frame = {}
        self._frame_corrente = None
        self._pixmap_corrente = None
        self._rect_corrente = None
        self._landmarks_correnti = None
        # Senza questo il bottone "Undo" resta abilitato dopo un cambio di
        # progetto o di lato e agirebbe sui file del progetto PRECEDENTE:
        # nessun dato si perde (i percorsi della Mossa sono assoluti, e
        # cestino.annulla salta se l'origine esiste gia'), ma il bottone
        # mentirebbe su cosa sta per annullare.
        self._ultima_mossa_debug = None
        self.tela.mostra(None, None, None)
        self._aggiorna_lato_bottoni()
        self._aggiorna_conteggi_filtro()
        self.etichetta_stato.setText(testi.estrazione_stato(len(percorsi)))
        self._rigenera_comandi()

    def _cache_dir(self):
        return self._radice_e / CACHE_SOTTOCARTELLA / cache_mod.id_cartella(self._cartella)

    def lato(self):
        return self._lato

    def _cambia_lato(self, lato):
        if self._progetto is not None and lato != self._lato:
            self.apri(self._progetto, lato)

    def _aggiorna_lato_bottoni(self):
        self.bottone_src.setChecked(self._lato == "src")
        self.bottone_dst.setChecked(self._lato == "dst")
        selezionata = self.selettore_operazione.currentData()
        self.selettore_operazione.clear()
        for op in azioni_mod.OPERAZIONI:
            if op.chiave not in _OPERAZIONI_JOB:
                continue
            passo_lato = op.passo_src if self._lato == "src" else op.passo_dst
            if passo_lato is None:
                continue
            self.selettore_operazione.addItem(op.etichetta, op.chiave)
        indice = self.selettore_operazione.findData(selezionata)
        self.selettore_operazione.setCurrentIndex(indice if indice >= 0 else 0)

    def _aggiorna_conteggi_filtro(self):
        conteggi = indice_mod.conta(self._voci)
        for chiave, etichetta, _predicato in indice_mod.FILTRI:
            self._bottoni_filtro[chiave].setText(
                "%s (%d)" % (etichetta, conteggi.get(chiave, 0)))

    def _su_filtro(self, chiave):
        self.modello.applica_filtro(chiave)

    # -- job manager e conflitti ------------------------------------------

    def imposta_job_manager(self, gestore):
        if gestore is self._gestore:
            return
        self._gestore = gestore
        for nome in ("job_started", "job_finished"):
            segnale = getattr(gestore, nome, None)
            if segnale is not None and hasattr(segnale, "connect"):
                segnale.connect(lambda *_a: self._rigenera_comandi())
        self._rigenera_comandi()

    def _occupata(self):
        if self._gestore is None or self._progetto is None:
            return None
        return chi_occupa(self._gestore, self._progetto, self._cartella, self._lato)

    # -- job: estrazione automatica ---------------------------------------

    def _cartella_e_report(self):
        """gli extra-args comuni a ogni lavoro "extract" lanciato da qui:
        --report-dir punta alla STESSA cartella che `apri()` rilegge
        (`_cache_dir()`, un solo posto che lo sa) -- senza, nessuno scrive
        mai `frames.ndjson` in produzione (C1 del ledger)."""
        return ("--report-dir", str(self._cache_dir()))

    def avvia_operazione(self, chiave=None, risposte=None):
        if self._gestore is None or self._progetto is None:
            return None
        if self.servizio is not None:
            QMessageBox.warning(self, testi.TITLE_STEP_BUSY,
                                testi.ESTRAZIONE_MANUALE_OCCUPA)
            return None
        chiave = chiave if chiave is not None else self.selettore_operazione.currentData()
        if chiave is None:
            return None
        try:
            passo = azioni_mod.passo_per(chiave, self._lato)
        except KeyError:
            return None
        if risposte is None:
            risposte = self._chiedi_risposte(passo)
            if risposte is None:      # annullato
                return None
        return self._lancia(passo, risposte, self._cartella,
                            extra_args=self._cartella_e_report())

    def riestrai_selezione(self, risposte=None):
        """Ri-estrazione selettiva: i frame
        selezionati nella pellicola vengono marcati «da rifare» spostando
        nel cestino il loro `aligned_debug`, poi si lancia lo stesso passo
        del catalogo che oggi si raggiunge cancellando quei file a mano dal
        gestore di file. Solo dst: e' l'unico lato per cui l'operazione
        esiste (`passo_dst` non None).

        Il cestinamento sta DOPO il dialogo, non prima: annullare il
        dialogo non deve lasciare file gia' spostati e nessun lavoro
        partito. Non dopo il LANCIO, pero' -- il processo figlio scansiona
        `aligned_debug` per decidere cosa rifare appena parte
        (`DeletedFilesSearcherSubprocessor`), e spostare i file dopo
        avviarlo sarebbe una corsa vera fra il genitore e il figlio, non
        solo un fastidio di UI. La `Mossa` si tiene comunque, e resta
        annullabile con `annulla_riestrazione()` anche se il lancio viene
        rifiutato per conflitto: quel caso lascerebbe altrimenti i file nel
        cestino senza nessun lavoro partito e senza modo di riportarli
        indietro da qui."""
        if self._gestore is None or self._progetto is None or self._lato != "dst":
            return None
        if self.servizio is not None:
            QMessageBox.warning(self, testi.TITLE_STEP_BUSY,
                                testi.ESTRAZIONE_MANUALE_OCCUPA)
            return None
        percorsi = self.pellicola.selezione()
        if not percorsi:
            return None
        try:
            passo = azioni_mod.passo_per(CHIAVE_RIESTRAI, self._lato)
        except KeyError:
            return None
        if risposte is None:
            risposte = self._chiedi_risposte(passo)
            if risposte is None:      # annullato
                return None
        cartella_debug = self._cartella / "aligned_debug"
        da_spostare = [cartella_debug / (p.stem + ".jpg") for p in percorsi]
        da_spostare = [p for p in da_spostare if p.exists()]
        if da_spostare:
            self._ultima_mossa_debug = cestino_mod.sposta_nel_cestino(da_spostare, cartella_debug)
        job = self._lancia(passo, risposte, self._cartella,
                           extra_args=self._cartella_e_report())
        self._rigenera_comandi()
        return job

    def annulla_riestrazione(self):
        """Riporta al loro posto i debug spostati dall'ultima
        `riestrai_selezione()` -- reso raggiungibile perche' senza un
        chiamante `cestino.annulla` sarebbe irraggiungibile da qui."""
        if self._ultima_mossa_debug is None:
            return 0
        riportati = cestino_mod.annulla(self._ultima_mossa_debug)
        self._ultima_mossa_debug = None
        self._rigenera_comandi()
        return riportati

    def aggiorna_indice(self, risposte=None):
        """Il ripiego per le cartelle estratte prima che il rapporto per
        frame esistesse, che non ne hanno mai scritto uno:
        `extracttool index` lo ricostruisce per
        inferenza da `aligned/` e `aligned_debug/`, senza rieseguire il
        rilevamento. `risposte={}` di default: il passo non ha campi."""
        if self._gestore is None or self._progetto is None or self._cartella is None:
            return None
        job = self._lancia(azioni_mod.PASSO_INDICE, {} if risposte is None else risposte,
                           self._cartella,
                           extra_args=("--aligned-dir", str(self._cartella / "aligned"),
                                       "--cache-dir", str(self._cache_dir())),
                           controlla_occupazione=False)
        return job

    def _chiedi_risposte(self, passo):
        """Il dialogo del passo, precaricato con cio' che il progetto
        ricorda e che al termine glielo fa ricordare.

        Stesso meccanismo della vista-passo del menu principale, dalle
        stesse due funzioni (`gui/progetti.py`): la scelta del motore si
        ricorda per progetto, e da qui non si ricordava affatto -- il dialogo nasceva come uno StepForm
        nudo. La memoria e' del PROGETTO, quindi la chiave e' il nome del
        passo e non il lato: `4) data_src faceset extract` e
        `5) data_dst faceset extract` sono due passi diversi e ricordano
        separatamente, che e' cio' che serve (src e dst non si estraggono
        con le stesse risposte).

        La scrittura e' protetta come ogni azione che tocca project.json:
        uno slot Qt e' un vicolo cieco per un'eccezione -- PyQt5 la
        trasforma in qFatal e si porta via il processo con dentro ogni
        altro lavoro aperto -- e un permesso negato o un antivirus con un
        handle sul file non devono impedire l'estrazione, che e' cio' che
        l'utente ha chiesto davvero.
        """
        dialogo = DialogoOperazione(passo, self)
        ricordate = risposte_ricordate(self._progetto, passo.name)
        if ricordate:
            dialogo.form.set_remembered_values(ricordate)
        if not dialogo.exec_():
            return None
        risposte = dialogo.risposte()
        try:
            ricorda_risposte(self._progetto, passo.name, risposte)
        except Exception as errore:
            QMessageBox.warning(self, testi.TITLE_PROJECT_ACTION_FAILED,
                                testi.msg_project_action_failed(str(errore)))
        return risposte

    def _lancia(self, passo, risposte, cartella, extra_args=(), controlla_occupazione=True):
        # Un job alla volta da questa pagina -- stessa scelta di
        # gui/faceset/pagina.py, e per la stessa ragione (le barre
        # numerate dal figlio si pilotano a vicenda con due job vivi).
        if self._job_corrente is not None:
            QMessageBox.warning(self, testi.TITLE_FACESET_ONE_AT_A_TIME,
                                testi.faceset_one_job_at_a_time(
                                    self._nome_job_corrente))
            return None
        # controlla_occupazione=False e' solo per aggiorna_indice(): quel
        # passo non dichiara ne' consumes ne' produces ne' modifies (la
        # cache vive fuori dal progetto, come in gui/faceset/pagina.py),
        # quindi non contende mai niente -- e non deve restare grigio
        # perche' un ALTRO job tiene la cartella, esattamente come
        # avvia_indicizzazione() in gui/faceset/pagina.py.
        if controlla_occupazione:
            occupata = self._occupata()
            if occupata is not None:
                QMessageBox.warning(self, testi.TITLE_STEP_BUSY,
                                    testi.job_holds(occupata[0], occupata[1]))
                return None
        try:
            job = self._gestore.try_start(passo, risposte, self._progetto,
                                          extra_args=extra_args, input_dir=cartella)
        except StepConflict as exc:
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

    def _su_progresso(self, riga):
        self.pila.applica(riga)

    def _su_job_finito(self, _codice):
        self._job_corrente = None
        self._nome_job_corrente = ""
        self.pila.pulisci()
        # Il job ha scritto il rapporto incrementalmente: ricaricare
        # rilegge cio' che ha appena prodotto.
        self.apri(self._progetto, self._lato)

    # -- sessione manuale ---------------------------------------------------

    def _su_manuale_toggled(self, attivo):
        if attivo:
            self._entra_modalita_manuale()
        else:
            self.ferma_servizio()

    def _entra_modalita_manuale(self):
        if self.servizio is not None or self._progetto is None:
            return
        occupata = self._occupata()
        if occupata is not None:
            QMessageBox.warning(self, testi.TITLE_STEP_BUSY,
                                testi.job_holds(occupata[0], occupata[1]))
            self.bottone_manuale.setChecked(False)
            return
        try:
            passo = azioni_mod.passo_per(CHIAVE_MANUALE, self._lato)
        except KeyError:
            self.bottone_manuale.setChecked(False)
            return
        risposte = self._chiedi_risposte(passo)
        if risposte is None:          # annullato
            self.bottone_manuale.setChecked(False)
            return
        self._parametri_manuale = _valori_con_default(passo, risposte)
        # Non parte all'apertura della pagina: importa
        # torch, quindi solo qui, entrando davvero in modalita' manuale.
        # ExtractManual non carica alcun modello -- landmarks_da_vettore e'
        # geometria pura -- il costo e' l'import, non dei pesi in VRAM.
        workdir = Path(tempfile.mkdtemp(prefix="dfl_estrazione_"))
        self._trasporto = _TrasportoProcesso(workdir)
        self.servizio = servizio_mod.Servizio(self._trasporto)
        self._salvati_per_frame = {}
        # Visibile a chi_occupa()/try_start() come se fosse un job (I4 del
        # ledger): senza questo, un lavoro avviato dalla lista Steps o
        # dalla pagina faceset sullo stesso `aligned` non verrebbe mai
        # rifiutato mentre questa sessione ci scrive -- il QProcess del
        # servizio non e' un Job, quindi job_manager.active_jobs() non lo
        # vede da solo.
        self._identita_occupante = identita_workspace(self._progetto)
        self._occupante = PassoFittizio(passo.name, (), (),
                                        (artefatto_di(self._cartella, self._lato),))
        registra_occupante(self._identita_occupante, self._occupante)
        self._prossimo_frame()
        self._rigenera_comandi()

    def ferma_servizio(self):
        """Pubblico e idempotente: e' il metodo che i percorsi di chiusura
        garantiti chiamano. Tre chiamanti di produzione: il toggle esplicito
        dell'utente, il cambio di progetto/lato (`apri`), e
        `MainWindow.closeEvent` -- quest'ultimo e' quello che conta davvero,
        perche' questa pagina vive dentro una scheda di un QTabWidget e non
        e' mai una finestra a se': il `closeEvent` di QUESTA classe (sotto)
        non riceve mai l'evento quando e' la finestra principale a chiudersi,
        solo se la pagina fosse mai mostrata da sola. Un `ferma()` senza
        chiamante di produzione e' esattamente il difetto lasciato aperto dal
        ciclo faceset (`gui/faceset/dettaglio.py`, registro difetti) -- qui il
        costo sarebbe comunque piu' alto da lasciare aperto: un processo
        appeso resta un processo appeso, anche se (a differenza di
        FacesetDetail) ExtractManual non tiene nessun modello in VRAM."""
        if self.servizio is not None:
            self.servizio.ferma()
            self.servizio = None
        if self._occupante is not None:
            libera_occupante(self._identita_occupante, self._occupante)
            self._occupante = None
            self._identita_occupante = None
        self._trasporto = None
        self._frame_corrente = None
        self._pixmap_corrente = None
        self._rect_corrente = None
        self._landmarks_correnti = None
        if self.bottone_manuale.isChecked():
            self.bottone_manuale.setChecked(False)
        self._rigenera_comandi()

    #override
    def closeEvent(self, evento):
        self.ferma_servizio()
        super().closeEvent(evento)

    def _percorsi_visibili(self):
        """I percorsi che la pellicola mostra ORA -- rispetta il filtro
        acceso, cosi' la navigazione manuale scorre la stessa fetta che
        l'utente sta guardando."""
        m = self.modello
        return [m.data(m.index(i, 0), RUOLO_PERCORSO) for i in range(m.rowCount())]

    def _su_frame_scelto(self, percorso):
        if self.servizio is not None:
            self._carica_frame(percorso)

    def _prossimo_frame(self):
        percorsi = self._percorsi_visibili()
        if not percorsi:
            self._frame_corrente = None
            self._pixmap_corrente = None
            self.tela.mostra(None, None, None)
            return
        if self._frame_corrente in percorsi:
            i = (percorsi.index(self._frame_corrente) + 1) % len(percorsi)
        else:
            i = 0
        self._carica_frame(percorsi[i])

    def _carica_frame(self, percorso):
        if self.servizio is None:
            return
        # `shape` si scarta apposta: il raster che il servizio scrive E' il
        # frame a risoluzione nativa (mainscripts/ExtractManual.py::_op_frame
        # ricodifica senza ridimensionare), quindi la dimensione del pixmap
        # e' gia' quella del frame -- ed e' l'unica sorgente della scala che
        # la tela applica (gui/estrazione/tela.py::trasformazione). Tenerne
        # due sarebbe tenere due verita' da riconciliare.
        raster, _forma = self.servizio.frame(percorso)
        self._frame_corrente = percorso
        self._rect_corrente = None
        self._landmarks_correnti = None
        pixmap = None
        if raster is not None and self._trasporto is not None:
            candidata = QPixmap(str(self._trasporto.workdir / raster))
            if not candidata.isNull():
                pixmap = candidata
        self._pixmap_corrente = pixmap
        self.tela.mostra(pixmap, None, None)

    def _su_vettore_tracciato(self, centro, punta):
        if self.servizio is None:
            return
        rect, landmarks = self.servizio.landmark(centro, punta)
        self._rect_corrente = rect
        self._landmarks_correnti = landmarks
        self.tela.mostra(self._pixmap_corrente, rect, landmarks)

    def _prossimo_face_idx_libero(self, frame):
        """Il face_idx da usare per il PROSSIMO volto salvato da QUESTO
        frame, seminato da cio' che e' gia' su disco -- mai da 0.

        Lo spazio dei nomi e' condiviso con l'estrazione automatica
        (`mainscripts/Extractor.py`: `f"{filepath.stem}_{face_idx}.jpg"`),
        e `ExtractorLib.salva_volto` scrive senza nessuna guardia di
        esistenza: il primo volto salvato a mano su un frame gia' estratto
        in automatico cancellerebbe `<stem>_0.jpg`, e uscire e rientrare in
        modalita' manuale cancellerebbe il volto della sessione precedente
        (I1 del ledger). Il MASSIMO fra gli indici gia' presenti, non un
        conteggio: un buco (un volto cancellato a mano) non deve far
        ripetere un indice ancora occupato."""
        aligned = self._cartella / "aligned"
        prefisso = frame.stem + "_"
        trovati = [-1]
        try:
            candidati = list(aligned.glob(prefisso + "*.jpg"))
        except OSError:
            candidati = []
        for f in candidati:
            coda = f.stem[len(prefisso):]
            # isdecimal(), non isdigit(): isdigit() e' vero anche per cifre
            # tipografiche Unicode ("00000_².jpg", un apice) che int()
            # NON accetta -- solleverebbe ValueError dentro lo slot di
            # tela.confermato, e uno slot che solleva chiama qFatal e si
            # porta via il processo con dentro ogni training aperto (N1 del
            # ledger). Il try/except intorno al solo int() e' la seconda
            # guardia: un nome che superasse comunque isdecimal() per una
            # ragione che non ho previsto non deve fermare gli altri file.
            if coda.isdecimal():
                try:
                    trovati.append(int(coda))
                except ValueError:
                    continue
        return max(trovati) + 1

    def _su_confermato(self):
        if (self.servizio is None or self._frame_corrente is None
                or self._rect_corrente is None or self._landmarks_correnti is None):
            return
        valori = self._parametri_manuale
        # face_idx: la pagina sa gia' quanti volti ha salvato in QUESTA
        # sessione per QUESTO frame -- e la prima volta lo semina da cio'
        # che e' gia' su disco, non da 0 (I1 del ledger). Senza passarlo il
        # servizio nomina il file con face_idx=0 sempre, e un secondo volto
        # dallo stesso frame sovrascriverebbe il primo in silenzio.
        if self._frame_corrente not in self._salvati_per_frame:
            self._salvati_per_frame[self._frame_corrente] = \
                self._prossimo_face_idx_libero(self._frame_corrente)
        face_idx = self._salvati_per_frame[self._frame_corrente]
        nome_file = self.servizio.salva(
            path=str(self._frame_corrente),
            output_dir=str(self._cartella / "aligned"),
            face_idx=face_idx,
            rect=self._rect_corrente,
            landmarks=self._landmarks_correnti,
            face_type=_FACE_TYPE_LUNGO.get(valori.get("face-type"), "whole_face"),
            image_size=valori.get("image-size", 512),
            jpeg_quality=valori.get("jpeg-quality", 90))
        if nome_file is not None:
            self._salvati_per_frame[self._frame_corrente] = face_idx + 1
            self.etichetta_stato.setText(testi.estrazione_volto_salvato(nome_file))
        self._prossimo_frame()

    # -- cosa e' permesso ora ------------------------------------------------

    def _rigenera_comandi(self):
        libera = self._job_corrente is None
        pronto = self._progetto is not None
        manuale_attiva = self.servizio is not None
        self.bottone_src.setEnabled(pronto)
        self.bottone_dst.setEnabled(pronto)
        self.selettore_operazione.setEnabled(pronto and self.selettore_operazione.count() > 0)
        self.bottone_avvia.setEnabled(pronto and libera and not manuale_attiva
                                      and self._gestore is not None
                                      and self.selettore_operazione.count() > 0)
        self.bottone_manuale.setEnabled(pronto and libera)
        self.bottone_manuale.setText(
            testi.ESTRAZIONE_MANUALE_ESCI if manuale_attiva else testi.ESTRAZIONE_MANUALE)
        self.bottone_riestrai.setEnabled(pronto and libera and not manuale_attiva
                                         and self._lato == "dst"
                                         and self._gestore is not None)
        self.bottone_annulla_riestrai.setEnabled(self._ultima_mossa_debug is not None)
        # `not manuale_attiva` anche qui, benche' aggiorna_indice() passi
        # apposta controlla_occupazione=False: quel ripiego ricostruisce il
        # rapporto leggendo `aligned/`, ed e' proprio la cartella che la
        # sessione manuale sta scrivendo -- l'unico caso in cui saltare il
        # controllo di occupazione fa danno invece di evitare un grigio di
        # troppo. Fuori dalla sessione manuale resta raggiungibile anche
        # mentre un ALTRO job tiene la cartella, che e' la ragione per cui
        # quella deroga esiste.
        self.bottone_indice.setEnabled(pronto and libera and not manuale_attiva
                                       and self._gestore is not None)
