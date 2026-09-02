"""La pagina di fusione: coordina servizio, tela, pannello, timeline,
pellicola e rapporto.

Nessun import di merger/models/mainscripts: il nucleo della fusione vive
in un processo figlio, e con lui si parla solo il protocollo di
gui/fusione/servizio.py. I codici di guasto non li legge nemmeno questa
pagina -- la frase la sceglie gui/testi.py dal codice che viaggia, e
fatale o no lo dice lo STATO in cui l'errore arriva.

Ogni slot che riceve dati dal figlio li valida prima di toccare un widget:
un'eccezione dentro uno slot Qt e' un qFatal, non una traccia da leggere.
"""
import json
import os
import tempfile
from pathlib import Path

from PyQt5.QtCore import QEventLoop, Qt, QTimer
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (QAction, QCheckBox, QDialog, QHBoxLayout, QLabel,
                             QLineEdit, QMessageBox, QPushButton, QSpinBox,
                             QSplitter, QVBoxLayout, QWidget)

from gui import fascia_aiuto, numeri, testi, theme
from gui.catalog.merging import STEPS as PASSI_MERGE
from gui.execution.conflicts import libera_occupante, registra_occupante
from gui.execution.jobs import StepConflict
from gui.faceset.conflitti import PassoFittizio
from gui.faceset.progresso import PilaProgresso
from gui.fascia_aiuto import FasciaAiuto
from gui.fusione import avvio as avvio_mod
from gui.fusione import comandi as comandi_mod
from gui.fusione import esporta as esporta_mod
from gui.fusione import pellicola as pl
from gui.fusione import servizio as servizio_mod
from gui.fusione import tela as tela_mod
from gui.fusione import timeline as tl
from gui.fusione.esporta import DialogoExport, PannelloEsito
from gui.fusione.pannello_parametri import PannelloParametri
from gui.fusione.preset import BarraPreset
from gui.fusione.rapporto import PannelloRapporto
from gui.fusione.sonda import StrisciaSonda
from gui.progetti import identita_workspace, ricorda_risposte, risposte_ricordate
from gui.telemetry import EventTail
from gui.workspace import saved_model_class, saved_model_names

NOME_PASSO_OCCUPANTE = "merge session"

# Le stesse cinque di core/pathex.py::image_extensions, che e' cio' con cui
# il figlio enumera `data_dst` (merger/preparazione.py::raccogli_frames):
# ricopiate perche' `gui/` non importa `core`, e legate a quella lista da un
# test. Enumerare meno estensioni qui vorrebbe dire che ogni indice del
# protocollo punta a un altro fotogramma, in silenzio.
ESTENSIONI = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

# Quanto si aspetta la risposta a `chiudi` prima di uccidere il figlio: e'
# il tempo che gli si concede per spegnere il batch, finire cio' che ha in
# volo e salvare <nome>_<Classe>_merger_session.dat (nome_file_sessione).
# Scaduto, si uccide comunque: una GUI ferma per sempre e' peggio di una
# sessione non ripresa.
ATTESA_CHIUSURA_MS = 5000

# Debounce dell'anteprima del piano: un cambio di keyframe o di spunta
# chiede al servizio quanti frame cambierebbero, ma non a ogni tick di
# slider -- stesso ritmo del pannello parametri (DEBOUNCE_MS).
ANTEPRIMA_PIANO_MS = 150

# Il numero di frame che «Sample» chiede la prima volta che il campo si
# abilita: clippato al totale del progetto in _su_pronto.
SONDA_DEFAULT = 8

# Nome di cartella, non un testo da mostrare: costante di modulo perche'
# dentro una chiamata a un widget (`addItems`) un letterale con lettere e'
# indistinguibile da una frase, e la rete di tests_gui lo intercetta.
CARTELLA_MODELLI = "model"

# Il rapporto che il figlio scrive in `merged` a fine batch
# (mainscripts/MergeSession.py::_fine_batch), riletto a pagina riaperta:
# l'evento `rapporto` passa una volta sola, il file resta. Chi chiude a
# batch acceso non ne ha nessuno -- lo scrive solo la fine spontanea.
NOME_RAPPORTO = "merge_report.json"

# Le tre chiavi del catalogo che questa pagina chiede prima di avviare: il
# resto dei parametri e' nel pannello, che li chiede al nucleo a sessione
# aperta. Etichette, aiuti e presenza vengono da li' -- non riscritti qui.
CHIAVE_GPU = "which-gpu-indexes-to-choose"
CHIAVE_MORPH = "morph-factor"
CHIAVE_WORKER = "number-of-workers"

# Il prompt a cui risponde il valore del campo morph, con le parole del
# nucleo (models/Model_H2/Model.py, Model_AMP): la risposta si consegna al
# figlio per nome del prompt, come fa ogni passo del catalogo.
PROMPT_MORPH = "Morph factor"


def nome_file_sessione(nome_modello, classe):
    """Il nome del `.dat` di sessione che il servizio scrive accanto al
    modello.

    Lo compone `ModelBase.get_strpath_storage_for_file('merger_session.dat')`
    su `get_model_name()`, che porta gia' la classe: il file e'
    `<nome>_<Classe>_merger_session.dat`, non `<nome>_merger_session.dat`.
    Vive qui e non dentro la pagina perche' il nome lo nomina anche chi
    guarda il disco da fuori (la voce 3.79 nasce da una copia sbagliata di
    questa regola in un posto solo).
    """
    return "%s_%s_merger_session.dat" % (nome_modello, classe)


def _passo_merge(classe):
    """Il passo del catalogo che fonde con quella classe di modello."""
    for passo in PASSI_MERGE:
        if passo.name == "7) merge %s" % classe:
            return passo
    return None


def _campo(passo, chiave):
    if passo is None:
        return None
    for field in passo.fields:
        if field.key == chiave:
            return field
    return None


def _campo_globale(chiave):
    """Il FieldDef di quella chiave, da qualunque passo della famiglia lo
    porti: label e aiuto sono gli stessi oggetti per tutti."""
    for passo in PASSI_MERGE:
        field = _campo(passo, chiave)
        if field is not None:
            return field
    return None


class PaginaFusione(QWidget):
    """La fase «Merge», che e' una pagina e non piu' un elenco di passi."""

    def __init__(self, radice_e=None, parent=None):
        super().__init__(parent)
        self._radice_e = Path(radice_e) if radice_e is not None else Path(tempfile.gettempdir())
        self._progetto = None
        self._stato = "idle"
        self.servizio = None
        self._trasporto = None
        self._occupante = None
        self._identita_occupante = None
        self._rect = []
        self._cursore = 0
        self._totali = 0
        self._indice_per_nome = {}
        self._tail_progresso = None
        self._in_chiusura = False       # dentro l'attesa della risposta a `chiudi`
        self._scheda_chiusa = False     # la scheda e' stata tolta dal QTabWidget
        self._keyframes = []
        self._timer_anteprima = QTimer(self)
        self._timer_anteprima.setSingleShot(True)
        self._timer_anteprima.setInterval(ANTEPRIMA_PIANO_MS)
        self._timer_anteprima.timeout.connect(self._chiedi_anteprima_piano)
        self._job_manager = None
        self._job_export = None
        self._contenitore_export = None
        self.apri_dialogo_export = self._apri_dialogo_export
        self.avvisa = self._avvisa_con_dialogo
        self._costruisci()
        self._collega_scorciatoie()

    # -- costruzione ---------------------------------------------------------

    def _costruisci(self):
        colonna = QVBoxLayout(self)
        riga = QHBoxLayout()
        riga.addWidget(QLabel(testi.FUSIONE_MODEL))
        self.tendina_modello = theme.tendina()
        riga.addWidget(self.tendina_modello)
        self.tendina_classe = theme.tendina()
        self.tendina_classe.addItems(sorted(p.name.split()[-1] for p in PASSI_MERGE))
        riga.addWidget(self.tendina_classe)

        campo_gpu = _campo_globale(CHIAVE_GPU)
        riga.addWidget(QLabel(campo_gpu.label if campo_gpu else ""))
        self.campo_gpu = QLineEdit()
        self.campo_gpu.setPlaceholderText(testi.FUSIONE_GPU_PLACEHOLDER)
        self.campo_gpu.setToolTip(campo_gpu.help if campo_gpu else "")
        riga.addWidget(self.campo_gpu)

        campo_morph = _campo_globale(CHIAVE_MORPH)
        self.etichetta_morph = QLabel(campo_morph.label if campo_morph else "")
        self.campo_morph = QLineEdit(str(campo_morph.default if campo_morph else 1.0))
        self.campo_morph.setToolTip(campo_morph.help if campo_morph else "")
        riga.addWidget(self.etichetta_morph)
        riga.addWidget(self.campo_morph)

        campo_worker = _campo_globale(CHIAVE_WORKER)
        riga.addWidget(QLabel(campo_worker.label if campo_worker else ""))
        self.campo_worker = QSpinBox()
        # Il tetto e' quello che la console si calcola a runtime (il numero
        # di core): oltre, i processi di compositing si contendono la CPU.
        nuclei = max(1, os.cpu_count() or 4)
        self.campo_worker.setRange(1, nuclei)
        self.campo_worker.setValue(min(8, nuclei))
        self.campo_worker.setToolTip(campo_worker.help if campo_worker else "")
        riga.addWidget(self.campo_worker)

        self.bottone_avvio = QPushButton(testi.FUSIONE_START)
        self.bottone_avvio.clicked.connect(lambda: self.avvia_sessione())
        riga.addWidget(self.bottone_avvio)
        self.bottone_fine = QPushButton(testi.FUSIONE_STOP_SESSION)
        self.bottone_fine.clicked.connect(self.ferma_servizio)
        riga.addWidget(self.bottone_fine)
        colonna.addLayout(riga)
        self.nota_ripresa = QLabel("")
        colonna.addWidget(self.nota_ripresa)

        self.pila = PilaProgresso()
        colonna.addWidget(self.pila)

        centro = QSplitter(Qt.Horizontal)
        self.modello_frame = pl.ModelloFrameFusione()
        self.pellicola = pl.PellicolaFusione()
        self.pellicola.setModel(self.modello_frame)
        self.pellicola.frame_scelto.connect(self._vai)
        mezzo = QWidget()
        vm = QVBoxLayout(mezzo)
        self.tela = tela_mod.Tela()
        self.tela.setFocusPolicy(Qt.StrongFocus)
        self.lente = tela_mod.Lente()
        vm.addWidget(self.tela, 3)
        vm.addWidget(self.lente, 1)
        viste = QHBoxLayout()
        for chiave, etichetta in (("original", testi.FUSIONE_VIEW_ORIGINAL),
                                  ("merged", testi.FUSIONE_VIEW_MERGED),
                                  ("mask", testi.FUSIONE_VIEW_MASK)):
            b = QPushButton(etichetta)
            b.clicked.connect(lambda _c, k=chiave: self.tela.imposta_vista(k))
            viste.addWidget(b)
        vm.addLayout(viste)
        centro.addWidget(mezzo)
        # I parametri e, sotto di loro, la fascia che spiega quello che il
        # mouse tocca -- compresa la singola VOCE di una tendina aperta,
        # che e' il momento in cui la spiegazione serve (gui/fascia_aiuto.py).
        # Sta nella colonna dei parametri e non in fondo alla pagina per la
        # stessa ragione del rapporto: li' la sua altezza si sommerebbe a
        # quella di tutto il resto.
        colonna_parametri = QWidget()
        vp = QVBoxLayout(colonna_parametri)
        vp.setContentsMargins(0, 0, 0, 0)
        self.fascia = FasciaAiuto()
        self.fascia.riposo(testi.HELP_REST)
        self.barra_preset = BarraPreset(cfg_corrente=lambda: self.pannello.cfg())
        self.barra_preset.preset_scelto.connect(self._su_preset_scelto)
        fascia_aiuto.osserva(self.barra_preset.tendina, self.fascia, testi.FUSIONE_PRESET,
                             testi.FUSIONE_HELP_PRESET)
        vp.addWidget(self.barra_preset)
        self.pannello = PannelloParametri()
        self.pannello.cfg_cambiata.connect(self._su_cfg_cambiata)
        vp.addWidget(self.pannello, 1)
        self.pannello.collega_fascia(self.fascia)
        vp.addWidget(self.fascia)
        centro.addWidget(colonna_parametri)
        # Il rapporto sta nello splitter, non in fondo alla colonna: da li'
        # aggiungeva la propria altezza minima a quella di tutto il resto, e
        # comparendo allargava la finestra da 900 a 1166 px (misurato
        # offscreen, sonda su MainWindow) -- su uno schermo 1080 sarebbe la
        # coda dell'elenco a finire fuori. Accanto ai parametri costa
        # larghezza, che lo splitter sa distribuire.
        self.rapporto = PannelloRapporto()
        self.rapporto.frame_scelto.connect(self._vai)
        self.rapporto.setVisible(False)
        centro.addWidget(self.rapporto)
        colonna.addWidget(centro, 1)
        colonna.addWidget(self.pellicola)

        self.striscia_sonda = StrisciaSonda()
        self.striscia_sonda.frame_scelto.connect(self._vai)
        colonna.addWidget(self.striscia_sonda)

        self.timeline = tl.Timeline()
        self.timeline.frame_scelto.connect(self._vai)
        colonna.addWidget(self.timeline)

        # Due righe, non una: i quindici controlli in fila imponevano 1487 px
        # di larghezza minima alla pagina intera a scala normale e 1820 a
        # xlarge, cioe' erano loro a deciderla. Sopra la navigazione e la
        # sessione, sotto cio' che l'ondata 2 ha aggiunto -- il piano, la
        # sonda e l'export.
        comandi = QHBoxLayout()
        comandi_2 = QHBoxLayout()
        self.bottoni = {}
        for chiave, etichetta, riga in (("precedente", testi.FUSIONE_PREV, comandi),
                                        ("successivo", testi.FUSIONE_NEXT, comandi),
                                        ("successivo_propaga", testi.FUSIONE_PROPAGATE_NEXT, comandi),
                                        ("ultimo_propaga", testi.FUSIONE_PROPAGATE_ALL, comandi),
                                        ("keyframe", testi.FUSIONE_KEYFRAME, comandi_2),
                                        ("piano", testi.FUSIONE_APPLY_PLAN, comandi_2),
                                        ("batch", testi.FUSIONE_PROCESS_ALL, comandi),
                                        ("salva_sessione", testi.FUSIONE_SAVE_SESSION, comandi)):
            b = QPushButton(etichetta)
            b.clicked.connect(lambda _c, k=chiave: self._su_comando(k))
            self.bottoni[chiave] = b
            riga.addWidget(b)
        self.spunta_interpola = QCheckBox(testi.FUSIONE_INTERPOLATE)
        self.spunta_interpola.setChecked(True)
        self.spunta_interpola.toggled.connect(lambda _v: self._timer_anteprima.start())
        comandi_2.addWidget(self.spunta_interpola)
        self.etichetta_piano = QLabel("")
        comandi_2.addWidget(self.etichetta_piano)
        # Separa la frase del piano dal gruppo della sonda: attaccate, «7
        # frames will be redone» e «Sample N» si leggono come una riga sola.
        comandi_2.addStretch(1)
        fascia_aiuto.osserva(self.bottoni["piano"], self.fascia, testi.FUSIONE_APPLY_PLAN,
                             testi.FUSIONE_HELP_PLAN)
        self.bottone_stop = QPushButton(testi.FUSIONE_STOP)
        self.bottone_stop.clicked.connect(self._ferma_batch)
        comandi.addWidget(self.bottone_stop)
        self.etichetta_sonda = QLabel(testi.FUSIONE_SAMPLE_N)
        self.campo_sonda = QSpinBox()
        self.campo_sonda.setRange(1, 1)
        self.campo_sonda.setValue(1)
        self.bottone_sonda = QPushButton(testi.FUSIONE_SAMPLE)
        self.bottone_sonda.clicked.connect(self._sonda)
        for w in (self.etichetta_sonda, self.campo_sonda, self.bottone_sonda):
            comandi_2.addWidget(w)
        fascia_aiuto.osserva(self.bottone_sonda, self.fascia, testi.FUSIONE_SAMPLE,
                             testi.FUSIONE_HELP_SAMPLE)
        self.bottone_export = QPushButton(testi.FUSIONE_EXPORT)
        self.bottone_export.setEnabled(False)
        self.bottone_export.clicked.connect(self._esporta)
        comandi_2.addWidget(self.bottone_export)
        fascia_aiuto.osserva(self.bottone_export, self.fascia, testi.FUSIONE_EXPORT,
                             testi.FUSIONE_HELP_EXPORT)
        colonna.addLayout(comandi)
        colonna.addLayout(comandi_2)
        self.pannello_esito = PannelloEsito()
        colonna.addWidget(self.pannello_esito)

        self.etichetta_stato = QLabel("")
        colonna.addWidget(self.etichetta_stato)
        # Collegate DOPO la costruzione: `addItems` emette gia'
        # currentTextChanged sulla prima voce, e gli slot leggono widget che
        # a quel punto non esisterebbero ancora.
        self.tendina_modello.currentTextChanged.connect(lambda _t: self._su_modello_scelto())
        self.tendina_classe.currentTextChanged.connect(lambda _t: self._aggiorna_campi_del_passo())
        self._aggiorna_campi_del_passo()
        self._applica_stato("idle")

    def _collega_scorciatoie(self):
        """Le azioni appartengono alla PAGINA.

        `WidgetWithChildrenShortcut` scatta quando ad avere il focus e' il
        widget su cui l'azione e' stata aggiunta, o un suo discendente
        (gui/estrazione/comandi.py, in testa al modulo): la pagina e'
        l'antenato di tutti, ed e' lei a prendere il focus quando la scheda
        diventa corrente (MainWindow._su_scheda_cambiata). Che `,` `.` `-`
        vengano rubate a un campo di testo non e' un rischio di questo
        scope: un QLineEdit accetta il ShortcutOverride dei caratteri
        stampabili e se li tiene -- verificato con QTest scrivendo "0,1" nel
        campo degli indici GPU con le scorciatoie tutte collegate."""
        for chiave, (tasto, descrizione) in comandi_mod.COMANDI.items():
            azione = QAction(descrizione, self)
            azione.setShortcut(QKeySequence(tasto))
            azione.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            azione.triggered.connect(lambda _c, k=chiave: self._su_comando(k))
            self.addAction(azione)

    def _aggiorna_campi_del_passo(self):
        """Il morph lo chiede solo chi ce l'ha fra i propri campi (AMP e H2):
        mostrarlo per SAEHD prometterebbe una leva che quel merge non ha."""
        passo = _passo_merge(self.tendina_classe.currentText())
        ha_morph = _campo(passo, CHIAVE_MORPH) is not None
        self.etichetta_morph.setVisible(ha_morph)
        self.campo_morph.setVisible(ha_morph)
        # Il nome del .dat porta anche la CLASSE: cambiarla a mano cambia
        # il file da cercare, quindi la nota va rifatta di qui.
        self._aggiorna_nota_ripresa()

    # -- apertura ------------------------------------------------------------

    def apri(self, progetto):
        """Il progetto da fondere. Ferma la sessione che stesse girando: i
        frame di quella vecchia non sono questi."""
        self.ferma_servizio()
        self._progetto = Path(progetto)
        self.barra_preset.imposta(self._progetto)
        # `iterdir` solleva se la cartella c'e' ma non si elenca (permessi,
        # unita' di rete sparita): siamo dentro uno slot, e un'eccezione qui
        # ucciderebbe la finestra invece di mostrare una pagina vuota.
        try:
            # Stesso predicato e stesso ordine di pathex.get_image_paths:
            # `endswith` sul nome (non `suffix`) e ordinamento sul percorso
            # come STRINGA -- `sorted` su Path confronta in modo diverso su
            # Windows, e qui la posizione nella lista E' l'indice del
            # protocollo.
            percorsi = sorted((p for p in (self._progetto / "data_dst").iterdir()
                               if p.name.lower().endswith(ESTENSIONI)), key=str)
        except OSError:
            percorsi = []
        self.modello_frame.imposta(percorsi)
        # nome del file -> riga della pellicola, costruito una volta sola:
        # il rapporto e gli avvisi viaggiano per NOME, e cercarli scorrendo
        # il modello e' una scansione per ogni nome (l'elenco dei senza
        # volto di un data_dst lungo ne ha migliaia)
        self._indice_per_nome = {p.name: i for i, p in enumerate(percorsi)}
        self._totali = len(percorsi)
        self._rect = []
        self._cursore = 0
        self.timeline.imposta(0, ())
        self.pellicola.decodificatore.svuota()
        self.tendina_modello.clear()
        self.tendina_modello.addItems(saved_model_names(self._progetto / CARTELLA_MODELLI))
        ricordate = risposte_ricordate(self._progetto, NOME_PASSO_OCCUPANTE)
        if ricordate.get("model_name"):
            self.tendina_modello.setCurrentText(ricordate["model_name"])
        # La classe la dice il file del modello (`_su_modello_scelto`, gia'
        # passato di qui col cambio di testo sopra): quella ricordata vale
        # solo se il disco non sa rispondere -- un modello H2 fuso come AMP
        # fallisce dentro il figlio, minuti dopo il click.
        if ricordate.get("model") and not self._classe_del_modello():
            self.tendina_classe.setCurrentText(ricordate["model"])
        self._su_modello_scelto()
        # Il rapporto dell'ultima fusione di questo progetto, se c'e': resta
        # nascosto (lo mostra solo lo stato `done`), ma e' gia' pronto.
        self.rapporto.imposta(
            PannelloRapporto.leggi_da_file(
                self._progetto / "data_dst" / "merged" / NOME_RAPPORTO),
            self._indice_di)
        self._applica_stato("idle")

    def imposta_job_manager(self, gestore):
        self._job_manager = gestore

    def _classe_del_modello(self):
        """La classe salvata accanto al modello scelto, o None."""
        if self._progetto is None:
            return None
        return saved_model_class(self._progetto / CARTELLA_MODELLI,
                                 self.tendina_modello.currentText())

    def _su_modello_scelto(self):
        """Scegliere il modello sceglie anche la sua classe -- restando
        cambiabile a mano, perche' un `.dat` puo' sempre non esserci."""
        classe = self._classe_del_modello()
        if classe and self.tendina_classe.findText(classe) >= 0:
            self.tendina_classe.setCurrentText(classe)
        self._aggiorna_nota_ripresa()

    def _aggiorna_nota_ripresa(self):
        if self._progetto is None:
            return
        nome = self.tendina_modello.currentText()
        classe = self.tendina_classe.currentText()
        dat = self._progetto / CARTELLA_MODELLI / nome_file_sessione(nome, classe)
        self.nota_ripresa.setText(
            testi.FUSIONE_RESUME_FOUND if nome and classe and dat.exists() else "")

    # -- stati ---------------------------------------------------------------

    def stato_pagina(self):
        """Uno di: idle (nessuna sessione), loading (il figlio carica il
        modello), tuning (si regola un frame per volta), batch (il pool
        macina i restanti), done (il batch e' finito e il rapporto c'e')."""
        return self._stato

    def _applica_stato(self, stato):
        self._stato = stato
        in_sessione = stato in ("tuning", "batch", "done")
        self.pannello.abilita(stato in ("tuning", "done"))
        self.pannello.setEnabled(stato in ("tuning", "done"))
        self.barra_preset.setEnabled(stato in ("tuning", "done"))
        for _chiave, b in self.bottoni.items():
            b.setEnabled(in_sessione and stato != "batch")
        # Il piano non si applica senza almeno un keyframe: la scorciatoia
        # Ctrl+K resta viva anche a batch acceso, il bottone segue la
        # stessa regola degli altri (spento in batch).
        self.bottoni["piano"].setEnabled(in_sessione and stato != "batch" and bool(self._keyframes))
        self.spunta_interpola.setEnabled(in_sessione)
        self.bottone_sonda.setEnabled(stato in ("tuning", "done", "batch"))
        self.campo_sonda.setEnabled(stato in ("tuning", "done", "batch"))
        self.bottone_stop.setEnabled(stato == "batch")
        self.bottone_avvio.setEnabled(stato == "idle")
        self.bottone_fine.setEnabled(stato != "idle")
        # Il blocco d'avvio si legge una volta sola, quando la sessione
        # parte: dopo non e' solo inerte, e' un campo di testo che si
        # prenderebbe il focus (e con lui le lettere delle scorciatoie).
        for controllo in (self.tendina_modello, self.tendina_classe, self.campo_gpu,
                          self.campo_morph, self.campo_worker):
            controllo.setEnabled(stato == "idle")
        self.rapporto.setVisible(stato == "done")
        self.bottone_export.setEnabled(stato in ("idle", "done") and self._job_export is None
                                       and self._ha_png_fusi())

    def mostra_messaggio(self, testo):
        self.etichetta_stato.setText(testo)

    # -- la sessione ---------------------------------------------------------

    def _morph(self):
        """Il valore del campo, o il default del catalogo se chi scrive ha
        scritto qualcos'altro: il figlio riceve un JSON, e un campo di testo
        libero non deve poterlo rompere."""
        campo = _campo_globale(CHIAVE_MORPH)
        default = float(campo.default) if campo and campo.default is not None else 1.0
        try:
            valore = float(self.campo_morph.text().strip())
        except ValueError:
            return default
        if not numeri.numero_finito(valore):
            return default
        minimo, massimo = campo.valid_range if campo and campo.valid_range else (0.0, 1.0)
        return min(max(valore, minimo), massimo)

    def _parametri(self):
        classe = self.tendina_classe.currentText()
        return {"model": classe, "model_dir": self._progetto / CARTELLA_MODELLI,
                "input_dir": self._progetto / "data_dst",
                "output_dir": self._progetto / "data_dst" / "merged",
                "output_mask_dir": self._progetto / "data_dst" / "merged_mask",
                "aligned_dir": self._progetto / "data_dst" / "aligned",
                "force_model_name": self.tendina_modello.currentText(),
                "force_gpu_idxs": self.campo_gpu.text().strip() or None,
                "workers": self.campo_worker.value()}

    def _workdir(self):
        w = self._radice_e / "merge-session"
        w.mkdir(parents=True, exist_ok=True)
        return w

    def avvia_sessione(self, servizio=None, parametri=None):
        """Apre la sessione. `servizio` iniettabile: i test non avviano un
        processo con torch dentro."""
        if self._progetto is None or self._stato != "idle":
            return
        parametri = parametri or self._parametri()
        workdir = self._workdir()
        if servizio is None:
            from gui.estrazione.trasporto import TrasportoAsincrono
            risposte = workdir / "risposte.json"
            risposte.write_text(json.dumps({PROMPT_MORPH: self._morph()}), encoding="utf-8")
            progresso = workdir / "progresso.jsonl"
            if progresso.exists():
                progresso.unlink()
            ambiente = {"DFL_ANSWERS_FILE": str(risposte), "DFL_PROGRESS_FILE": str(progresso)}
            self._trasporto = TrasportoAsincrono(
                workdir, su_evento=self.su_evento,
                comando_servizio=lambda w: avvio_mod.comando_servizio(w, parametri),
                ambiente=ambiente)
            servizio = servizio_mod.Servizio(self._trasporto)
            self._tail_progresso = EventTail(str(progresso), parent=self)
            self._tail_progresso.event.connect(self.pila.applica)
        self.servizio = servizio
        # Visibile a chi_occupa()/try_start() come se fosse un job: il
        # processo del servizio non e' un Job, quindi active_jobs() non lo
        # vede, e senza questo un passo avviato dalla lista Steps
        # scriverebbe su `merged` mentre la sessione ci scrive.
        self._identita_occupante = identita_workspace(self._progetto)
        self._occupante = PassoFittizio(NOME_PASSO_OCCUPANTE,
                                        ("frame_dst", "faceset_dst", "modello"),
                                        ("merged", "merged_mask"), ())
        registra_occupante(self._identita_occupante, self._occupante)
        try:
            ricorda_risposte(self._progetto, NOME_PASSO_OCCUPANTE,
                             {"model_name": parametri["force_model_name"],
                              "model": parametri["model"]})
        except OSError:
            pass        # un progetto senza project.json scrivibile non e' un guasto
        self.pila.mostra_avvio(testi.FUSIONE_LOADING)
        self._applica_stato("loading")
        self.servizio.stato(self._su_stato)     # il primo comando avvia il processo

    def ferma_servizio(self):
        """Pubblico e idempotente: e' il metodo che ogni percorso di uscita
        chiama -- il bottone «End session», `apri`, un `error` del figlio,
        `su_chiusura_scheda` (chiusura della scheda) e `MainWindow.closeEvent`.
        I tre percorsi di chiusura passano tutti di qui, ed e' qui che si
        aspetta il salvataggio della sessione."""
        if self.servizio is not None:
            # Staccato PRIMA dell'attesa: durante il ciclo annidato puo'
            # arrivare un evento che rientra qui, e deve trovare il campo
            # gia' vuoto invece di chiudere una seconda volta.
            servizio, self.servizio = self.servizio, None
            try:
                self._aspetta_la_chiusura(servizio)
            finally:
                servizio.ferma()
        self._smonta()

    def _smonta(self):
        """Cio' che resta da smontare quando il figlio non serve piu':
        trasporto, coda del progresso, occupante, stato. Non parla col
        figlio -- e' la parte che vale sia che lo si sia chiuso noi, sia
        che sia uscito da solo."""
        self._trasporto = None
        if self._tail_progresso is not None:
            self._tail_progresso.stop()
            self._tail_progresso = None
        if self._occupante is not None:
            libera_occupante(self._identita_occupante, self._occupante)
            self._occupante = None
            self._identita_occupante = None
        self.pila.togli_avvio()
        self._keyframes = []
        self.etichetta_piano.setText("")
        self._timer_anteprima.stop()
        self.striscia_sonda.svuota()
        self._applica_stato("idle")

    def _aspetta_la_chiusura(self, servizio):
        """Chiede `chiudi` e aspetta la risposta, al massimo
        `ATTESA_CHIUSURA_MS`.

        Non e' cortesia: il figlio salva `<nome>_<Classe>_merger_session.dat`
        (`nome_file_sessione`) solo mentre serve questo comando, e ucciderlo
        prima della risposta butterebbe via la ripresa che la pagina
        promette. La pagina si spegne per la durata dell'attesa -- il ciclo
        annidato continua a consegnare eventi, e un click su «Start session»
        a sessione morente non deve poter partire.

        Un figlio gia' morto non fa aspettare, e sono due strade diverse:
        se una richiesta era in sospeso quando e' morto, il trasporto le
        consegna da se' una risposta di guasto (`_su_guasto`); se non ce
        n'era nessuna, `invia_subito` vede il canale senza processo vivo,
        risponde subito e NON lo riavvia -- prima riaccendeva un processo
        con un modello dentro solo per consegnargli il `chiudi`, e la
        pagina restava spenta fino ad ATTESA_CHIUSURA_MS (voce 3.83)."""
        if self._in_chiusura:
            return
        self._in_chiusura = True
        fatto = []
        ciclo = QEventLoop()
        scadenza = QTimer(self)
        scadenza.setSingleShot(True)
        scadenza.timeout.connect(ciclo.quit)

        def _risposto(_risposta):
            fatto.append(True)
            ciclo.quit()

        try:
            servizio.chiudi(_risposto)
            if not fatto:
                self.setEnabled(False)
                scadenza.start(ATTESA_CHIUSURA_MS)
                ciclo.exec_()
        finally:
            scadenza.stop()
            self.setEnabled(True)
            self._in_chiusura = False

    def su_chiusura_scheda(self):
        """`removeTab()`+`setParent()` non consegnano nessun closeEvent:
        senza questa chiamata esplicita la sessione (e il suo pool) resterebbe
        viva dietro una scheda chiusa."""
        self._scheda_chiusa = True
        self.ferma_servizio()

    def su_apertura_scheda(self):
        """La scheda torna in primo piano dopo essere stata chiusa: si
        rilegge il disco, che nel frattempo puo' aver preso un modello
        appena addestrato o dei frame rinominati da un sort.

        Solo se era stata chiusa davvero: alla prima costruzione la finestra
        ha gia' chiamato `apri`, e su un `data_dst` da decine di migliaia di
        fotogrammi elencarlo due volte di fila si sente."""
        if self._scheda_chiusa and self._stato == "idle" and self._progetto is not None:
            self.apri(self._progetto)
        self._scheda_chiusa = False

    #override
    def closeEvent(self, evento):
        self.ferma_servizio()
        super().closeEvent(evento)

    # -- eventi dal figlio ---------------------------------------------------

    def su_evento(self, evento):
        if not isinstance(evento, dict):
            return
        op = evento.get("op")
        if op == "pronto":
            self._su_pronto(evento)
        elif op == "frame_pronto":
            self._su_frame_pronto(evento)
        elif op == "avanzamento":
            self._su_avanzamento(evento)
        elif op == "rapporto":
            self._su_rapporto(evento)
        elif op == "error":
            self._su_errore(evento)
        elif op == "chiudi" and evento.get("id") is None:
            # `chiudi` senza `id` non risponde a nessun comando: e' il
            # figlio che si e' chiuso DA SOLO. La sessione e' gia' salvata
            # e il processo e' gia' uscito, quindi non c'e' niente da
            # chiedergli e niente da aspettare.
            self._su_chiusura_spontanea()

    def _su_pronto(self, evento):
        totali = evento.get("frame_totali")
        if not (isinstance(totali, int) and not isinstance(totali, bool)
                and numeri.intero_qt_utilizzabile(totali)):
            return
        if totali != self.modello_frame.rowCount():
            # Le due enumerazioni non coincidono: ogni indice del protocollo
            # punterebbe a un altro fotogramma, e non c'e' modo di
            # accorgersene guardando lo schermo. Si chiude invece di
            # proseguire in silenzio.
            self.mostra_messaggio(testi.FUSIONE_FRAME_DISALLINEATI)
            self.ferma_servizio()
            return
        self._totali = totali
        self.campo_sonda.setRange(1, max(1, self._totali))
        self.campo_sonda.setValue(min(SONDA_DEFAULT, max(1, self._totali)))
        avvisi = evento.get("avvisi") if isinstance(evento.get("avvisi"), dict) else {}
        # Dentro `avvisi` sono NOMI di file: sono indici solo nelle risposte
        # ai comandi, che portano lo stato del nucleo.
        nomi = avvisi.get("senza_volto")
        senza_volto = {self._indice_di(n) for n in nomi} if isinstance(nomi, list) else set()
        senza_volto.discard(None)
        self.timeline.imposta(totali, senza_volto)
        for i in senza_volto:
            self.modello_frame.imposta_stato(i, tl.STATO_SENZA_VOLTO)
        # I frame gia' fusi di una sessione ripresa: senza questi la
        # timeline di una fusione finita ripartirebbe tutta da fare, con la
        # clessidra sopra un fotogramma che sta gia' sul disco. `segna_fatto`
        # preserva il rosso di chi non ha un volto -- i due elenchi si
        # sovrappongono sempre, perche' un frame senza volto viene copiato e
        # quindi e' fatto.
        fatti = evento.get("fatti_idx")
        for i in (fatti if isinstance(fatti, list) else []):
            if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < totali:
                self.timeline.segna_fatto(i)
                if i not in senza_volto:
                    self.modello_frame.imposta_stato(i, tl.STATO_FATTO)
        rect = evento.get("rect")
        self._rect = rect if isinstance(rect, list) and len(rect) == totali else [None] * totali
        if isinstance(evento.get("cfg"), dict):
            self.pannello.imposta_cfg(evento["cfg"])
        self.pila.togli_avvio()
        self._applica_stato("tuning")
        self._imposta_cursore(evento.get("cursore", 0))
        # Dopo `_imposta_cursore`, che scrive lui stesso sulla riga di stato:
        # l'esito della ripresa e' la notizia piu' importante delle due, e
        # deve restare quella che si legge.
        ripresa = evento.get("ripresa")
        if ripresa == "azzerata_per_iter":
            self.nota_ripresa.setText(testi.FUSIONE_RESUME_STALE)
        elif ripresa == "non_corrisponde":
            self.mostra_messaggio(testi.FUSIONE_SESSIONE_SCARTATA)

    def _indice_di(self, nome):
        """La riga della pellicola che porta quel nome di file, o None."""
        return self._indice_per_nome.get(nome)

    def _su_frame_pronto(self, evento):
        idx = evento.get("idx")
        if not isinstance(idx, int) or isinstance(idx, bool) or not (0 <= idx < self._totali):
            return
        # Un frame senza volto resta SENZA_VOLTO: segna_fatto lo preserva.
        self.timeline.segna_fatto(idx)
        self.modello_frame.imposta_stato(idx, tl.STATO_FATTO)
        self.striscia_sonda.segna_pronto(idx)
        if idx == self._cursore:
            self._ridisegna()

    def _su_avanzamento(self, evento):
        fatti, totali, eta = evento.get("fatti"), evento.get("totali"), evento.get("eta_s")
        if not (isinstance(fatti, int) and isinstance(totali, int)):
            return
        eta = eta if isinstance(eta, int) and numeri.intero_qt_utilizzabile(eta) else None
        self.mostra_messaggio(testi.fusione_avanzamento(fatti, totali, eta))

    def _su_rapporto(self, evento):
        self.rapporto.imposta(evento, self._indice_di)
        self._applica_stato("done")
        ms = evento.get("ms_per_frame")
        if numeri.numero_finito(ms):
            self.mostra_messaggio(testi.fusione_ms_per_frame(float(ms)))

    def _su_errore(self, evento):
        """Un `error` senza `id`, cioe' non la risposta a un comando.

        PRIMA di `pronto` e' fatale -- la sessione non e' mai nata, e non
        c'e' niente su cui restare. DOPO, e' la caduta di un processo di
        compositing su un frame: nel pool quel frame torna in coda
        (`SessioneMerge.su_ritorno`) e gli altri processi continuano,
        esattamente come nella finestra `cv2`. Fermare la sessione qui
        vorrebbe dire buttare una fusione di ore per un frame."""
        codice, motivo = evento.get("codice"), evento.get("motivo")
        if self._stato not in ("tuning", "batch", "done"):
            self._guasto_fatale(codice, motivo)
            return
        self.mostra_messaggio(testi.fusione_guasto(codice, motivo))
        idx = evento.get("idx")
        if isinstance(idx, int) and not isinstance(idx, bool) and 0 <= idx < self._totali:
            self.timeline.segna_da_fare(idx)
            self.modello_frame.imposta_stato(idx, tl.STATO_DA_FARE)

    def _su_chiusura_spontanea(self):
        """Il figlio e' uscito per conto suo.

        Succede quando muore l'ULTIMO processo di compositing: `on_tick`
        non ha piu' nessun client, esce dal ciclo, `on_clients_finalized`
        salva la sessione e manda il suo `chiudi`. Senza questo ramo la
        pagina restava in `batch` con un servizio morto e l'avanzamento
        fermo, senza dire niente -- e da quando un `error` dopo `pronto`
        non ferma piu' la sessione (e' la caduta di un frame, non della
        fusione) non c'era piu' nessun'altra strada per accorgersene."""
        if self.servizio is None:
            return
        servizio, self.servizio = self.servizio, None
        servizio.ferma()          # chiude il canale: il processo e' gia' uscito
        self._smonta()
        self.mostra_messaggio(testi.FUSIONE_SERVIZIO_TERMINATO)

    def _guasto_fatale(self, codice, motivo):
        """Il guasto che spegne la sessione: lo dice e chiude."""
        self.mostra_messaggio(testi.fusione_guasto(codice, motivo))
        self.ferma_servizio()

    # -- risposte ai comandi -------------------------------------------------

    def _su_stato(self, risposta):
        """La risposta di stato/vai/cfg/propaga/batch: la stessa forma per
        tutte e cinque, quindi un solo slot."""
        if not isinstance(risposta, dict):
            # nessuna risposta valida a un comando: e' il tubo, non un frame
            self._guasto_fatale(self.servizio.ultimo_codice if self.servizio else None,
                                self.servizio.ultimo_errore if self.servizio else None)
            return
        self._assorbi_keyframes(risposta)
        if self._stato == "batch" and risposta.get("batch") is False:
            self._applica_stato("tuning")
        if isinstance(risposta.get("cfg"), dict):
            self.pannello.imposta_cfg(risposta["cfg"])
        cursore = risposta.get("cursore")
        if isinstance(cursore, int) and 0 <= cursore < max(1, self._totali):
            self._imposta_cursore(cursore)

    def _imposta_cursore(self, idx):
        if not (isinstance(idx, int) and not isinstance(idx, bool)
                and 0 <= idx < max(1, self._totali)):
            return
        self._cursore = idx
        self.timeline.imposta_cursore(idx)
        self.pellicola.scorri_a(idx)
        self._ridisegna()
        fatti = int((self.timeline._stati == tl.STATO_FATTO).sum()) if self._totali else 0
        if self._stato in ("tuning", "done"):
            self.mostra_messaggio(testi.fusione_stato(idx + 1, self._totali, fatti))
        self._aggiorna_bottone_keyframe()

    def _ridisegna(self):
        """I frame non viaggiano nel tubo: il pool li scrive su disco e
        l'evento arriva DOPO la scrittura, quindi qui si legge il PNG."""
        p = self.modello_frame.data(self.modello_frame.index(self._cursore, 0), pl.RUOLO_PERCORSO)
        if p is None:
            return
        fuso = self._progetto / "data_dst" / "merged" / (p.stem + ".png")
        maschera = self._progetto / "data_dst" / "merged_mask" / (p.stem + ".png")
        # Lo stato si legge dal MODELLO della pellicola, non dalla timeline:
        # quella e' dimensionata dall'evento `pronto`, e prima che arrivi --
        # la pagina mostra gia' i frame del disco -- un indice valido qui
        # sarebbe fuori dal suo array.
        stato = self.modello_frame.data(self.modello_frame.index(self._cursore, 0), pl.RUOLO_STATO)
        fatto = stato == tl.STATO_FATTO
        originale, fuso_img, maschera_img = tela_mod.carica_tre(
            p, fuso if fatto else None, maschera if fatto else None)
        rect = self._rect[self._cursore] if self._cursore < len(self._rect) else None
        self.tela.mostra(originale, fuso_img, maschera_img, rect, in_attesa=not fatto)
        self.lente.mostra(fuso_img if fuso_img is not None else originale, rect)

    def _vai(self, idx):
        if self.servizio is None or self._stato not in ("tuning", "done"):
            return
        self.servizio.vai(int(idx), self._su_stato)

    def _su_cfg_cambiata(self, cfg):
        if self.servizio is None or self._stato not in ("tuning", "done"):
            return
        self.timeline.segna_da_fare(self._cursore)
        self.modello_frame.imposta_stato(self._cursore, tl.STATO_DA_FARE)
        self._applica_stato("tuning")
        self.servizio.cfg(cfg, self._su_stato)
        self._timer_anteprima.start()

    def _su_preset_scelto(self, cfg):
        if self.servizio is None or self._stato not in ("tuning", "done") or not isinstance(cfg, dict):
            return
        self.pannello.imposta_cfg(cfg)          # senza emettere: la cfg parte una volta sola
        self._su_cfg_cambiata(self.pannello.cfg())

    def _ferma_batch(self):
        if self.servizio is not None:
            self.servizio.batch(False, self._su_stato)
        if self._stato == "batch":
            self._applica_stato("tuning")

    # -- dispatch ------------------------------------------------------------

    def _precedente(self, propaga=False, primo=False):
        if self.servizio is None or self._stato not in ("tuning", "done"):
            return
        if propaga:
            self.servizio.propaga("indietro", "tutti" if primo else "prossimo", self._su_stato)
        else:
            self.servizio.vai(0 if primo else max(0, self._cursore - 1), self._su_stato)

    def _successivo(self, propaga=False, ultimo=False):
        if self.servizio is None or self._stato not in ("tuning", "done"):
            return
        if propaga:
            self.servizio.propaga("avanti", "tutti" if ultimo else "prossimo", self._su_stato)
        else:
            self.servizio.vai(min(self._totali - 1, self._cursore + 1), self._su_stato)

    def _su_comando(self, chiave):
        """Instrada tramite `_DISPATCH`. Una chiave che non e' nella tabella
        delle scorciatoie esce invece di sollevare KeyError: qui ci si
        arriva da uno slot Qt, e li' un'eccezione e' un qFatal."""
        if chiave not in comandi_mod.CHIAVI_INSTRADATE:
            return
        self._DISPATCH[chiave](self)

    def _batch(self):
        """Lo stesso tasto accende e spegne, come nella finestra `cv2`."""
        if self._stato == "batch":
            self._ferma_batch()
            return
        if self.servizio is None or self._stato not in ("tuning", "done"):
            return
        self._applica_stato("batch")
        self.servizio.batch(True, self._su_stato)

    def _salva_sessione(self):
        if self.servizio is None or self._stato not in ("tuning", "done"):
            return
        self.servizio.salva_sessione(lambda _r: None)

    def _assorbi_keyframes(self, risposta):
        """I keyframe che una risposta porta: stessa forma in `_su_stato` e
        in `_su_anteprima_piano`, un solo posto che li applica alla
        timeline, al bottone e all'abilitazione del piano."""
        keyframes = risposta.get("keyframes")
        if isinstance(keyframes, list):
            self._keyframes = [k for k in keyframes if isinstance(k, int)]
            self.timeline.imposta_keyframes(self._keyframes)
            self._aggiorna_bottone_keyframe()
            self._applica_stato(self._stato)

    def _aggiorna_bottone_keyframe(self):
        acceso = self._cursore in self._keyframes
        self.bottoni["keyframe"].setText(testi.FUSIONE_KEYFRAME_CLEAR if acceso
                                         else testi.FUSIONE_KEYFRAME)

    def _keyframe(self):
        if self.servizio is None or self._stato not in ("tuning", "done"):
            return
        acceso = self._cursore not in self._keyframes
        self.servizio.keyframe(self._cursore, acceso, self._su_stato_e_anteprima)

    def _su_stato_e_anteprima(self, risposta):
        self._su_stato(risposta)
        self._timer_anteprima.start()

    def _chiedi_anteprima_piano(self):
        if self.servizio is None or self._stato not in ("tuning", "done", "batch"):
            return
        if not self._keyframes:
            self.etichetta_piano.setText(testi.fusione_anteprima_piano(None))
            return
        self.servizio.piano(self.spunta_interpola.isChecked(), True, self._su_anteprima_piano)

    def _su_anteprima_piano(self, risposta):
        if not isinstance(risposta, dict):
            return
        n = risposta.get("da_rifare")
        self.etichetta_piano.setText(testi.fusione_anteprima_piano(
            n if isinstance(n, int) and not isinstance(n, bool) and n >= 0 else None))
        self._assorbi_keyframes(risposta)

    def _piano(self):
        """Il piano si applica QUI e solo qui: mai da solo a un cambio di
        slider. A batch acceso il pool riprende da se' i frame cambiati."""
        if self.servizio is None or self._stato not in ("tuning", "done", "batch") \
                or not self._keyframes:
            return
        self.servizio.piano(self.spunta_interpola.isChecked(), False, self._su_piano_applicato)

    def _su_piano_applicato(self, risposta):
        if not isinstance(risposta, dict):
            self._guasto_fatale(self.servizio.ultimo_codice if self.servizio else None,
                                self.servizio.ultimo_errore if self.servizio else None)
            return
        # i frame cambiati sono tornati da fare: la timeline li rilegge da
        # fatti_idx, che la risposta di stato porta sempre
        fatti = risposta.get("fatti_idx")
        if isinstance(fatti, list):
            fatti = {f for f in fatti if isinstance(f, int)}
            for idx in range(self._totali):
                if idx in fatti:
                    self.timeline.segna_fatto(idx); self.modello_frame.imposta_stato(idx, tl.STATO_FATTO)
                else:
                    self.timeline.segna_da_fare(idx); self.modello_frame.imposta_stato(idx, tl.STATO_DA_FARE)
        n = risposta.get("da_rifare")
        self.mostra_messaggio(testi.fusione_anteprima_piano(
            n if isinstance(n, int) and not isinstance(n, bool) else None))
        self.etichetta_piano.setText("")
        if self._stato == "done" and isinstance(n, int) and n > 0:
            self._applica_stato("tuning")
        self._su_stato(risposta)

    def _percorso_fuso(self, idx):
        """Il PNG che la fusione di quell'indice scrive in `merged`, o None
        fuori dal modello -- stessa regola di `_ridisegna`."""
        if self._progetto is None or not (0 <= idx < self.modello_frame.rowCount()):
            return None
        p = self.modello_frame.data(self.modello_frame.index(idx, 0), pl.RUOLO_PERCORSO)
        if p is None:
            return None
        return self._progetto / "data_dst" / "merged" / (p.stem + ".png")

    def _sonda(self):
        if self.servizio is None or self._stato not in ("tuning", "done", "batch"):
            return
        self.servizio.sonda(self.campo_sonda.value(), self._su_sonda)

    def _su_sonda(self, risposta):
        if not isinstance(risposta, dict):
            return
        indici = risposta.get("indici")
        if isinstance(indici, list):
            self.striscia_sonda.imposta(indici, self._percorso_fuso)
            # I gia' fusi con questa cfg non tornano da fare nel nucleo e
            # non manderanno un frame_pronto: la striscia li segna subito,
            # e restano fuori dal giro che rimette gli altri «da fare».
            gia_fatti = set(risposta.get("gia_fatti") or [])
            for idx in gia_fatti:
                self.striscia_sonda.segna_pronto(idx)
            for idx in self.striscia_sonda.indici:
                if idx < self._totali and idx not in gia_fatti:
                    self.timeline.segna_da_fare(idx)
                    self.modello_frame.imposta_stato(idx, tl.STATO_DA_FARE)
        self._su_stato(risposta)

    # -- export ----------------------------------------------------------

    def _ha_png_fusi(self):
        if self._progetto is None:
            return False
        cartella = self._progetto / "data_dst" / "merged"
        try:
            return any(p.suffix.lower() == ".png" for p in cartella.iterdir())
        except OSError:
            return False

    def _apri_dialogo_export(self):
        dialogo = DialogoExport(riferimento="data_dst.*", parent=self)
        if dialogo.exec_() != QDialog.Accepted:
            return None
        return dialogo.scelta()

    def _avvisa_con_dialogo(self, titolo, testo):
        QMessageBox.warning(self, titolo, testo)

    def _esporta(self):
        """Lancia uno dei quattro passi «8) merged to …» col JobManager: lo
        stesso passo della lista Steps, cosi' conflicts.py lo vede occupare
        `merged`/`risultato` come farebbe da li'."""
        if self._progetto is None or self._job_manager is None or self._job_export is not None:
            return
        if not self._ha_png_fusi():
            self.mostra_messaggio(testi.FUSIONE_EXPORT_NO_FRAMES)
            return
        scelta = self.apri_dialogo_export()
        if scelta is None:
            return
        contenitore, lossless, bitrate = scelta
        passo = esporta_mod.passo_per(contenitore, lossless)
        if passo is None:
            return
        try:
            job = self._job_manager.try_start(passo, esporta_mod.risposte_per(passo, bitrate),
                                              self._progetto)
        except StepConflict as errore:
            self.avvisa(testi.TITLE_STEP_BUSY, str(errore))
            return
        if job is None:
            return
        self._job_export = job
        self._contenitore_export = contenitore
        job.finished.connect(self._su_export_finito)
        self.pannello_esito.imposta(None, None)
        self.mostra_messaggio(testi.FUSIONE_EXPORT_RUNNING)
        self._applica_stato(self._stato)

    def _su_export_finito(self, codice):
        self._job_export = None
        if not isinstance(codice, int) or codice != 0:
            self.mostra_messaggio(
                testi.FUSIONE_EXPORT_FAILED % (codice if isinstance(codice, int) else -1))
            self._applica_stato(self._stato)
            return
        percorso = self._progetto / ("result.%s" % self._contenitore_export)
        self.pannello_esito.imposta(esporta_mod.esito_del_file(percorso), percorso)
        self.mostra_messaggio(self.pannello_esito.etichetta.text())
        self._applica_stato(self._stato)

    def _zoom(self, delta):
        self.tela.zoom(delta)

    def _alterna_vista(self, vista):
        self.tela.imposta_vista("merged" if self.tela.vista() == vista else vista)

    _DISPATCH = {
        "precedente": lambda self: self._precedente(),
        "primo": lambda self: self._precedente(primo=True),
        "precedente_propaga": lambda self: self._precedente(propaga=True),
        "primo_propaga": lambda self: self._precedente(propaga=True, primo=True),
        "successivo": lambda self: self._successivo(),
        "batch": lambda self: self._batch(),
        "successivo_propaga": lambda self: self._successivo(propaga=True),
        "ultimo_propaga": lambda self: self._successivo(propaga=True, ultimo=True),
        "zoom_meno": lambda self: self._zoom(-0.1),
        "zoom_piu": lambda self: self._zoom(0.1),
        "vista_maschera": lambda self: self._alterna_vista("mask"),
        "vista_originale": lambda self: self._alterna_vista("original"),
        "salva_sessione": lambda self: self._salva_sessione(),
        "keyframe": lambda self: self._keyframe(),
        "piano": lambda self: self._piano(),
    }
