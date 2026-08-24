"""La pagina di estrazione: composizione, niente logica propria.

Segue il PROGETTO, non il job -- come gui/faceset/pagina.py, e per la
stessa ragione: la scheda e' una sola, e cambiare progetto o lato cambia
cio' che mostra.

In alto la barra (lato, operazione automatica, parametri, sessione
manuale, ri-estrazione della selezione), al centro la `Tela`, in basso i
sei filtri del rapporto e la `Pellicola`.
"""
import tempfile
from pathlib import Path

from PyQt5.QtCore import QRunnable, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QImageReader, QPixmap
from PyQt5.QtWidgets import (QApplication, QButtonGroup, QCheckBox,
                             QHBoxLayout, QLabel, QMessageBox, QPushButton,
                             QVBoxLayout, QWidget)

from gui import testi
from gui import theme
from gui.estrazione import avvio as avvio_mod
from gui.estrazione import azioni as azioni_mod
from gui.estrazione import indice as indice_mod
from gui.estrazione import maschera as maschera_mod
from gui.estrazione import servizio as servizio_mod
from gui.estrazione import volti as volti_mod
from gui.estrazione.comandi import CHIAVI_FRECCE, COMANDI, ColonnaComandi
from gui.estrazione.modello import RUOLO_PERCORSO, ModelloFrame
from gui.estrazione.pellicola import Pellicola
from gui.estrazione.rapporto_vivo import LettoreIncrementale
from gui.estrazione.tela import Tela
from gui.estrazione.trasporto import TrasportoAsincrono
from gui.execution.conflicts import libera_occupante, registra_occupante
from gui.execution.jobs import StepConflict
from gui.faceset import cache as cache_mod
from gui.faceset import cestino as cestino_mod
from gui.faceset.conflitti import PassoFittizio, artefatto_di, chi_occupa
from gui.faceset.decodifica import Decodificatore
from gui.faceset.dialogo import DialogoOperazione
from gui.faceset.indice import elenca as elenca_cartella
from gui.faceset.indice import mappa_per_fotogramma
from gui.faceset.progresso import PilaProgresso
from gui.progetti import identita_workspace, ricorda_risposte, risposte_ricordate
from mainscripts import MotoriCatalog

ESTENSIONI = (".png", ".jpg", ".jpeg")

# Il lato dell'anteprima di revisione (fuori dalla sessione manuale): non
# uno dei LATI della pellicola (gui/faceset/decodifica.py), che sono
# miniature -- qui serve leggere i dettagli di un frame 4K.
LATO_ANTEPRIMA = 1024

# Lo stesso verde con cui la finestra di dettaglio disegna i poligoni
# "include" e con cui la griglia della cura evidenzia i fratelli: e' il
# colore che in questa interfaccia vuol dire "questo pezzo di immagine e'
# il volto".
COLORE_MASCHERA = QColor(80, 220, 140)

# Dove vive la cache del rapporto, fuori dal progetto: <radice_e>/extract-cache/<id>.
# id_cartella() e' l'unica funzione che lo sa (gui/faceset/cache.py) -- si
# riusa, non si ricalcola.
CACHE_SOTTOCARTELLA = "extract-cache"

# Un pixel per pressione, come le frecce di Extractor.py.
_DELTA_FRECCE = {
    "muovi-sinistra": (-1, 0),
    "muovi-destra": (1, 0),
    "muovi-su": (0, -1),
    "muovi-giu": (0, 1),
}

CHIAVE_AUTO = "auto"
CHIAVE_MANUALE = "manuale"
CHIAVE_AUTO_CORREZIONE = "auto-con-correzione"
CHIAVE_RIESTRAI = "riestrai-selezione"
# Una costante invece della stringa letterale ripetuta a mano tre volte in
# _su_job_finito -- la chiave del filtro "No face" (gui/estrazione/indice.py::FILTRI).
FILTRO_SENZA_VOLTO = "senza-volto"

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

# Le chiavi con cui il progetto ricorda le tre scelte della sessione
# manuale. Deliberatamente NON "detector"/"landmarker": quelle sono le
# chiavi dei campi del form dei passi automatici, e il form ricorda le
# ETICHETTE (gui/forms.py::set_remembered_values) mentre qui si ricordano
# le CHIAVI del registro. I passi MANUAL oggi non hanno quei campi -- se
# un ciclo futuro glieli aggiungesse, due significati diversi finirebbero
# nella stessa voce di project.json, e il dialogo precaricherebbe una
# tendina con una stringa che non e' una sua voce.
CHIAVE_MEMORIA_RILEVATORE = "manual-detector"
CHIAVE_MEMORIA_ALLINEATORE = "manual-landmarker"
CHIAVE_MEMORIA_TIENI = "manual-keep-models-in-memory"

# Il nome dell'attributo letto con getattr(...) da _su_volti -- non un
# testo a schermo, ma tests_gui/test_testi_soltanto_da_un_posto.py non
# distingue una stringa-chiave da un letterale visibile per sola
# ispezione dell'AST; promuoverla a costante di modulo la fa uscire dalla
# rete, la stessa uscita gia' usata da gui/faceset/pagina.py::CHIAVE_SORT.
_ATTRIBUTO_ULTIMO_STDERR = "ultimo_stderr"


def _radice_e_predefinita():
    """<pacchetto>/_internal/_e, calcolata dalla posizione di questo file
    -- lo stesso posto che MainWindow passa esplicitamente
    (`self._dfl_root.parent / "_e"`), qui ricavato per chi costruisce la
    pagina senza passare radice_e (i test)."""
    return Path(__file__).resolve().parent.parent.parent.parent / "_e"


def _pesi_mancanti(dfl_root, motori):
    """Le chiavi, fra `motori`, i cui pesi non sono sotto facelib/ in
    questa installazione. `dfl_root` puo' essere None -- la pagina
    costruita senza passare da avvio.configura(), come la maggior parte
    dei test -- e in quel caso non sappiamo se i pesi ci sono: meglio non
    disabilitare una scelta valida che disabilitarne una per un dato che
    manca. Un `OSError` (permessi, percorso strano) e' trattato allo
    stesso modo, per la stessa ragione: fallire chiuso qui spegnerebbe la
    tendina invece di limitarsi a non saperlo.

    Chiamata una sola volta, alla costruzione della pagina -- non e' un
    controllo da ripetere a ogni ridisegno della barra."""
    if dfl_root is None:
        return frozenset()
    try:
        cartella = Path(dfl_root) / "facelib"
        return frozenset(m.key for m in motori
                         if not all((cartella / nome).exists() for nome in m.pesi))
    except OSError:
        return frozenset()


def _selettore_motori(motori, chiave_predefinita, aiuto, mancanti=frozenset()):
    """Una tendina sul registro: mostra la `label`, porta la `key`, e
    l'aiuto per voce e' il `help` del registro come nei form del catalogo
    (gui/forms.py::_build_choice). Nessuna stringa scritta qui.

    Una voce in `mancanti` resta nella tendina -- sparire senza spiegazione
    sarebbe peggio -- ma disabilitata, con un aiuto che dice perche'
    (`testi.estrazione_pesi_mancanti_tip`)."""
    selettore = theme.tendina()
    selettore.setToolTip(aiuto)
    modello = selettore.model()
    for i, motore in enumerate(motori):
        selettore.addItem(motore.label, motore.key)
        if motore.key in mancanti:
            selettore.setItemData(i, testi.estrazione_pesi_mancanti_tip(motore), Qt.ToolTipRole)
            modello.item(i).setEnabled(False)
        else:
            selettore.setItemData(i, motore.help, Qt.ToolTipRole)
    indice = selettore.findData(chiave_predefinita)
    selettore.setCurrentIndex(indice if indice >= 0 else 0)
    return selettore


def _chiave_valida(valore, chiavi, predefinita):
    """La chiave di un motore, o il default: un project.json scritto a mano
    -- o un motore tolto dal registro -- non deve sollevare dentro
    l'ingresso in sessione."""
    return valore if valore in chiavi else predefinita


# Cosa conta come "no" in un project.json scritto a mano. `bool()` da solo
# non basta: `bool("false")` e' True, quindi una spunta scritta come
# stringa verrebbe letta come messa -- e la scelta di liberare la VRAM
# sarebbe l'unica che si perde in silenzio, cioe' quella che l'utente ha
# preso per far posto a un training.
_SPUNTA_FALSA = ("false", "0", "no", "off", "")


def _spunta_ricordata(valore):
    """Il valore ricordato della spunta, normalizzato. Assente = messa, il
    comportamento di sempre."""
    if valore is None:
        return True
    if isinstance(valore, str):
        return valore.strip().lower() not in _SPUNTA_FALSA
    return bool(valore)


def _valori_con_default(passo, risposte):
    """I campi del passo coi loro default, sovrascritti dalle risposte
    dell'utente. Il protocollo del servizio vuole un valore per campo,
    sempre -- a differenza di un job CLI, dove un campo non toccato
    semplicemente non compare in riga di comando e il valore di bordo lo
    fornisce lo script."""
    valori = dict((f.key, f.default) for f in passo.fields)
    valori.update(risposte)
    return valori


def _dimensione_nativa(percorso, pixmap):
    """La dimensione del frame su DISCO, che e' lo spazio in cui il rapporto
    scrive i rettangoli. L'anteprima e' decodificata a LATO_ANTEPRIMA,
    quindi le due non coincidono, e la differenza E' lo scarto del
    rettangolo.

    Solo l'intestazione del file, non i pixel: QImageReader.size() non
    decodifica. Se non torna una dimensione dello STESSO orientamento del
    pixmap si ripiega su None -- il lettore delle miniature applica
    setAutoTransform, e un frame ruotato dall'EXIF darebbe una dimensione
    trasposta: meglio un rettangolo scalato come prima che uno girato.
    """
    dimensione = QImageReader(str(percorso)).size()
    if not dimensione.isValid() or dimensione.width() <= 0 or dimensione.height() <= 0:
        return None
    if pixmap.width() <= 0 or pixmap.height() <= 0:
        return None
    atteso = dimensione.width() / float(dimensione.height())
    reale = pixmap.width() / float(pixmap.height())
    if abs(atteso - reale) > 0.01:
        return None
    return dimensione.width(), dimensione.height()


class _LavoroMappa(QRunnable):
    """Costruisce la mappa fotogramma -> volti fuori dal thread di Qt.

    Solo funzioni pure (pathlib, json): nessun oggetto Qt viene toccato
    da questo thread, la consegna al thread proprietario passa dal
    segnale `_mappa_pronta`, mai da una chiamata diretta -- lo stesso
    schema di `gui/faceset/decodifica.py::_Lavoro`.
    """

    def __init__(self, cache_dir, cartella, generazione, emettitore):
        super().__init__()
        self._cache_dir = cache_dir
        self._cartella = cartella
        self._generazione = generazione
        self._emettitore = emettitore

    def run(self):
        # Controllo INIZIALE, come `decodifica._Lavoro.run`: con un solo
        # thread nel pool, una raffica di apri() metterebbe in coda giri
        # gia' superati -- senza questo ognuno pagherebbe comunque
        # l'enumerazione completa prima di scoprirsi inutile, allargando
        # la finestra in cui la pagina risponde con la mappa vuota.
        if self._generazione != self._emettitore._generazione_mappa:
            return
        try:
            mappa, mancanti = mappa_per_fotogramma(self._cache_dir, self._cartella)
        except OSError:
            # I2 della revisione finale: `mancanti=None` e' un sentinella,
            # non una lista vuota -- una cartella IRRAGGIUNGIBILE non
            # equivale a una vuota, o `_mappa_affidabile()` la crederebbe
            # affidabile e `_riconcilia_rapporto` azzererebbe ogni riga.
            mappa, mancanti = {}, None
        try:
            self._emettitore._mappa_pronta.emit(self._generazione, mappa, mancanti)
        except RuntimeError:
            # La pagina e' stata distrutta (finestra chiusa, non solo
            # scheda) mentre l'enumerazione girava: non c'e' piu' nessuno
            # a cui consegnare, e un'eccezione qui abortirebbe il processo
            # -- stesso guasto e stessa cura di gui/training_panel.py.
            pass


class PaginaEstrazione(QWidget):
    # Emesso dal bottone "Show faces from this frame": chi ospita questa
    # pagina (la finestra principale) lo ascolta per portare la scheda di
    # cura del faceset sui volti del fotogramma indicato.
    richiesta_volti_del_frame = pyqtSignal(str, str)
    # Segnale interno: l'unico passaggio fra il thread di lavoro di
    # `_LavoroMappa` e il thread proprietario di questa pagina. Mai
    # emesso ne' collegato fuori da questo modulo.
    _mappa_pronta = pyqtSignal(int, object, object)

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
        self._accurato = True
        # I motori della sessione manuale: le CHIAVI del registro, mai le
        # etichette. Come `_accurato`, si azzerano ai default in due punti
        # (qui e in `ferma_servizio`) -- all'ingresso in sessione e' il
        # progetto a rimetterle, se le ricorda.
        self._rilevatore = MotoriCatalog.DEFAULT_RILEVATORE
        self._allineatore = MotoriCatalog.DEFAULT_ALLINEATORE
        self._tieni_in_memoria = True
        # Un motore nuovo sta per essere costruito: lo alzano l'ingresso in
        # sessione e i due selettori, lo consuma `_rileva` -- l'unico punto
        # che sa se una rilevazione parte DAVVERO, e quindi se ci sara' una
        # risposta a spegnere la barra. Resta alzato se la rilevazione non
        # e' partita, ed e' giusto: i motori sono ancora tutti da caricare,
        # e la prima che parte davvero mostrera' la barra.
        self._motori_da_caricare = False
        # L'avviso di "project.json non scrivibile" e' gia' stato dato in
        # questa sessione: vedi `_ricorda_motori`.
        self._avvisato_memoria_non_scritta = False
        # Il nome del passo con cui questa sessione e' entrata: e' la
        # chiave con cui il progetto ricorda le tre scelte, e si prende da
        # `_entra_modalita_manuale` invece di ricalcolarlo (passo_per puo'
        # sollevare KeyError, e uno slot non e' il posto dove scoprirlo).
        self._passo_manuale = None
        # La nota "Saved ...": _salva_corrente la
        # scrive qui invece che direttamente in etichetta_stato, perche' fra
        # il salvataggio e il prossimo ridisegno c'e' sempre uno sfogliamento
        # (_vai_a -> _carica_frame -> _rileva) che altrimenti la
        # cancellerebbe prima che l'utente la legga mai. _aggiorna_stato_manuale
        # la consuma una volta sola.
        self._nota_salvataggio = None
        # Gemella di `_nota_salvataggio`, per l'avviso scritto da `apri()`
        # quando ricaricare chiude una sessione manuale viva: senza questo,
        # `_su_frame_scelto` -> `_mostra_anteprima` -- che `mostra_frame`
        # chiama SEMPRE subito dopo -- lo sovrascrive prima che l'utente
        # possa leggerlo. Consumata una volta sola da `_mostra_anteprima`.
        self._nota_sessione_interrotta = None
        self._salvati_per_frame = {}
        self._ultima_mossa_debug = None
        # Quale OPERAZIONE (non quale passo -- "auto" e
        # "auto-con-correzione" lanciano oggi lo stesso passo del catalogo)
        # ha avviato il job corrente, con le risposte gia' date. Azzerati
        # insieme in _su_job_finito: solo cosi' _su_job_finito sa se deve
        # entrare da sola nella sessione manuale sui frame mancati, invece
        # di dipendere dal nome del passo -- che oggi i due bottoni condividono.
        self._operazione_in_corso = None
        self._risposte_operazione_in_corso = None
        # Il progetto e il lato di QUANDO il job
        # e' partito -- il menu Project non e' gated da "libera" apposta, e
        # puo' chiamare apri() su un progetto DIVERSO mentre questo job gira
        # ancora. Senza questo confronto, al `finished` del job di A (con la
        # pagina ormai su B) _su_job_finito entrerebbe in manuale su
        # B/data_dst con le risposte date per A -- il costo e' un dataset
        # rovinato scoperto ore dopo, non solo uno step busy di troppo.
        self._workspace_operazione_in_corso = None
        self._lato_operazione_in_corso = None
        self._occupante = None            # PassoFittizio registrato mentre il servizio e' vivo
        self._identita_occupante = None
        # I1/I2 della revisione finale: removeTab()+setParent() (come fa
        # MainWindow._on_central_tab_close_requested) NON consegnano mai un
        # closeEvent, quindi senza un percorso di chiusura ESPLICITO
        # ne' closeEvent (sotto) ne' l'ingresso automatico in manuale di
        # _su_job_finito saprebbero mai che la scheda non e' piu' sullo
        # schermo. Un flag proprio, non isVisible()/isHidden(): sotto la
        # piattaforma offscreen la visibilita' dei widget e' una trappola
        # gia' misurata in questo repository (show()+repaint() non dipinge
        # un figlio, widget.grab() da solo riempie la tela prima di
        # renderizzare) -- un test che dipendesse da lei sarebbe fragile
        # per ragioni che non hanno niente a che vedere con cio' che
        # verifica. su_chiusura_scheda()/su_apertura_scheda() sotto sono i
        # due soli scrittori.
        self._scheda_aperta = True
        # Un secondo: il rapporto cresce di una riga per frame, e un
        # ridisegno al secondo e' gia' piu' veloce di quanto si legga.
        self._lettore_vivo = None
        self._timer_rapporto = QTimer(self)
        self._timer_rapporto.setInterval(1000)
        self._timer_rapporto.timeout.connect(self._pulsa_rapporto)
        # I volti del fotogramma corrente, recuperati pigramente dal client
        # di dettaglio -- solo quando una spunta di sovrapposizione li vuole
        # davvero. `_volti_di` e' il fotogramma a cui `_volti_correnti`
        # appartiene, `_frame_chiesto` quello dell'ultima richiesta in volo
        # (per scartare le consegne in ritardo, come `_su_anteprima_pronta`).
        self._volti_correnti = []
        self._volti_di = None
        self._frame_chiesto = None
        # L'id della richiesta `frame` in volo, o None: il client e'
        # ASINCRONO, quindi "sto aspettando" si riconosce per id -- non
        # per un booleano "dentro la chiamata", che con una risposta
        # differita non varrebbe piu' niente. Vedi `_su_volti_pronti`, che
        # ascolta un canale condiviso con la finestra di dettaglio.
        self._id_volti_atteso = None
        # Il click che `_su_volto_scelto` non ha potuto risolvere subito
        # perche' la richiesta al servizio era ancora in volo: (frame,
        # punto), o None. Si riesegue alla consegna (`_su_volti_pronti`)
        # invece di sparire -- col client sincrono di prima la risposta
        # arrivava dentro la stessa chiamata, quindi il primo click non
        # era mai in questa situazione.
        self._click_in_sospeso = None
        # La bandierina che copre il solo caso in cui il client CONSEGNA
        # nello stesso giro della chiamata (i doppi dei test): li' la
        # risposta arriva PRIMA che `_id_volti_atteso` sia assegnato.
        self._attesa_diretta_volti = False
        # Vero mentre il cursore d'attesa dei volti e' acceso: un
        # `setOverrideCursor` per richiesta in volo, mai due, o lo stack
        # di Qt resterebbe con un push non ripristinato quando una
        # risposta ne supera un'altra.
        self._cursore_volti_attivo = False
        # L'ultimo guasto del client di dettaglio, (motivo, codice) o None
        # se nessuno e' arrivato -- o se l'ultima risposta e' andata a buon
        # fine. Lo legge SOLO `_perche_nessun_volto`, sul click: senza,
        # un servizio muto faceva accusare al rapporto una vecchiaia che
        # non aveva.
        self._ultimo_guasto = None
        self._cartella_dettaglio = None
        self._cliente = None
        # La finestra di dettaglio aperta dal click sulla tela -- None
        # finche' nessuno ha ancora cliccato un volto. Costruita al primo
        # bisogno come `_cliente`, e per la stessa ragione.
        self._finestra_dettaglio = None
        # La mappa fotogramma -> percorsi degli allineati, e i volti che
        # l'indice non copre. Costruite da `_ricostruisci_mappa`, che gira
        # fuori dal thread di Qt: su una cartella grande enumerare i volti
        # costa secondi, e farlo nello slot che apre la cartella
        # congelerebbe la pellicola.
        self._mappa = {}
        self._mancanti_indice = []
        # La generazione della mappa che seguira' il ricaricamento provocato
        # dal FINIRE della nostra stessa passata automatica -- None quando
        # nessuna e' in attesa di essere consumata. Impostata da
        # `_su_job_finito`, mai da `avvia_indicizzazione_volti`: solo a job
        # finito si conosce il numero VERO del ricaricamento che quel job
        # provoca. Un file non indicizzabile o una riconciliazione fallita
        # non azzerano `_mancanti_indice`, quindi senza questo la consegna
        # che segue rilancerebbe la stessa passata all'infinito;
        # sulla GENERAZIONE e non sulla cartella perche' un apri() interposto
        # (F5, cambio progetto) mentre la passata e' in volo si prende
        # sempre un numero diverso, e non puo' consumare la guardia al
        # posto del ricaricamento vero.
        self._generazione_indicizzazione_da_consumare = None
        # Ogni chiamata a `_ricostruisci_mappa` incrementa questo contatore
        # PRIMA di avviare il lavoro: una consegna la cui generazione non
        # e' piu' quella corrente arriva da una cartella che non e' piu'
        # quella aperta, e va scartata invece di sovrascrivere `_mappa`
        # con dati vecchi.
        self._generazione_mappa = 0
        self._pool_mappa = QThreadPool()
        self._pool_mappa.setMaxThreadCount(1)

        self.modello = ModelloFrame()
        self.tela = Tela()
        self.colonna = ColonnaComandi()
        self.pellicola = Pellicola()
        self.pellicola.setModel(self.modello)
        self.pila = PilaProgresso()
        self.servizio = None
        # Un secondo decodificatore, non quello della pellicola: le
        # anteprime grandi peserebbero cento volte una miniatura e
        # sfratterebbero l'intera striscia, che si ridecodificherebbe a ogni
        # scorrimento. Tetto proprio, in byte (mai in voci: una voce non
        # pesa sempre uguale), sufficiente per qualche frame 4K.
        self.decodificatore_anteprima = Decodificatore(self)
        self.decodificatore_anteprima.TETTO_CACHE_BYTE = 48 * 1024 * 1024
        self.decodificatore_anteprima.pronta.connect(self._su_anteprima_pronta)
        self._mappa_pronta.connect(self._su_mappa_pronta)

        self.bottone_src = QPushButton(testi.FACESET_SRC)
        self.bottone_dst = QPushButton(testi.FACESET_DST)
        self.bottone_src.setCheckable(True)
        self.bottone_dst.setCheckable(True)
        self.selettore_operazione = theme.tendina()
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
        self.bottone_volti = QPushButton(testi.ESTRAZIONE_VOLTI_DEL_FRAME)
        self.bottone_volti.setToolTip(testi.ESTRAZIONE_VOLTI_DEL_FRAME_TIP)
        # I tre controlli dei motori: elenco, ordine, etichette e aiuti
        # vengono da MotoriCatalog, che ne e' la sorgente unica -- qui non
        # si riscrive nessuna voce. La tendina mostra la `label` e porta la
        # `key`, come i campi del catalogo (gui/catalog/extraction.py).
        # Letto una sola volta, qui alla costruzione: e' un accesso al
        # filesystem, non un dato da ricontrollare a ogni ridisegno della
        # barra. `avvio_mod.dfl_root()` e' la stessa radice da cui
        # `TrasportoAsincrono` lancia il figlio -- l'albero in cui il
        # figlio cerchera' davvero i pesi, non un'approssimazione.
        self._pesi_mancanti_rilevatori = _pesi_mancanti(avvio_mod.dfl_root(), MotoriCatalog.RILEVATORI)
        self._pesi_mancanti_allineatori = _pesi_mancanti(avvio_mod.dfl_root(), MotoriCatalog.ALLINEATORI)
        self.etichetta_rilevatore = QLabel(testi.ESTRAZIONE_RILEVATORE)
        self.selettore_rilevatore = _selettore_motori(MotoriCatalog.RILEVATORI,
                                                      MotoriCatalog.DEFAULT_RILEVATORE,
                                                      testi.ESTRAZIONE_RILEVATORE_TIP,
                                                      self._pesi_mancanti_rilevatori)
        self.etichetta_allineatore = QLabel(testi.ESTRAZIONE_ALLINEATORE)
        self.selettore_allineatore = _selettore_motori(MotoriCatalog.ALLINEATORI,
                                                       MotoriCatalog.DEFAULT_ALLINEATORE,
                                                       testi.ESTRAZIONE_ALLINEATORE_TIP,
                                                       self._pesi_mancanti_allineatori)
        self.spunta_memoria = QCheckBox(testi.ESTRAZIONE_MEMORIA)
        self.spunta_memoria.setToolTip(testi.ESTRAZIONE_MEMORIA_TIP)
        self.spunta_memoria.setChecked(True)
        self._controlli_motori = (self.etichetta_rilevatore, self.selettore_rilevatore,
                                  self.etichetta_allineatore, self.selettore_allineatore,
                                  self.spunta_memoria)
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
        self.spunta_rect = QCheckBox(testi.ESTRAZIONE_SOVRAPP_RECT)
        self.spunta_rect.setChecked(True)
        self.spunta_landmarks = QCheckBox(testi.ESTRAZIONE_SOVRAPP_LANDMARKS)
        self.spunta_maschera = QCheckBox(testi.ESTRAZIONE_SOVRAPP_MASCHERA)
        for spunta in self._spunte_sovrapposizioni():
            spunta.toggled.connect(self._su_sovrapposizioni)
            riga_filtri.addWidget(spunta)

        barra = QHBoxLayout()
        for w in (self.bottone_src, self.bottone_dst, self.selettore_operazione,
                  self.bottone_avvia, self.bottone_manuale, self.bottone_riestrai,
                  self.bottone_annulla_riestrai, self.bottone_indice, self.bottone_volti):
            barra.addWidget(w)
        barra.addStretch(1)
        barra.addWidget(self.etichetta_stato)

        # I tre controlli dei motori su una riga PROPRIA, sotto la barra, e
        # non dentro: misurato, in sessione manuale allargavano la barra da
        # 951 a 1651 px alla scala normale e da 1171 a 2058 alla xlarge --
        # piu' di un monitor 1920 -- e un QHBoxLayout non va a capo, schiaccia:
        # a 1280 px "Keep models in memory" era tagliato a meta' e con lui
        # "Manual session", "Re-extract selection" e "Rebuild report".
        #
        # Un WIDGET contenitore, non il solo layout: un widget nascosto esce
        # del tutto dal layout genitore, mentre una riga di soli widget
        # nascosti resta una voce del QVBoxLayout e si porta dietro la
        # propria spaziatura. Fuori dalla sessione la pagina non cambia di
        # un pixel, che e' la condizione per aggiungere una riga.
        self.riga_motori = QWidget()
        barra_motori = QHBoxLayout(self.riga_motori)
        barra_motori.setContentsMargins(0, 0, 0, 0)
        for w in self._controlli_motori:
            barra_motori.addWidget(w)
        barra_motori.addStretch(1)
        self.riga_motori.setVisible(False)

        radice = QVBoxLayout(self)
        radice.addLayout(barra)
        radice.addWidget(self.riga_motori)
        radice.addWidget(self.pila)
        centro = QHBoxLayout()
        centro.addWidget(self.tela, 1)
        centro.addWidget(self.colonna)
        radice.addLayout(centro, 1)
        radice.addLayout(riga_filtri)
        radice.addWidget(self.pellicola)

        self.bottone_src.clicked.connect(lambda: self._cambia_lato("src"))
        self.bottone_dst.clicked.connect(lambda: self._cambia_lato("dst"))
        self.bottone_avvia.clicked.connect(lambda: self.avvia_operazione())
        self.bottone_manuale.toggled.connect(self._su_manuale_toggled)
        self.bottone_riestrai.clicked.connect(lambda: self.riestrai_selezione())
        self.bottone_annulla_riestrai.clicked.connect(lambda: self.annulla_riestrazione())
        self.bottone_indice.clicked.connect(lambda: self.aggiorna_indice())
        self.bottone_volti.clicked.connect(self._su_volti_del_frame)
        # currentIndexChanged e non activated: `_applica_motori_ricordati`
        # scrive i selettori da codice a bordo sessione, e li' i segnali
        # sono bloccati apposta -- vedi la sua nota.
        self.selettore_rilevatore.currentIndexChanged.connect(self._su_rilevatore_scelto)
        self.selettore_allineatore.currentIndexChanged.connect(self._su_allineatore_scelto)
        self.spunta_memoria.toggled.connect(self._su_memoria_scelta)
        for chiave, bottone in self._bottoni_filtro.items():
            bottone.clicked.connect(lambda _c=False, chiave=chiave: self._su_filtro(chiave))
        self.pellicola.frame_scelto.connect(self._su_frame_scelto)
        self.tela.vettore_tracciato.connect(self._su_vettore_tracciato)
        self.tela.rettangolo_cambiato.connect(self._su_rettangolo_cambiato)
        self.tela.blocco_cambiato.connect(self._su_blocco_cambiato)
        self.tela.volto_scelto.connect(self._su_volto_scelto)
        self.colonna.scelto.connect(self._su_comando)

        # Qt.WidgetWithChildrenShortcut scatta quando ad avere il focus e' il
        # widget su cui l'azione e' stata aggiunta con addAction(), o un suo
        # discendente -- MAI il widget che l'ha costruita (ColonnaComandi non
        # chiama mai self.addAction, vedi la sua docstring). Tela, colonna e
        # pellicola sono tutte discendenti di questa pagina: aggiungere qui
        # le rende raggiungibili da qualunque loro focus, che e' cio' che un
        # test con QTest.keyClick ha verificato mancare del tutto prima di
        # questa riga (14 comandi su 15 non scattavano ne' da tela ne' dalla
        # pagina). Le quattro frecce sono l'eccezione: vanno alla TELA
        # soltanto, per non rubare la navigazione da tastiera alla pellicola
        # (una QListView che le usa gia' per spostarsi fra i frame) quando il
        # focus e' sulla striscia.
        for c in COMANDI:
            azione = self.colonna.azione(c.chiave)
            if c.chiave in CHIAVI_FRECCE:
                self.tela.addAction(azione)
            else:
                self.addAction(azione)

        self._aggiorna_lato_bottoni()
        self._rigenera_comandi()
        self._aggiorna_spunte_sovrapposizioni()
        self._su_sovrapposizioni()

    # -- progetto e lato -------------------------------------------------

    def apri(self, progetto, lato):
        """Elenca i frame di <progetto>/data_<lato> e legge il rapporto
        dalla cache di quella cartella. Non avvia il servizio -- vedi
        `_entra_modalita_manuale`."""
        # M4 della revisione finale: F5 e il menu Project restano gli unici
        # due percorsi che possono chiudere una sessione manuale aperta
        # SENZA chiedere (deliberato: un ricaricamento esplicito deve poter
        # uscire da ogni stato) -- ma farlo in silenzio lasciava l'utente a
        # scoprirlo da un rettangolo sparito. Catturato PRIMA di
        # ferma_servizio(), che azzera self.servizio insieme al resto.
        sessione_interrotta = self.servizio is not None
        self.ferma_servizio()
        # La pila appartiene al bersaglio, non alla pagina: si svuota quando
        # si guarda un'altra cartella, mai quando si rilegge la stessa. Senza
        # questa distinzione, "Refresh state" premuto durante un'estrazione
        # cancellerebbe barre che non tornano piu' -- il figlio manda `open`
        # una volta sola e `PilaProgresso._inc` scarta gli id sconosciuti.
        stesso_bersaglio = (self._progetto == Path(progetto) and self._lato == lato)
        self._progetto = Path(progetto)
        self._lato = lato
        self._cartella = self._progetto / ("data_%s" % lato)
        # La cache della decodifica e' indicizzata per percorso: le
        # miniature della cartella di prima non serviranno mai piu'.
        self.pellicola.decodificatore.svuota()
        self.decodificatore_anteprima.svuota()
        letti = elenca_cartella(self._cartella, ESTENSIONI)
        percorsi = [p for p, _dimensione, _mtime in letti]
        self._voci = indice_mod.leggi(self._cache_dir())
        self.modello.imposta(percorsi, self._voci)
        # Il lettore riparte dalla FINE di cio' che `indice_mod.leggi` ha
        # appena letto: altrimenti il primo battito riconsegnerebbe l'intero
        # rapporto come se fosse nuovo.
        self._lettore_vivo = LettoreIncrementale(self._cache_dir())
        self._lettore_vivo.nuove()
        self._salvati_per_frame = {}
        self._frame_corrente = None
        self._pixmap_corrente = None
        self._rect_corrente = None
        self._landmarks_correnti = None
        # Un guasto del client appartiene alla cartella appena lasciata: su
        # quella nuova non e' ancora successo niente.
        self._ultimo_guasto = None
        # Senza questo il bottone "Undo" resta abilitato dopo un cambio di
        # progetto o di lato e agirebbe sui file del progetto PRECEDENTE:
        # nessun dato si perde (i percorsi della Mossa sono assoluti, e
        # cestino.annulla salta se l'origine esiste gia'), ma il bottone
        # mentirebbe su cosa sta per annullare.
        self._ultima_mossa_debug = None
        # Solo se il bersaglio e' cambiato, o non c'e' piu' un job vivo a
        # possederle: altrimenti "Refresh state" (F5) sulla STESSA cartella
        # mentre un'estrazione gira cancellerebbe le sue barre per sempre
        # (vedi il commento sopra `stesso_bersaglio`). Il cambio di
        # progetto passa qui anche con un job vivo altrove -- il menu
        # Project non e' gated da "libera", di proposito (piu' progetti
        # possono avere job attivi insieme) -- quindi resta pulito.
        if not stesso_bersaglio or self._job_corrente is None:
            self.pila.pulisci()
        self.tela.mostra(None, None, None)
        self._aggiorna_lato_bottoni()
        self._aggiorna_conteggi_filtro()
        self._aggiorna_messaggio_vuoto()
        if sessione_interrotta:
            # Tenuta da parte, non solo scritta: se `mostra_frame` chiama
            # subito dopo `_su_frame_scelto` -> `_mostra_anteprima`, quella
            # deve COMPORLA nella propria riga invece di sovrascriverla --
            # altrimenti l'avviso vive sullo schermo per zero secondi.
            self._nota_sessione_interrotta = \
                testi.estrazione_sessione_interrotta_dal_ricaricamento(len(percorsi))
            self.etichetta_stato.setText(self._nota_sessione_interrotta)
        else:
            self._nota_sessione_interrotta = None
            self.etichetta_stato.setText(testi.estrazione_stato(len(percorsi)))
        self._rigenera_comandi()
        self._ricostruisci_mappa()

    def _cache_dir(self):
        return self._radice_e / CACHE_SOTTOCARTELLA / cache_mod.id_cartella(self._cartella)

    def lato(self):
        return self._lato

    def _cambia_lato(self, lato):
        if self._progetto is not None and lato != self._lato:
            self.apri(self._progetto, lato)
        else:
            # Il bottone si e' gia' scommutato da solo al click: senza
            # questa riga un riclic sul lato attivo lascia la barra senza
            # nessun lato acceso mentre la pagina e' ancora su quel lato.
            self._aggiorna_lato_bottoni()

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
        self._aggiorna_messaggio_vuoto()

    def _aggiorna_messaggio_vuoto(self):
        """Due vuoti diversi meritano due frasi diverse: la cartella senza
        frame manda l'utente ai passi 2 e 3, il filtro senza risultati no --
        li' i frame ci sono."""
        if self._cartella is None or self._pixmap_corrente is not None \
                or self.modello.rowCount() > 0:
            self.tela.imposta_messaggio("")
        elif self.modello.totale() == 0:
            self.tela.imposta_messaggio(
                testi.estrazione_cartella_vuota(self._cartella.name))
        else:
            self.tela.imposta_messaggio(testi.ESTRAZIONE_FILTRO_VUOTO)

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
        job = self._lancia(passo, risposte, self._cartella,
                           extra_args=self._cartella_e_report())
        if job is not None:
            # Solo se il lancio e' davvero partito: un rifiuto (occupato,
            # conflitto) non deve lasciare qui un'operazione che nessun
            # _su_job_finito verra' mai a consumare -- il prossimo lancio
            # riuscito la sovrascriverebbe comunque, ma un valore fermo qui
            # non e' mai stato letto da nessuno se non partito da un job vero.
            self._operazione_in_corso = chiave
            self._risposte_operazione_in_corso = risposte
            self._workspace_operazione_in_corso = identita_workspace(self._progetto)
            self._lato_operazione_in_corso = self._lato
        return job

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

    def avvia_indicizzazione_volti(self):
        """La passata sull'indice di `aligned/`, per i soli volti che
        `_ricostruisci_mappa` ha trovato scoperti -- diverso dal passo
        sopra: quello ricostruisce il RAPPORTO per frame, questo l'indice
        dei VOLTI (fratelli, landmark, maschera) che
        `gui/faceset/indice.py::mappa_per_fotogramma` legge.

        `controlla_occupazione=False` come `aggiorna_indice`, per la stessa
        ragione: la cache vive fuori dal progetto, quindi non deve restare
        grigio perche' un ALTRO job tiene la cartella. Ma qui il passo
        dichiara `consumes` sull'artefatto allineato del lato corrente
        (`azioni_mod.passo_indice_volti`): `try_start` lo rifiuta comunque,
        per via interna, contro un job vero o la sessione manuale nativa
        (registrata come occupante esterno) che scrivono LI' in questo
        momento -- e' la corsa vera, non un caso di laboratorio: e' esattamente
        cio' che `_su_job_finito` fa entrando da sola in sessione manuale a
        estrazione finita, PRIMA che questa consegna arrivi.

        `silenzioso=True`: nessuna finestra per un lavoro che l'utente non
        ha chiesto -- il rifiuto finisce sulla riga di stato, e il
        tentativo resta disponibile al prossimo giro (nessuno stato viene
        marcato "in volo" qui: quella guardia vive in `_su_job_finito`,
        sulla generazione del ricaricamento che la passata VERA provoca).
        """
        if self._gestore is None or self._cartella is None:
            return None
        cartella = self._cartella / "aligned"
        passo = azioni_mod.passo_indice_volti(self._lato)
        return self._lancia(passo, {}, cartella,
                            extra_args=("--cache-dir", str(self._cache_dir_aligned()),
                                        "--only-missing"),
                            controlla_occupazione=False, silenzioso=True)

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

    def _lancia(self, passo, risposte, cartella, extra_args=(), controlla_occupazione=True,
                silenzioso=False):
        # `silenzioso=True` e' per un lancio che l'utente non ha chiesto
        # (avvia_indicizzazione_volti, dalla mappa che si ricostruisce da
        # sola): una QMessageBox qui bloccherebbe l'interfaccia su un
        # dialogo che nessuno si aspetta, per un lavoro che nessuno ha
        # avviato -- un vuoto senza spiegazione sarebbe stato piu' onesto di
        # questo. Il rifiuto finisce sulla riga di stato, e non e' un vicolo
        # cieco: il tentativo resta disponibile al prossimo giro, la stessa
        # guardia "un job alla volta" e lo stesso StepConflict qui sotto
        # valgono anche per un lancio esplicito dell'utente subito dopo.
        def _rifiuta(titolo, testo):
            if silenzioso:
                self.etichetta_stato.setText(testo)
                return None
            QMessageBox.warning(self, titolo, testo)
            return None
        # Un job alla volta da questa pagina -- stessa scelta di
        # gui/faceset/pagina.py, e per la stessa ragione (le barre
        # numerate dal figlio si pilotano a vicenda con due job vivi).
        if self._job_corrente is not None:
            return _rifiuta(testi.TITLE_FACESET_ONE_AT_A_TIME,
                            testi.faceset_one_job_at_a_time(self._nome_job_corrente))
        # controlla_occupazione=False e' per aggiorna_indice() e
        # avvia_indicizzazione_volti(): quei due non contendono l'artefatto
        # per QUESTO controllo locale (aggiorna_indice non dichiara niente;
        # avvia_indicizzazione_volti dichiara `consumes`, ma la rete che
        # conta per lei e' quella dentro try_start qui sotto, non questa),
        # quindi non devono restare grigi perche' un ALTRO job tiene la
        # cartella, esattamente come avvia_indicizzazione() in
        # gui/faceset/pagina.py.
        if controlla_occupazione:
            occupata = self._occupata()
            if occupata is not None:
                return _rifiuta(testi.TITLE_STEP_BUSY, testi.job_holds(occupata[0], occupata[1]))
        try:
            job = self._gestore.try_start(passo, risposte, self._progetto,
                                          extra_args=extra_args, input_dir=cartella,
                                          avviato_da_utente=not silenzioso)
        except StepConflict as exc:
            esito = _rifiuta(testi.TITLE_STEP_BUSY, str(exc))
            self._rigenera_comandi()
            return esito
        self.pila.pulisci()
        if job is not None:
            self._job_corrente = job
            self._nome_job_corrente = passo.name
            job.progress.connect(self._su_progresso)
            job.finished.connect(self._su_job_finito)
            self._timer_rapporto.start()
            # Prima di ogni riga del figlio: fra il click e la prima barra
            # ci sono i secondi in cui torch e il modello si caricano.
            self.pila.mostra_avvio(passo.name)
            self.etichetta_stato.setText(testi.estrazione_avvio_in_corso(passo.name))
        self._rigenera_comandi()
        return job

    def _su_progresso(self, riga):
        # M5 della revisione finale: `_lancia` scrive "Starting %s..." e
        # prima di questa riga nessuno la sostituiva finche' il job non
        # finiva -- il testo mentiva per tutta la durata del job (quaranta
        # minuti, su un'estrazione vera). Il segnale che il job ha
        # cominciato a produrre davvero e' lo STESSO che toglie la barra
        # indeterminata (PilaProgresso._togli_avvio, dietro
        # avvio_visibile()): non serve inventarne un altro.
        avvio_prima = self.pila.avvio_visibile()
        self.pila.applica(riga)
        if avvio_prima and not self.pila.avvio_visibile():
            self.etichetta_stato.setText(testi.estrazione_job_in_corso(self._nome_job_corrente))

    def _pulsa_rapporto(self):
        """Il rapporto e' cresciuto mentre il job gira.

        Slot di QTimer: non deve sollevare per nessun dato sul disco. I
        contatori dei filtri si ricalcolano sempre -- e' il segno visibile
        che il lavoro procede -- mentre l'insieme delle righe mostrate resta
        fermo (vedi `ModelloFrame.aggiorna_voci`).
        """
        if self._lettore_vivo is None:
            return
        nuove = self._lettore_vivo.nuove()
        if not nuove:
            return
        per_nome = dict((v.get("nome"), v) for v in self._voci
                        if isinstance(v, dict) and isinstance(v.get("nome"), str))
        for v in nuove:
            per_nome[v["nome"]] = v
        self._voci = list(per_nome.values())
        self.modello.aggiorna_voci(nuove)
        self._aggiorna_conteggi_filtro()

    def _intervallo_image_size(self):
        """Il `valid_range` del campo "image-size" nel catalogo -- letto
        dal passo, non duplicato a mano: un cambio in
        gui/catalog/extraction.py non deve disallineare in silenzio la
        guardia sotto. `None` se il passo non si trova (non dovrebbe
        capitare qui: si arriva a questo metodo solo per
        CHIAVE_AUTO_CORREZIONE, che esiste solo per "dst"), o se il campo
        non dichiara un intervallo.

        Normalizzato QUI: `FieldDef.valid_range`
        vale `()`, non `None`, quando un campo non lo dichiara
        (gui/catalog/model.py:73) -- due sentinelle diverse per "nessun
        intervallo". Tornare `()` senza tradurla lascerebbe al chiamante
        l'obbligo di saperlo: `intervallo is not None` sarebbe vero per
        `()`, e `intervallo[0]` sollevava `IndexError` dentro
        `_image_size_da_aligned`, chiamata da uno slot Qt
        (`_su_job_finito`) -- un `IndexError` li' chiama `qFatal` e si
        porta via il processo con dentro ogni training aperto. Non
        scattava solo perche' `_IMAGE_SIZE` dichiara oggi
        `valid_range=(256, 2048)`: un difetto a scoppio ritardato, non
        raggiunto da nessun campo attuale ma pronto per il primo che non
        dichiari un intervallo."""
        try:
            passo = azioni_mod.passo_per(CHIAVE_MANUALE, self._lato)
        except KeyError:
            return None
        for campo in passo.fields:
            if campo.key == "image-size":
                return campo.valid_range if campo.valid_range else None
        return None

    def _image_size_da_aligned(self):
        """Il lato si legge da un volto VERO appena scritto da questo
        stesso job, invece di duplicare qui la regola di default che
        Extractor.py risolve da face_type (512 sotto HEAD, 768 da HEAD in
        su): sarebbero due verita' da tenere in sincronia, e il giorno che
        una cambia la divergenza torna in silenzio. `None` se `aligned/`
        e' vuota, illeggibile o il lato letto e' fuori dal `valid_range`
        del catalogo -- in ognuno di questi casi il chiamante ripiega sul
        default, cioe' cio' che `_valori_con_default` avrebbe usato
        comunque.

        Il candidato e' quello con
        `st_mtime` PIU' RECENTE, non il primo in ordine alfabetico --
        `continue-extraction` su una cartella gia' popolata a un'altra
        dimensione lascia in `aligned/` sia i volti VECCHI sia quelli
        appena scritti, e l'ordine alfabetico non e' l'ordine temporale.
        Lo `stat()` per il confronto rende l'`except OSError` finalmente
        raggiungibile: un file cancellato fra `glob()` e `stat()`, o
        `aligned/` diventata illeggibile a meta' del giro, sollevano da li'
        -- prima lo faceva solo la `glob()` iniziale, che su una cartella
        `chmod 000` torna vuota invece di sollevare."""
        try:
            candidati = list((self._cartella / "aligned").glob("*.jpg"))
            if not candidati:
                return None
            piu_recente = max(candidati, key=lambda p: p.stat().st_mtime)
        except OSError:
            return None
        immagine = QPixmap(str(piu_recente))
        if immagine.isNull():
            return None
        lato = immagine.width()
        intervallo = self._intervallo_image_size()
        if intervallo is not None and not (intervallo[0] <= lato <= intervallo[1]):
            return None
        return lato

    def _su_job_finito(self, _codice):
        self._timer_rapporto.stop()
        # Preso PRIMA di azzerarlo: serve dopo apri() per riconoscere se il
        # ricaricamento che quella chiamata provoca e' quello della NOSTRA
        # passata automatica (vedi sotto).
        nome_job_finito = self._nome_job_corrente
        self._job_corrente = None
        self._nome_job_corrente = ""
        self.pila.pulisci()
        operazione = self._operazione_in_corso
        risposte = self._risposte_operazione_in_corso
        workspace_lancio = self._workspace_operazione_in_corso
        lato_lancio = self._lato_operazione_in_corso
        self._operazione_in_corso = None
        self._risposte_operazione_in_corso = None
        self._workspace_operazione_in_corso = None
        self._lato_operazione_in_corso = None
        # Il job ha scritto il rapporto incrementalmente: ricaricare
        # rilegge cio' che ha appena prodotto.
        self.apri(self._progetto, self._lato)
        # Se il job appena finito era la nostra passata automatica,
        # `apri()` ha appena incrementato la generazione per il
        # ricaricamento che lei stessa ha provocato: e' QUELLA consegna,
        # e nessun'altra, a non dover rilanciare un secondo tentativo
        # (vedi _su_mappa_pronta). Catturare il NUMERO invece della
        # cartella regge anche un apri() interposto (F5, cambio progetto)
        # nel frattempo: quello si prende sempre una generazione diversa,
        # quindi non puo' consumare questa guardia al posto suo.
        if nome_job_finito == azioni_mod.passo_indice_volti(self._lato).name:
            self._generazione_indicizzazione_da_consumare = self._generazione_mappa
        # "Extract and fix the misses" non apre piu' la finestra
        # cv2 -- entra da sola nella sessione manuale nativa, filtrata sui
        # frame che il rilevatore ha mancato. E' l'OPERAZIONE scelta
        # dall'utente a deciderlo, non il passo del catalogo appena
        # lanciato: "auto" e "auto-con-correzione" lanciano
        # lo STESSO passo, quindi guardare il passo farebbe entrare in
        # manuale anche l'estrazione semplice.
        #
        # Tre guardie in piu' di quella originale:
        # - `_codice == 0` (I1): uno Stop premuto dall'utente o un lancio
        #   mai partito (`_on_error_occurred` -> `_finish(-1)`) non deve
        #   aprire una sessione manuale sopra un'estrazione che non e'
        #   successa -- importerebbe torch e prenderebbe VRAM per niente,
        #   o peggio leggerebbe il rapporto della corsa PRECEDENTE (la
        #   cache sotto `_e/extract-cache/` sopravvive a un lancio fallito).
        # - `stesso_bersaglio` (I3): il menu Project non e' gated da
        #   "libera" per scelta, quindi il progetto o il lato mostrati ORA
        #   possono non essere piu' quelli su cui il job e' partito --
        #   entrare comunque scriverebbe volti nella cartella sbagliata,
        #   col face type e la dimensione dell'ALTRO progetto.
        # - senza frame mancati non si entra affatto (invariato): una
        #   sessione manuale vuota sarebbe solo un ostacolo fra l'utente e
        #   il passo dopo.
        # - `self._scheda_aperta` (I2 della revisione finale): la pagina
        #   chiusa (scheda tolta da MainWindow) resta viva e collegata a
        #   `job.finished` -- senza questa guardia un job che finisce a
        #   scheda chiusa entrerebbe comunque in sessione manuale,
        #   costruendo un processo vero (S3FD+FAN in VRAM) che nessuno
        #   vede ne' puo' fermare dal bottone "Exit manual session".
        # Il confronto usa `==`, non
        # gui/progetti.py::stesso_workspace -- deliberato, non un
        # disallineamento da uniformare. stesso_workspace e' apposta
        # sovrabbondante (identita' OPPURE percorso) perche' li' un falso
        # positivo costa solo uno "step busy" di troppo e un falso negativo
        # un dataset rovinato: sbaglia verso il permissivo. Qui la posta e'
        # rovesciata -- un falso positivo e' il dataset rovinato -- quindi
        # la forma giusta e' la piu' STRETTA (identita' E percorso), che
        # sbaglia verso il non entrare mai per errore. Un futuro
        # "uniformiamo a stesso_workspace" allenterebbe questa guardia
        # senza che nessun test lo mostri (nessuna delle due strade e'
        # esercitabile facilmente in un ambiente di test single-filesystem).
        stesso_bersaglio = (workspace_lancio is not None and self._progetto is not None
                            and identita_workspace(self._progetto) == workspace_lancio
                            and self._lato == lato_lancio)
        # M1: mai una chiave letterale nuda in uno slot Qt -- la riga sotto
        # gia' usava .get(..., 0) per la meta' del conteggio, questo la
        # uniforma sul bottone.
        bottone_senza_volto = self._bottoni_filtro.get(FILTRO_SENZA_VOLTO)
        mancati = indice_mod.conta(self._voci).get(FILTRO_SENZA_VOLTO, 0)
        if (_codice == 0 and operazione == CHIAVE_AUTO_CORREZIONE and stesso_bersaglio
                and bottone_senza_volto is not None and mancati > 0
                and self._scheda_aperta):
            # I5: `image-size`, quando l'utente non l'ha toccato, prende dal
            # figlio il default risolto da face_type -- si legge quello
            # scritto davvero, invece di duplicare la regola qui.
            if "image-size" not in risposte:
                lato_vero = self._image_size_da_aligned()
                if lato_vero is not None:
                    risposte = dict(risposte)
                    risposte["image-size"] = lato_vero
            # M4: l'interfaccia passa da sola in sessione manuale -- va
            # detto, non solo fatto vedere succedere. `_nota_salvataggio` e'
            # il canale gia' pensato per una riga mostrata una volta sola:
            # il primo `_su_volti` che
            # `_entra_modalita_manuale` scatena, tramite `_prossimo_frame`,
            # la consuma dentro `_aggiorna_stato_manuale`. M6 della
            # revisione finale: passata come `nota_ingresso`, non scritta
            # qui -- `_entra_modalita_manuale` ha quattro uscite anticipate
            # (servizio gia' presente, progetto None, cartella occupata,
            # KeyError sul passo), e scriverla qui PRIMA di quelle la
            # lascerebbe in campo a un ingresso mai avvenuto: il primo
            # `_su_volti` di una sessione SUCCESSIVA, che non c'entra
            # niente, la consumerebbe al posto suo.
            bottone_senza_volto.setChecked(True)
            self._su_filtro(FILTRO_SENZA_VOLTO)
            self._entra_modalita_manuale(
                risposte=risposte,
                nota_ingresso=testi.estrazione_correzione_avviata(mancati))

    # -- sessione manuale ---------------------------------------------------

    def _su_manuale_toggled(self, attivo):
        if attivo:
            self._entra_modalita_manuale()
        else:
            self.ferma_servizio()

    def _entra_modalita_manuale(self, servizio=None, risposte=None, nota_ingresso=None):
        """`servizio`, se dato, sostituisce la costruzione di
        TrasportoAsincrono + Servizio -- l'unica aggiunta pensata per i
        test: scrivere `self.servizio` da fuori DOPO l'ingresso salterebbe
        proprio il rilevamento automatico che questo metodo scatena
        chiamando `_prossimo_frame` -> `_carica_frame` -> `_rileva`.

        `risposte`, se dato, sostituisce `_chiedi_risposte(passo)`:
        l'ingresso automatico a job finito di "Extract and fix the
        misses" non deve riaprire il dialogo -- le risposte gia' date per
        "5) data_dst faceset extract" bastano perche' i tre campi che questa
        sessione legge davvero (face-type, image-size, jpeg-quality,
        `_valori_con_default` sotto) compaiono in ENTRAMBI i passi. Non e'
        `_CAMPI_DST` condivisa per intero -- "5) data_dst faceset extract
        MANUAL" ha la sua tupla scritta a mano, con `which-gpu-index-to-choose`
        al posto di `which-gpu-indexes-to-choose` e senza detector/
        landmarker/minimum-face-size.

        `nota_ingresso`, se dato (M6 della revisione finale), diventa
        `self._nota_salvataggio` solo DOPO che ogni uscita anticipata qui
        sotto e' stata superata -- mai prima, o un ingresso fallito
        (servizio gia' presente, progetto None, cartella occupata, KeyError
        sul passo) la lascerebbe in campo a un ingresso mai avvenuto, pronta
        a essere consumata dal primo `_su_volti` di una sessione successiva
        a cui non appartiene."""
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
        if risposte is None:
            risposte = self._chiedi_risposte(passo)
            if risposte is None:      # annullato
                self.bottone_manuale.setChecked(False)
                return
        self._parametri_manuale = _valori_con_default(passo, risposte)
        # Prima del servizio e del primo `_prossimo_frame`: la prima
        # richiesta della sessione deve gia' portare i motori ricordati,
        # non i default seguiti da una seconda richiesta che li corregge.
        self._passo_manuale = passo.name
        self._applica_motori_ricordati(passo.name)
        if servizio is not None:
            self.servizio = servizio
        else:
            # Non parte all'apertura della pagina: importa
            # torch, quindi solo qui, entrando davvero in modalita' manuale.
            # Il primo `rileva` costruisce S3FD e FAN e li tiene
            # vivi in VRAM per tutta la sessione
            # (mainscripts/ExtractManual.py::_RILEVATORE/_ALLINEATORI) --
            # non e' piu' vero che sia sola geometria: solo `landmark` lo
            # e' ancora (landmarks_da_vettore). Un timeout d'inattivita'
            # di cinque minuti evita che una sessione dimenticata aperta
            # tenga i due modelli per sempre.
            workdir = Path(tempfile.mkdtemp(prefix="dfl_estrazione_"))
            self._trasporto = TrasportoAsincrono(workdir)
            self.servizio = servizio_mod.Servizio(self._trasporto)
        # I volti della revisione vengono dal disco: in sessione manuale
        # rettangolo e landmark vengono dal motore, e le tre spunte si
        # spengono di conseguenza.
        self._dimentica_volti()
        self._aggiorna_spunte_sovrapposizioni()
        # `self.servizio` e' gia' valorizzato qui sopra: porta la tela allo
        # stato neutro della sessione manuale (rettangolo vivo acceso,
        # landmark e maschera spenti) qualunque fosse lo stato lasciato
        # dalla revisione -- altrimenti una spunta "Rect" spenta prima di
        # entrare lascerebbe il rettangolo vivo invisibile, con la spunta
        # che lo prometterebbe indietro disabilitata e quindi non
        # riaccendibile da qui dentro.
        self._su_sovrapposizioni()
        # Il primo `rileva` costruisce i modelli veri e li porta in VRAM.
        # Stessa barra pulsante che `_lancia` mostra fra il click e la
        # prima riga di un job, per lo stesso motivo: qui pero' non c'e'
        # nessun canale di avanzamento da cui aspettarsi un `open`, quindi
        # la spegne `_su_volti` alla prima risposta. Non la si accende qui:
        # su una pellicola vuota nessuna rilevazione parte e nessuna
        # risposta arriverebbe mai a spegnerla.
        self._motori_da_caricare = True
        # L'ingresso automatico da
        # _su_job_finito non passa dal toggle dell'utente -- il bottone
        # resterebbe "unchecked" mentre il suo testo gia' dice "Exit manual
        # session" (_rigenera_comandi lo decide da self.servizio, non da
        # isChecked()), e il primo clic sarebbe un no-op silenzioso:
        # rientrerebbe in _su_manuale_toggled(True) -> qui sopra -> torna
        # subito perche' self.servizio e' gia' valorizzato. blockSignals
        # evita che setChecked() faccia proprio quel giro attraverso
        # toggled().
        if not self.bottone_manuale.isChecked():
            self.bottone_manuale.blockSignals(True)
            self.bottone_manuale.setChecked(True)
            self.bottone_manuale.blockSignals(False)
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
        self._nota_salvataggio = nota_ingresso
        self._prossimo_frame()
        self._rigenera_comandi()
        self.tela.imposta_sessione_manuale(True)

    def ferma_servizio(self):
        """Pubblico e idempotente: e' il metodo che i percorsi di chiusura
        garantiti chiamano. Quattro chiamanti di produzione: il toggle
        esplicito dell'utente, il cambio di progetto/lato (`apri`),
        `MainWindow.closeEvent`, e -- dalla revisione finale --
        `su_chiusura_scheda()` sotto, che e' quello che conta DAVVERO nella
        GUI a schede: questa pagina vive dentro una scheda di un
        QTabWidget e non e' mai una finestra a se', quindi il `closeEvent`
        di QUESTA classe non riceve mai l'evento a scheda chiusa --
        `removeTab()`+`setParent()` (`MainWindow._on_central_tab_close_
        requested`) non consegnano mai un closeEvent, solo un vero
        `MainWindow.closeEvent` lo fa. Prima della revisione finale un
        `ferma()` senza questo chiamante era esattamente il difetto
        lasciato aperto dal ciclo faceset (`gui/faceset/dettaglio.py`,
        registro difetti) -- qui il costo sarebbe comunque piu' alto da
        lasciare aperto: un processo appeso resta un processo appeso, e
        ExtractManual tiene S3FD e FAN vivi in VRAM esattamente
        come il servizio di FacesetDetail, con in piu'
        un timeout d'inattivita' di cinque minuti (`sorveglia` in
        mainscripts/ExtractManual.py) -- una rete di sicurezza, non una
        sostituzione: chiamare `ferma()` da ogni percorso di chiusura
        resta piu' pulito che aspettare la sua scadenza con la GPU
        occupata."""
        self.tela.imposta_sessione_manuale(False)
        # M7 della revisione finale: senza questa riga il blocco della tela
        # sopravviveva alla sessione che l'aveva acceso -- innocuo oggi
        # (bloccato() si legge solo insieme a sessione_manuale(), gia'
        # spenta qui sopra), ma e' residuo che attraversa le sessioni, e la
        # prossima cosa che legge tela.bloccato() da sola lo erediterebbe.
        self.tela.blocca(False)
        # Uscire prima che la prima risposta dei motori arrivi e' il caso
        # in cui la barra resterebbe accesa sopra una pagina che non aspetta
        # piu' niente.
        self.pila.togli_avvio()
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
        self._accurato = True
        self._rilevatore = MotoriCatalog.DEFAULT_RILEVATORE
        self._allineatore = MotoriCatalog.DEFAULT_ALLINEATORE
        self._tieni_in_memoria = True
        self._motori_da_caricare = False
        # "Una volta per sessione" e' la sessione MANUALE, come ogni altro
        # stato di questo blocco: chi rientra dopo aver dato i permessi
        # alla cartella deve poter essere avvisato di nuovo se non basta.
        self._avvisato_memoria_non_scritta = False
        self._passo_manuale = None
        self._mostra_motori()
        self._nota_salvataggio = None
        self._dimentica_volti()
        self._aggiorna_spunte_sovrapposizioni()
        # `self.servizio` e' gia' None qui sopra: riapplica alla tela le tre
        # spunte come stavano prima della sessione manuale, che
        # `_su_sovrapposizioni` non aveva mai potuto scrivere mentre il
        # servizio era vivo.
        self._su_sovrapposizioni()
        if self.bottone_manuale.isChecked():
            self.bottone_manuale.setChecked(False)
        self._rigenera_comandi()

    #override
    def closeEvent(self, evento):
        self.su_chiusura_scheda()
        super().closeEvent(evento)

    def su_chiusura_scheda(self):
        """Il percorso di chiusura ESPLICITO per la GUI a schede (I1/I2
        della revisione finale): `MainWindow._on_central_tab_close_
        requested` lo chiama PRIMA di `removeTab()`+`setParent()`, che da
        soli non consegnano mai un closeEvent. Senza questo, una sessione
        manuale aperta restava viva a scheda chiusa -- il processo, il
        PassoFittizio che blocca ogni altro lavoro sul faceset (visto da
        `chi_occupa()`/`try_start()`), tutto -- e un job che finiva nel
        frattempo poteva ancora entrare da solo in una sessione manuale
        che nessuno vedeva (vedi la guardia `self._scheda_aperta` in
        `_su_job_finito`). Idempotente come `ferma_servizio` stesso."""
        self.ferma_servizio()
        # La finestra di dettaglio PRIMA del client, e il client solo se lei
        # si e' chiusa davvero: `close()` chiede conferma quando ci sono
        # modifiche non salvate e torna falso se l'utente risponde di no.
        # Azzerare comunque lasciava la pagina senza la sua finestra e senza
        # il suo client, e il click successivo ne costruiva un SECONDO --
        # cioe' un secondo processo figlio che importa torch mentre il primo
        # e' vivo, l'errore che questa pagina evita costruendo il client una
        # volta sola. E fermare il servizio prima della domanda lo uccideva
        # proprio mentre l'utente diceva di voler tenere il lavoro: la
        # finestra che resta aperta deve poterlo ancora salvare.
        if self._finestra_dettaglio is None or self._finestra_dettaglio.close():
            self._finestra_dettaglio = None
            # `ClienteDettaglio.ferma()` non aveva nessun chiamante di
            # produzione: questa pagina gliene da' uno. Il sorvegliante
            # d'inattivita' del servizio e' una rete di sicurezza, non una
            # sostituzione -- chiudere qui e' piu' pulito che lasciare un
            # processo appeso per cinque minuti.
            if self._cliente is not None:
                self._cliente.ferma()
                self._cliente = None
        # FUORI dall'if, e non per distrazione: questa bandierina dice se la
        # scheda e' nel QTabWidget, non se la finestra di dettaglio si e'
        # chiusa. La scheda se ne va comunque, e lasciarla a "aperta" perche'
        # l'utente vuole tenere il suo trascinamento rimetterebbe in piedi
        # proprio il buco che la guardia in `_su_job_finito` chiude: un job
        # che finisce dopo entrerebbe da solo in sessione manuale su una
        # pagina che nessuno vede piu'.
        self._scheda_aperta = False

    def su_apertura_scheda(self):
        """Il percorso di riapertura, gemello di `su_chiusura_scheda`:
        chiamato da `MainWindow.apri_pagina_estrazione` ogni volta che la
        scheda torna nel QTabWidget (prima costruzione compresa), rimette
        il flag ad 'aperta' -- senza, un job lanciato PRIMA della chiusura
        e finito DOPO la riapertura resterebbe bloccato fuori dalla
        sessione manuale per sempre, anche con la scheda di nuovo bene in
        vista."""
        self._scheda_aperta = True

    # -- le sovrapposizioni della revisione ---------------------------------

    def _spunte_sovrapposizioni(self):
        return (self.spunta_rect, self.spunta_landmarks, self.spunta_maschera)

    def _su_sovrapposizioni(self):
        """Le tre spunte valgono SOLO in revisione: e' l'unico posto che lo
        controlla, chiamato anche dall'ingresso in sessione manuale e da
        ogni cambio di fotogramma, cosi' la regola vale ovunque senza
        ripetersi. In sessione manuale la tela torna allo stato neutro
        (rettangolo vivo sempre acceso, landmark e maschera spenti perche'
        vengono dal motore, non dal disco) senza toccare le spunte -- che
        restano com'erano per quando si rientra in revisione -- e senza
        chiedere niente al servizio di dettaglio."""
        if self.servizio is not None:
            self.tela.imposta_sovrapposizioni(True, False, False)
            return
        self.tela.imposta_sovrapposizioni(self.spunta_rect.isChecked(),
                                          self.spunta_landmarks.isChecked(),
                                          self.spunta_maschera.isChecked())
        if self.spunta_landmarks.isChecked() or self.spunta_maschera.isChecked():
            self._assicura_volti()

    def _aggiorna_spunte_sovrapposizioni(self):
        """Spente durante la sessione manuale: li' rettangolo e landmark
        vengono dal MOTORE, non dal disco, e un comando che promette di
        nasconderli mentre non li governa mente. Spente e VISIBILI, non
        nascoste: una riga di comandi che cambia di lunghezza fra le due
        modalita' sposta tutto quello che le sta accanto."""
        in_revisione = self.servizio is None
        suggerimenti = (testi.ESTRAZIONE_SOVRAPP_RECT_TIP,
                        testi.ESTRAZIONE_SOVRAPP_LANDMARKS_TIP,
                        testi.ESTRAZIONE_SOVRAPP_MASCHERA_TIP)
        for spunta, suo in zip(self._spunte_sovrapposizioni(), suggerimenti):
            spunta.setEnabled(in_revisione)
            spunta.setToolTip(suo if in_revisione
                              else testi.ESTRAZIONE_SOVRAPP_MANUALE_TIP)

    def _dimentica_volti(self):
        """Il fotogramma e' cambiato, o si e' entrati in sessione manuale:
        un insieme stantio resterebbe cliccabile sotto un fotogramma
        diverso, che e' la peggiore cosa che questa pagina possa fare."""
        self._volti_correnti = []
        self._volti_di = None
        self._frame_chiesto = None
        self._click_in_sospeso = None
        self.tela.imposta_volti([])

    def _cache_dir_aligned(self):
        """La cache dell'indice faceset per la cartella `aligned/` di
        oggi -- lo stesso `percorso_cache` che usa la pagina di cura,
        NON `_cache_dir()` qui sopra: quella e' la cache del rapporto di
        estrazione, un'altra cosa."""
        return cache_mod.percorso_cache(self._radice_e, self._cartella / "aligned")

    def _ricostruisci_mappa(self):
        """La mappa fotogramma -> percorsi degli allineati, fuori dal
        thread di Qt.

        `_mappa`/`_mancanti_indice` si svuotano QUI, prima ancora di
        accodare il lavoro -- non alla consegna. Senza questo, per tutta
        l'enumerazione (1,34 s sul dataset di riferimento, secondi su uno
        grande) la pagina risponderebbe ancora con la mappa della
        cartella APERTA PRIMA: `_cambia_lato`/il menu Project cambiano
        `self._cartella` ma non hanno nessun altro modo di dirlo, e un
        click in quella finestra disegnerebbe e aprirebbe il volto
        dell'altro dataset.

        Non solleva mai verso il chiamante: e' un lavoro di sfondo, e una
        cartella non leggibile o non ancora indicizzata deve lasciare la
        pagina usabile con la mappa vuota -- esattamente lo stato di un
        progetto non ancora estratto. La barra pulsante e' la stessa di
        un job che parte (`_lancia`) e di un motore che si carica in
        sessione manuale (`_rileva`): una cartella molto grande su drvfs
        costa secondi di sola enumerazione, ed e' l'unico segnale che il
        "Refresh state" non e' rimasto fermo.
        """
        self._generazione_mappa += 1
        generazione = self._generazione_mappa
        self._mappa, self._mancanti_indice = {}, []
        if self._cartella is None:
            return
        cartella = self._cartella / "aligned"
        self.pila.mostra_avvio(testi.ESTRAZIONE_MAPPA_IN_COSTRUZIONE)
        self._pool_mappa.start(
            _LavoroMappa(self._cache_dir_aligned(), cartella, generazione, self))

    def _mappa_affidabile(self):
        """Posso fidarmi di `self._mappa` per riscrivere una riga del
        rapporto -- puntuale o per riconciliazione?

        Tre condizioni, in un solo posto invece che ripetute in ogni sito
        che scrive dalla mappa: `_mancanti_indice` non e' None -- la
        cartella si e' potuta ENUMERARE, o la mappa non dice niente di
        vero sul disco, vedi `_LavoroMappa.run` -- ed e' vuoto -- l'indice
        e' completo, altrimenti la mappa sottostima i volti dei file non
        ancora indicizzati -- e nessuno deve occupare l'artefatto allineato
        di questa cartella (`_occupata()`, altrimenti la mappa e' uno
        snapshot in corsa contro uno scrittore vero -- un job, la cura del
        faceset, un occupante esterno). Le tre domande valgono a
        prescindere da CHI stia per scrivere: la riconciliazione globale e
        la correzione puntuale del fotogramma corrente pongono tutte e tre
        questa stessa domanda, e una copia locale del controllo e' il modo
        in cui una delle due resterebbe indietro il giorno che le altre
        cambiassero.
        """
        return (self._mancanti_indice is not None and not self._mancanti_indice
                and self._occupata() is None)

    def _su_mappa_pronta(self, generazione, mappa, mancanti):
        """Slot del segnale `_mappa_pronta`: gira sempre sul thread
        proprietario di questa pagina, mai su quello di `_LavoroMappa`.

        Il confronto di generazione viene PRIMA di toccare qualunque cosa,
        `pila` compresa: una consegna superata non deve spegnere la barra
        del giro nuovo, ancora in corso.
        """
        if generazione != self._generazione_mappa:
            return    # una cartella diversa e' stata aperta nel frattempo
        self.pila.togli_avvio()
        self._mappa, self._mancanti_indice = mappa, mancanti
        # Un click o una spunta possono essere arrivati PRIMA di questa
        # consegna, quando la mappa era ancora vuota: `_assicura_volti`
        # li ha gia' segnati come "risposta consegnata" (zero volti) sul
        # fotogramma corrente. Senza invalidare quel segno qui, la pagina
        # continuerebbe a dire "nessun volto" anche dopo che la mappa vera
        # e' arrivata, finche' non si cambia fotogramma e si torna --
        # esattamente la frase falsa che questo lavoro esiste per togliere.
        if self._frame_corrente is not None and self._volti_di == self._frame_corrente:
            self._volti_di = None
            self._assicura_volti()
            # La riga di stato puo' aver detto "sto ancora indicizzando"
            # per questo click: se la mappa appena arrivata ha chiuso
            # quella ragione, il messaggio vecchio non deve restare a
            # schermo in attesa di un secondo click che lo aggiorni.
            if self.etichetta_stato.text() == testi.ESTRAZIONE_INDICE_IN_CORSO:
                self.etichetta_stato.setText(self._perche_nessun_volto())
        # I volti che l'indice non copre fanno partire la passata da soli:
        # QUI, non in _ricostruisci_mappa, perche' e' questo slot -- non
        # quello che accoda il lavoro -- a ricevere `_mancanti_indice`
        # com'e' DAVVERO (in _ricostruisci_mappa e' ancora la lista appena
        # svuotata). La consegna che `_su_job_finito` marca come "quella
        # del nostro ricaricamento" (sulla generazione, vedi il costruttore)
        # non ne fa partire un secondo, anche se i mancanti sono rimasti --
        # altrimenti un file non indicizzabile rilancerebbe la stessa
        # passata a ogni consegna.
        if generazione == self._generazione_indicizzazione_da_consumare:
            self._generazione_indicizzazione_da_consumare = None
        elif self._mancanti_indice:
            self.avvia_indicizzazione_volti()
        # `_mappa_affidabile()`: con `_mancanti_indice` non vuoto la mappa
        # sottostima ancora i volti dei file non indicizzati, e azzerare un
        # fotogramma su quel conteggio provvisorio sarebbe il difetto
        # opposto -- un rettangolo vero cancellato perche' l'indicizzazione
        # non e' ancora arrivata; con un occupante vivo e' uno snapshot in
        # corsa contro uno scrittore vero.
        if self._mappa_affidabile():
            self._riconcilia_rapporto()

    def _fotogrammi_divergenti(self):
        """I fotogrammi il cui conteggio nel rapporto non e' quello che la
        mappa gli attribuisce.

        Il confronto e' per CONTEGGIO: prende un volto aggiunto o tolto ma
        non uno modificato -- un volto rilandmarkato cambia rect e posa, non
        il numero.
        """
        fuori = []
        for v in self._voci:
            if not isinstance(v, dict) or not isinstance(v.get("nome"), str):
                continue
            atteso = len(self._mappa.get(v["nome"]) or [])
            n = v.get("n_volti")
            if not isinstance(n, int) or n != atteso:
                fuori.append(v["nome"])
        return fuori

    def _riconcilia_rapporto(self):
        """Riscrive a zero volti le righe che la mappa smentisce.

        Solo quelle che si risolvono a ZERO: un fotogramma i cui allineati
        sono stati cancellati si sistema da solo, senza chiedere niente a
        nessuno. I divergenti che hanno ancora volti richiederebbero una
        risposta del servizio per fotogramma -- si contano soltanto, e il
        bottone "Rebuild report" resta la strada per sistemarli tutti in
        una passata sola.

        Il chiamante (`_su_mappa_pronta`) gia' verifica `_mappa_affidabile()`
        prima di entrare qui: non si ripete il controllo, per non lasciare
        una seconda copia che un giorno resti indietro rispetto alla prima.
        Vale comunque la pena dire perche' quel controllo esiste --
        "occupata" non e' solo un job o una sessione manuale DI QUESTA
        pagina: vede anche un occupante arrivato da un'altra pagina (la
        cura del faceset condivide lo stesso `aligned/` e lo stesso
        gestore) o registrato come esterno (`registra_occupante`).
        Chiunque sia, puo' aggiungere sul disco, in mezzo a una scansione
        gia' partita, un fotogramma che quella scansione non ha fatto in
        tempo a vedere: apparirebbe "senza volti" e verrebbe azzerato per
        sempre (ultima riga vince), anche se il file e' li'. Costa solo un
        giro: alla prossima apertura, a scrittura ferma, la riconciliazione
        lo riprende.
        """
        da_svuotare = [n for n in self._fotogrammi_divergenti()
                       if not self._mappa.get(n)]
        if not da_svuotare:
            return 0
        vecchie = dict((v["nome"], v) for v in self._voci
                       if isinstance(v, dict) and isinstance(v.get("nome"), str))
        nuove = []
        for nome in da_svuotare:
            voce = dict(vecchie[nome])
            voce.update({"nome": nome, "n_volti": 0, "volti": []})
            nuove.append(voce)
        try:
            for voce in nuove:
                indice_mod.scrivi_voce(self._cache_dir(), voce)
        except OSError:
            # Il rapporto e' un comodo, ricostruibile dal bottone: un disco
            # pieno non deve impedire di aprire la cartella.
            return 0
        nomi = set(da_svuotare)
        self._voci = [v for v in self._voci
                      if not (isinstance(v, dict) and v.get("nome") in nomi)]
        self._voci.extend(nuove)
        self.modello.aggiorna_voci(nuove)
        self._aggiorna_conteggi_filtro()
        return len(nuove)

    def _risolvi_fratelli(self, nome_frame):
        """I percorsi degli allineati di `nome_frame`, dalla mappa gia'
        costruita -- mai una nuova lettura del disco.

        METODO legato, non una mappa congelata: e' cosi' che
        `_apri_finestra_dettaglio` lo passa alla finestra di dettaglio, e
        ogni chiamata rilegge `self._mappa` COM'E' ORA -- se un
        `_ricostruisci_mappa` successivo la sostituisce, la finestra gia'
        aperta lo vede senza bisogno di reinserirlo.
        """
        return list(self._mappa.get(nome_frame) or [])

    def _assicura_volti(self):
        """I volti del fotogramma corrente, dalla mappa o dal servizio.

        Non avvia MAI il servizio da sola: la chiamano una spunta appena
        accesa, un click sulla tela, o un cambio di fotogramma con una
        spunta gia' accesa. Con la sola "Rect" accesa non viene chiamata,
        e la pagina non fa partire nessun processo -- che e' il
        comportamento di oggi, e il criterio da non perdere.
        """
        if self._frame_corrente is None or self._cartella is None:
            return
        if self._volti_di == self._frame_corrente:
            return
        percorsi = self._mappa.get(self._frame_corrente.name) or []
        if not percorsi:
            # Zero volti e' una risposta, non un'assenza di risposta: si
            # registra come consegnata (come il ramo sotto), cosi' non si
            # ripete la domanda a ogni cambio di spunta -- e soprattutto
            # non si paga l'avvio del servizio per farselo dire, che e'
            # il costo che questa mappa esiste per togliere.
            self._volti_correnti = []
            self._volti_di = self._frame_corrente
            self.tela.imposta_volti([])
            return
        cliente = self._cliente_dettaglio()
        if cliente is None:
            return
        self._frame_chiesto = self._frame_corrente
        # Il primo scambio col servizio costa l'import (~6 s): un cursore
        # d'attesa e' l'unico modo di dire che l'interfaccia non e' morta.
        # Il client e' ASINCRONO: il cursore si ripristina alla CONSEGNA
        # (`_su_volti_pronti`) o al guasto (`_su_dettaglio_fallito`), non
        # in un `finally` qui -- scatterebbe subito, prima che la
        # risposta arrivi. Un solo push anche se una richiesta ne
        # supera un'altra (cambio rapido di fotogramma): lo stack di Qt
        # va ripristinato lo stesso numero di volte che e' stato spinto.
        if not self._cursore_volti_attivo:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self._cursore_volti_attivo = True
        # La bandierina copre la consegna SINCRONA dei doppi di test (la
        # risposta arriva dentro `volti_del_frame`, prima che l'id sia
        # assegnato qui sotto); il client vero e' asincrono e usa l'id.
        self._attesa_diretta_volti = True
        try:
            self._id_volti_atteso = cliente.volti_del_frame(percorsi)
        finally:
            self._attesa_diretta_volti = False

    def _ripristina_cursore_volti(self):
        if self._cursore_volti_attivo:
            QApplication.restoreOverrideCursor()
            self._cursore_volti_attivo = False

    def _su_volti_pronti(self, dati):
        """Slot del client. Scarta le consegne che non riguardano il
        fotogramma mostrato ORA: si scorre la pellicola piu' in fretta di
        quanto un fotogramma si serva, e i landmark di due click fa sopra
        il fotogramma di adesso sarebbero la cosa peggiore che questa
        pagina possa disegnare -- lo stesso motivo per cui esiste il
        controllo in cima a `_su_anteprima_pronta`.

        ATTENZIONE: `volti_pronti` e' un segnale CONDIVISO. La finestra di
        dettaglio chiede i fratelli sullo STESSO client, per il fotogramma
        del volto che ha aperto -- che non e' detto sia questo: basta
        scorrere la pellicola a finestra aperta. Senza filtro i suoi volti
        finirebbero disegnati su questo fotogramma, in silenzio.
        Si accetta la sola risposta della richiesta in volo, riconosciuta
        per id -- il client e' ASINCRONO, quindi non basta piu' un
        booleano "dentro la chiamata": una consegna differita puo' arrivare
        a un giro dell'event loop di distanza, con un'altra gia' partita
        nel frattempo. La finestra si difende in un altro modo -- accetta
        la risposta che contiene il volto che ha aperto -- perche' li' la
        richiesta e' sua e la nostra le arriverebbe comunque.

        **La seconda guardia, `_frame_chiesto != _frame_corrente`, oggi e'
        RAGGIUNGIBILE in produzione**: con un client asincrono una consegna
        puo' arrivare dopo che il fotogramma e' gia' cambiato sotto di lei.
        Non e' piu' difesa in profondita', e' la difesa vera.
        """
        if not (self._attesa_diretta_volti or
                (self._id_volti_atteso is not None
                 and dati.get("id") == self._id_volti_atteso)):
            return
        self._id_volti_atteso = None
        self._ripristina_cursore_volti()
        if self._frame_chiesto != self._frame_corrente:
            return
        volti = [v._replace(maschera=self._maschera_colorata(v.nome_maschera))
                 for v in volti_mod.volti_da_risposta(dati)]
        self._volti_correnti = volti
        self._volti_di = self._frame_corrente
        # Una risposta e' arrivata: qualunque guasto precedente non
        # riguarda piu' lo stato di ORA.
        self._ultimo_guasto = None
        self.tela.imposta_volti(volti)
        self._aggiorna_rapporto_dal_servizio(dati)
        self._esegui_click_in_sospeso()

    def _esegui_click_in_sospeso(self):
        """Il click che `_su_volto_scelto` aveva dovuto rimandare perche'
        la richiesta era ancora in volo: si riesegue ORA, sul fotogramma
        per cui era stato fatto. Se nel frattempo la pellicola e' scorsa
        altrove si scarta in silenzio, come un gesto che non ha piu'
        senso -- e' lo stesso criterio di `_su_anteprima_pronta`."""
        sospeso, self._click_in_sospeso = self._click_in_sospeso, None
        if sospeso is None:
            return
        frame, punto = sospeso
        if frame != self._frame_corrente:
            return
        self._su_volto_scelto(None, punto)

    def _aggiorna_rapporto_dal_servizio(self, dati):
        """La riga del fotogramma mostrato, riscritta se non concorda.

        Costa zero richieste: e' la risposta che la pagina ha gia' chiesto
        per disegnare i landmark. `posa` e `lato` arrivano da li' perche'
        calcolarli qui vorrebbe dire importare facelib in gui/.

        Un volto senza `rect` valido (il servizio lo lascia None quando il
        file non ha un rettangolo sorgente registrato) CONTA comunque, col
        rettangolo normalizzato a [0,0,0,0]: e' la stessa canonicalizzazione
        di `ExtractIndex._volto_da_dfl`, l'altro produttore dello stesso
        conteggio -- quello che alimenta la mappa e la riconciliazione
        globale. Escluderlo farebbe divergere `n_volti` dal numero di
        percorsi che la mappa attribuisce a questo fotogramma, e
        `_riconcilia_rapporto` non lo saprebbe mai correggere: sana solo le
        righe che la mappa smentisce A ZERO, mai un conteggio parziale. Un
        rettangolo degenere resta comunque escluso dal disegno e dal click
        da `volti_mod.rect_utilizzabile`, che e' il posto giusto per quella
        decisione -- qui si conta un volto, non un rettangolo disegnabile.

        Si riscrive SOLO se qualcosa cambia: il formato e' append-only, e
        una riga identica per ogni fotogramma sfogliato farebbe crescere il
        rapporto senza dire niente di nuovo.

        Salta se la mappa non e' affidabile (`_mappa_affidabile()`, la
        stessa domanda del chiamante di `_riconcilia_rapporto`): i percorsi
        che il servizio ha appena letto vengono da uno snapshot della mappa
        (`self._mappa`), e un volto scritto sul disco dopo quello snapshot,
        o non ancora indicizzato, non ci sarebbe dentro -- riscrivere ora
        peggiorerebbe la riga invece di correggerla. Il prossimo giro, a
        mappa affidabile, la riprende.
        """
        if not self._mappa_affidabile():
            return
        nome = self._frame_corrente.name
        nuovi_volti = []
        for voce in dati.get("volti") or ():
            rect = voce.get("rect")
            if isinstance(rect, list) and len(rect) == 4:
                rect = [int(v) for v in rect]
            else:
                rect = [0, 0, 0, 0]
            nuovi_volti.append({"rect": rect,
                                "posa": [float(v) for v in
                                        (voce.get("posa") or [0.0, 0.0, 0.0])],
                                "lato": int(voce.get("lato") or 0)})
        vecchia = next((v for v in self._voci
                        if isinstance(v, dict) and v.get("nome") == nome), None)
        if vecchia is not None and vecchia.get("volti") == nuovi_volti:
            return
        try:
            st = self._frame_corrente.stat()
        except OSError:
            # Il frame e' sparito fra la lettura e qui: niente da scrivere,
            # `dimensione`/`mtime` sarebbero comunque inventati.
            return
        nuova = dict(vecchia or {"formato": indice_mod.FORMATO, "nome": nome,
                                 "luminanza": None, "stato": "automatico",
                                 "motore": None})
        # `dimensione`/`mtime` sono la CHIAVE del rapporto
        # (mainscripts/ExtractReport.py, docstring): un sort che rinomina i
        # frame non deve invalidarlo. Questa via -- il click sulla tela,
        # non un passo automatico -- le ometteva.
        nuova.update({"nome": nome, "n_volti": len(nuovi_volti), "volti": nuovi_volti,
                     "dimensione": st.st_size, "mtime": st.st_mtime_ns})
        try:
            indice_mod.scrivi_voce(self._cache_dir(), nuova)
        except OSError:
            # Il rapporto e' un comodo, ricostruibile dal bottone: un disco
            # pieno non deve impedire di navigare.
            return
        self._voci = [v for v in self._voci
                      if not (isinstance(v, dict) and v.get("nome") == nome)]
        self._voci.append(nuova)
        self.modello.aggiorna_voci([nuova])
        self._aggiorna_conteggi_filtro()
        if self._pixmap_corrente is not None:
            self.tela.mostra(self._pixmap_corrente, self._rects_di(self._frame_corrente),
                             None, _dimensione_nativa(self._frame_corrente,
                                                      self._pixmap_corrente))

    def _maschera_colorata(self, nome):
        """Il file annunciato dal servizio, aperto e tinto UNA VOLTA. None
        per un file che non c'e' o non si apre: il volto resta cliccabile
        e i suoi landmark disegnabili anche senza la sua maschera."""
        if not nome:
            return None
        immagine = QImage(str(self._workdir_dettaglio() / nome))
        return maschera_mod.pellicola_colorata(immagine, COLORE_MASCHERA)

    def _workdir_dettaglio(self):
        """Una cartella per PAGINA: due schede possono essere aperte su due
        progetti diversi, e un workdir condiviso farebbe collidere i nomi
        dei file delle maschere."""
        if self._cartella_dettaglio is None:
            self._cartella_dettaglio = Path(tempfile.mkdtemp(prefix="dfl_dettaglio_"))
        return self._cartella_dettaglio

    def _cliente_dettaglio(self):
        """Costruito al primo bisogno: finche' nessuno accende una spunta
        o clicca sulla tela, questa pagina non ha nessun client e non ha
        avviato nessun processo."""
        if self._cliente is None:
            from gui.faceset.dettaglio import ClienteDettaglio
            self._cliente = ClienteDettaglio(self._workdir_dettaglio(), parent=self)
            self._cliente.volti_pronti.connect(self._su_volti_pronti)
            self._cliente.fallito.connect(self._su_dettaglio_fallito)
            self._cliente.salvato.connect(self._su_volto_salvato)
        return self._cliente

    def _su_volto_salvato(self, _dati):
        """La finestra di dettaglio ha riscritto un allineato.

        Due conseguenze, e nessuna delle due chiede codice nuovo: la
        (dimensione, mtime) del file e' cambiata, quindi
        `_ricostruisci_mappa` lo trova fra i mancanti e fa ripartire
        l'indicizzazione per quel solo volto; e i volti in mano alla
        pagina sono vecchi, quindi si richiedono di nuovo -- ed e' la
        risposta a riscrivere la riga del rapporto
        (`_aggiorna_rapporto_dal_servizio`).

        Il conteggio NON cambia mai in questo caso: e' esattamente il buco
        che la riconciliazione per conteggio non copre, e questo slot ne
        e' l'altra meta'.
        """
        self._ricostruisci_mappa()
        self._dimentica_volti()
        self._su_sovrapposizioni()

    def _su_dettaglio_fallito(self, motivo, codice=None):
        """Il servizio non ha risposto -- morto per inattivita', o un file
        illeggibile.

        Continua a non scrivere sulla riga di stato: la richiesta
        successiva riavvia il servizio, e un avviso a ogni fotogramma
        scorso sarebbe rumore. E continua a non svuotare i volti: questo
        segnale porta anche i fallimenti di `open` (la finestra di
        dettaglio), e svuotare li' cancellerebbe i volti della revisione
        per un guasto che non li riguarda.

        Ma ORA ricorda l'ultimo guasto: e' `_su_volto_scelto` a usarlo, e
        senza di lui il click accusava il rapporto di essere vecchio anche
        quando era giusto -- la diagnosi falsa che ha aperto questo ciclo.

        E ripristina il cursore d'attesa dei volti, se acceso: il segnale
        e' condiviso e non porta l'id di chi e' caduto, quindi non si sa
        SE il guasto riguardi proprio la nostra richiesta -- ma un cursore
        che resta acceso per sempre e' peggio di uno spento un giro
        troppo presto.
        """
        self._ultimo_guasto = (motivo, codice)
        self._id_volti_atteso = None
        self._ripristina_cursore_volti()
        # Un click rimasto in sospeso non ha piu' niente da aspettare: si
        # scioglie qui, mostrando il guasto vero invece di restare muto
        # fino al click successivo.
        sospeso, self._click_in_sospeso = self._click_in_sospeso, None
        if sospeso is not None and sospeso[0] == self._frame_corrente:
            self.etichetta_stato.setText(self._perche_nessun_volto())

    def _su_volto_scelto(self, percorso, punto):
        """Il click sulla tela in revisione.

        `percorso` None significa «la tela non aveva volti caricati e non
        sapeva rispondere»: si recuperano e si RIPETE la ricerca su quel
        punto, cosi' il click funziona con qualunque combinazione di
        spunte -- la spunta decide cosa si disegna, non cosa esiste.

        Ma prima si verifica che il punto cada dentro almeno uno dei
        rettangoli del RAPPORTO: fuori da ognuno di loro il click e' un
        gesto neutro (cliccare sul fondo del fotogramma) e non deve
        scrivere niente ne' chiedere niente al servizio -- specialmente su
        un progetto senza `aligned/`, dove ogni click pagherebbe altrimenti
        l'avvio del client di dettaglio.
        """
        if percorso is None:
            if not self.tela.punto_in_rapporto(punto):
                return
            self._assicura_volti()
            if self._volti_di != self._frame_corrente:
                # La richiesta e' partita ma non e' ancora tornata (client
                # ASINCRONO): il click non si inghiotte, si rimanda alla
                # consegna -- `_esegui_click_in_sospeso`, chiamato da
                # `_su_volti_pronti`. La riga di stato deve dirlo SUBITO,
                # non restare quella di prima del click: senza, un secondo
                # click durante l'attesa resta muto quanto il primo.
                self._click_in_sospeso = (self._frame_corrente, punto)
                self.etichetta_stato.setText(self._perche_nessun_volto())
                return
            volto = volti_mod.volto_al_punto(self._volti_correnti, *punto)
            if volto is None:
                if self._frame_corrente is not None:
                    self.etichetta_stato.setText(self._perche_nessun_volto())
                return
            percorso = volto.percorso
        self._apri_finestra_dettaglio(percorso)

    def _perche_nessun_volto(self):
        """Quale delle cinque ragioni e', invece di indovinarne una sola.

        L'ordine conta: «il servizio non ha risposto» va provato per
        primo, perche' in quel caso non sappiamo niente del disco e ogni
        altra frase sarebbe un'affermazione senza prova. `_volti_di` e' il
        discriminante e c'era gia': vale il fotogramma corrente SOLO se una
        risposta e' arrivata. Fra un guasto e l'indicizzazione in corso
        c'e' un terzo stato che il client sincrono di prima non poteva mai
        produrre: la richiesta e' partita e basta aspettare, non un guasto
        e non l'indice -- `_id_volti_atteso` e' non-None SOLO mentre una
        richiesta `frame` e' in volo.
        """
        nome = self._frame_corrente.name
        if self._volti_di != self._frame_corrente:
            motivo, codice = self._ultimo_guasto or (None, None)
            if codice is not None or motivo is not None:
                return testi.estrazione_dettaglio_non_risponde(
                    testi.dettaglio_guasto(codice, motivo))
            if self._id_volti_atteso is not None:
                return testi.ESTRAZIONE_VOLTI_IN_ARRIVO
            return testi.ESTRAZIONE_INDICE_IN_CORSO
        if self._mancanti_indice:
            return testi.ESTRAZIONE_INDICE_IN_CORSO
        if self._volti_correnti:
            return testi.ESTRAZIONE_NESSUN_VOLTO_SOTTO_IL_PUNTO
        return testi.estrazione_rapporto_piu_vecchio(nome)

    def _apri_finestra_dettaglio(self, percorso):
        if self._finestra_dettaglio is None:
            from gui.dettaglio.finestra import FinestraDettaglio
            # Il client si costruisce prima e si passa: la finestra non ne
            # fa uno suo, o nascerebbe un secondo processo figlio che
            # importa torch mentre il primo e' vivo. E `pronto` non si
            # collega qui: la finestra ci si e' gia' collegata da se' alla
            # costruzione, e un secondo ascoltatore che richiama `mostra`
            # raddoppia il giro dei fratelli, e ogni giro
            # rilegge dal disco i dati DFL di tutti.
            cliente = self._cliente_dettaglio()
            self._finestra_dettaglio = FinestraDettaglio(
                self._workdir_dettaglio(), cliente=cliente)
            # Il METODO, non una mappa gia' calcolata: senza, la striscia
            # dei fratelli resterebbe vuota in silenzio -- lo stesso
            # ripiego di un volto senza `source_filename`, mai un
            # `TypeError` rumoroso come prima di questa riga.
            self._finestra_dettaglio.imposta_risolutore_fratelli(self._risolvi_fratelli)
        # I fratelli dello STESSO fotogramma: le frecce sfogliano il
        # gruppo invece di una lista vuota.
        self._finestra_dettaglio.imposta_ordine(
            volti_mod.percorsi(self._volti_correnti))
        # `aligned/` sta DENTRO la cartella che si sta guardando, quindi i
        # fotogrammi sono proprio lei. Si passa anche se non esiste -- chi
        # decide se la modifica e' possibile e' la finestra, in un posto
        # solo.
        self._finestra_dettaglio.imposta_frame_dir(self._cartella)
        self._finestra_dettaglio.show()
        self._finestra_dettaglio.raise_()
        self._finestra_dettaglio.activateWindow()
        # Il cambio di volto sta tutto dentro `apri_volto`: la domanda
        # sulle modifiche vive, il disegno e la richiesta al servizio. Da
        # qui si disegnava e si chiedeva in due passi, e la domanda non la
        # faceva nessuno. La finestra si mostra e si porta davanti
        # comunque: se rifiuta, e' proprio quella che tiene il lavoro che
        # l'utente ha appena deciso di non perdere.
        # Il client e' asincrono: la clessidra qui copre solo l'istante
        # del click, un lampo. La vera attesa -- il primo scambio costa
        # l'import -- e' coperta dall'indicatore della finestra stessa
        # (`FinestraDettaglio.indicatore_attesa`, sopra la tela), non da
        # questo cursore.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._finestra_dettaglio.apri_volto(percorso)
        finally:
            QApplication.restoreOverrideCursor()

    def _percorsi_visibili(self):
        """I percorsi che la pellicola mostra ORA -- rispetta il filtro
        acceso, cosi' la navigazione manuale scorre la stessa fetta che
        l'utente sta guardando."""
        m = self.modello
        return [m.data(m.index(i, 0), RUOLO_PERCORSO) for i in range(m.rowCount())]

    # -- ponte verso la cura del faceset ----------------------------------

    def _su_volti_del_frame(self):
        if self._frame_corrente is not None:
            self.richiesta_volti_del_frame.emit(self._lato, self._frame_corrente.name)

    def mostra_frame(self, lato, nome_frame):
        """Porta la pagina sul frame `nome_frame` di data_<lato>.

        Torna False -- MAI scrivendo la riga di stato di questa pagina, sul
        rifiuto -- se il file non e' sul disco (o non ha un'estensione che
        questa pagina raccoglie): la sonda guarda `data_<lato>` PRIMA di
        spostare lato o cartella, cosi' quel rifiuto non lascia la pagina
        spostata su un lato diverso -- e su questa pagina un cambio di lato
        passa da `apri()`, che interrompe una sessione manuale in corso. Il
        motivo del rifiuto non si scrive qui: come il suo gemello
        `PaginaCuraFaceset.mostra_solo_frame`, lascia dirlo a chi ha
        chiamato -- `MainWindow._vai_al_frame` lo scrive sulla pagina che
        l'utente sta DAVVERO guardando, non su questa, che potrebbe essere
        una scheda chiusa da nessuno.

        Il cambio di lato, quando la sonda passa, passa da `_cambia_lato` e
        non da `apri()` diretta: e' la strada che avverte quando una
        sessione manuale in corso viene chiusa (`sessione_interrotta`), e
        cambiarla in silenzio sotto un rettangolo gia' agganciato e'
        esattamente il difetto gia' corretto li'.
        """
        if self._progetto is None:
            return False
        bersaglio = self._progetto / ("data_%s" % lato) / nome_frame
        if bersaglio.suffix.lower() not in ESTENSIONI or not bersaglio.is_file():
            return False
        if lato != self._lato:
            self._cambia_lato(lato)
        if bersaglio not in self._percorsi_visibili():
            # Un filtro acceso puo' nascondere proprio il frame chiesto.
            # Il bottone si spunta a mano: `_su_filtro` applica il filtro al
            # modello, ma chi lo chiama da codice non passa dal QButtonGroup.
            self._bottoni_filtro["tutti"].setChecked(True)
            self._su_filtro("tutti")
        percorsi = self._percorsi_visibili()
        if bersaglio not in percorsi:
            return False
        index = self.modello.index(percorsi.index(bersaglio), 0)
        self.pellicola.setCurrentIndex(index)
        self.pellicola.scrollTo(index)
        self._su_frame_scelto(bersaglio)
        self._rigenera_comandi()
        return True

    def mostra_messaggio(self, testo):
        """La riga di stato scritta da fuori: e' dove finisce il motivo per
        cui una navigazione incrociata si e' fermata."""
        self.etichetta_stato.setText(testo)

    def _su_frame_scelto(self, percorso):
        # Il fotogramma cambia: i volti del precedente non appartengono
        # piu' a niente.
        self._dimentica_volti()
        if self.servizio is not None:
            self._carica_frame(percorso)
        else:
            self._mostra_anteprima(percorso)
        # `_frame_corrente` e' appena cambiato in entrambi i rami: senza
        # questa riga il bottone "Show faces from this frame" restava
        # spento fino al prossimo comando che ridisegna la barra per
        # tutt'altra ragione -- un clic sulla pellicola non ne era uno.
        self._rigenera_comandi()
        # Una spunta gia' accesa richiede i volti del fotogramma nuovo.
        self._su_sovrapposizioni()

    def _mostra_anteprima(self, percorso):
        """La revisione: il frame come sta su disco, con sopra i rettangoli
        che il rapporto ha registrato. Nessun processo figlio e nessun
        torch -- il servizio serve a estrarre, non a guardare.

        La decodifica passa dal pool di thread e torna via segnale: un 4K
        decodificato nello slot del clic bloccherebbe l'interfaccia.
        """
        if not isinstance(percorso, Path):
            return
        self._frame_corrente = percorso
        immagine = self.decodificatore_anteprima.in_cache(percorso, LATO_ANTEPRIMA)
        if immagine is None:
            self.decodificatore_anteprima.richiedi(percorso, LATO_ANTEPRIMA)
        else:
            self._su_anteprima_pronta(percorso, LATO_ANTEPRIMA, immagine)
        testo = testi.estrazione_frame_scelto(percorso.name, len(self._rects_di(percorso)))
        if self._nota_sessione_interrotta is not None:
            # `apri()` l'ha appena scritto, e mostrare il frame lo
            # sovrascriverebbe: si compone, non si sostituisce -- lo
            # stesso principio di `_aggiorna_stato_manuale`.
            testo = testi.estrazione_componi_stato(self._nota_sessione_interrotta, testo)
            self._nota_sessione_interrotta = None
        self.etichetta_stato.setText(testo)

    def _su_anteprima_pronta(self, percorso, _lato, immagine):
        """Slot del decodificatore. Scarta le consegne in ritardo: si
        scorre la pellicola piu' in fretta di quanto un 4K si decodifichi, e
        senza questo controllo il frame di due clic fa comparirebbe sopra
        quello scelto adesso."""
        if percorso != self._frame_corrente or self.servizio is not None:
            return
        pixmap = QPixmap.fromImage(immagine)
        if pixmap.isNull():
            return
        self._pixmap_corrente = pixmap
        self.tela.mostra(pixmap, self._rects_di(percorso), None,
                         _dimensione_nativa(percorso, pixmap))
        self._aggiorna_messaggio_vuoto()

    def _rects_di(self, percorso):
        """I rettangoli dei volti che il rapporto registra per questo frame,
        in coordinate del frame. Regge una voce scritta a meta'."""
        for v in self._voci:
            if not isinstance(v, dict) or v.get("nome") != percorso.name:
                continue
            fuori = []
            for volto in (v.get("volti") or []):
                if isinstance(volto, dict) and isinstance(volto.get("rect"), list) \
                        and len(volto["rect"]) == 4:
                    fuori.append(tuple(volto["rect"]))
            return fuori
        return []

    def _vai_a(self, passo):
        """Lo sfogliamento generico -- `_prossimo_frame` (sotto) e' il
        nome che `_entra_modalita_manuale` chiama, `passo=-1` e' il suo
        gemello per `precedente`."""
        percorsi = self._percorsi_visibili()
        if not percorsi:
            self._frame_corrente = None
            self._pixmap_corrente = None
            self.tela.mostra(None, None, None)
            self._aggiorna_messaggio_vuoto()
            # Nessun frame da caricare vuol dire nessuna `_rileva` in
            # partenza: la risposta che spegnerebbe la barra (in `_su_volti`)
            # non arrivera' mai. Una barra di caricamento accesa su una
            # pagina che non aspetta piu' niente mente come mentiva
            # l'assenza di barra prima di questo lavoro.
            self.pila.togli_avvio()
            return
        if self._frame_corrente in percorsi:
            i = (percorsi.index(self._frame_corrente) + passo) % len(percorsi)
        else:
            i = 0
        self._carica_frame(percorsi[i])

    def _prossimo_frame(self):
        self._vai_a(+1)

    def _carica_frame(self, percorso):
        if self.servizio is None:
            return
        # `shape` si scarta apposta: il raster che il servizio scrive E' il
        # frame a risoluzione nativa (mainscripts/ExtractManual.py::_op_frame
        # ricodifica senza ridimensionare), quindi la dimensione del pixmap
        # e' gia' quella del frame -- per questo qui sotto mostra() non
        # passa nessuna dimensione: senza di lei Tela la ricava dal pixmap
        # da sola (gui/estrazione/tela.py::trasformazione). Tenerne due
        # sarebbe tenere due verita' da riconciliare.
        raster, _forma = self.servizio.frame(percorso)
        self._frame_corrente = percorso
        self._rect_corrente = None
        self._landmarks_correnti = None
        # Il blocco NON deve sopravvivere al
        # cambio di frame. Senza questa riga, un rettangolo bloccato sul
        # frame precedente resta bloccato su questo -- che non ha ancora
        # nessun rettangolo -- e Tela.mouseMoveEvent esce subito da
        # bloccato: il mouse smette di rispondere proprio sui fotogrammi
        # dove il rilevatore non aggancia (206 su 983 nel materiale
        # dell'utente), esattamente il caso per cui questa sessione esiste.
        # mainscripts/Extractor.py:574 fa lo stesso ad ogni transizione.
        self.tela.blocca(False)
        pixmap = None
        if raster is not None and self._trasporto is not None:
            candidata = QPixmap(str(self._trasporto.workdir / raster))
            if not candidata.isNull():
                pixmap = candidata
        self._pixmap_corrente = pixmap
        self.tela.mostra(pixmap, None, None)
        # La riga di stato si compone in _su_volti, non qui: _rileva()
        # innesca sempre una risposta (trovato/nessun-volto/guasto), e
        # comporla due volte -- qui senza sapere ancora l'esito, poi di
        # nuovo in _su_volti -- consumerebbe la nota di salvataggio nella
        # prima chiamata senza che l'utente la vedesse mai (misurato: la
        # prima versione di M8 faceva esattamente questo).
        # Il rilevamento automatico all'apertura del frame -- la miglioria
        # concordata: se il rilevatore aggancia, il lavoro manuale e' un
        # tasto solo, non si insegue il volto col mouse.
        self._rileva(rect=None)

    def _rileva(self, rect):
        """Chiede i volti al motore vero. `rect=None` e' il rilevamento
        automatico (il rilevatore cerca da solo), un `rect` esplicito e' il
        rettangolo che l'utente sta muovendo -- si salta il rilevatore e si
        va direttamente all'allineatore, che e' cio' che rende il gesto
        fluido. Passa da `rileva_quando_puoi`, mai da `rileva`: bloccare
        l'interfaccia a ogni pressione di freccia e' esattamente il difetto
        che `rileva_quando_puoi` esiste per togliere.

        E' anche l'unico posto da cui la barra dei motori si accende: prima
        stava nei tre chiamanti che aspettano un motore nuovo, che pero'
        non condividono questa guardia -- senza frame corrente la
        rilevazione non parte, `_su_volti` non arriva mai e la barra resta
        accesa per tutta la sessione. Solo `_motori_da_caricare` la
        accende, mai ogni rilevazione: il rettangolo trascinato col mouse
        passa di qui a ogni pixel, e una barra che dice «sto caricando i
        motori» lampeggerebbe a ogni movimento dicendo il falso."""
        if self.servizio is None or self._frame_corrente is None:
            return
        if self._motori_da_caricare:
            self._motori_da_caricare = False
            self.pila.mostra_avvio(testi.ESTRAZIONE_CARICAMENTO_MOTORI)
        frame = self._frame_corrente
        self.servizio.rileva_quando_puoi(
            frame, rect,
            self._face_type_corrente(),
            self._accurato,
            lambda volti, f=frame, r=rect: self._su_volti(f, r, volti),
            rilevatore=self._rilevatore, allineatore=self._allineatore,
            tieni_in_memoria=self._tieni_in_memoria)

    def _face_type_corrente(self):
        """Il face type della sessione nella forma lunga del protocollo.
        Uno solo, letto da `rileva` e da `libera`: la voce di cache
        dell'allineatore dipende da questo valore, e due letture diverse
        farebbero liberare al figlio una voce che non e' quella corrente."""
        return _FACE_TYPE_LUNGO.get(self._parametri_manuale.get("face-type"),
                                    "whole_face")

    def _su_rilevatore_scelto(self, _indice):
        """Cambiare rilevatore ricomincia dal rilevamento AUTOMATICO
        (rect=None): e' il rilevatore a decidere il rettangolo, e tenere
        quello del motore precedente nasconderebbe proprio cio' che si sta
        provando a vedere."""
        self._rilevatore = _chiave_valida(self.selettore_rilevatore.currentData(),
                                          MotoriCatalog.CHIAVI_RILEVATORI,
                                          MotoriCatalog.DEFAULT_RILEVATORE)
        self._ricorda_motori()
        # Cambiare motore ne costruisce uno nuovo: e' la stessa attesa
        # dell'ingresso in sessione, e la barra la accende `_rileva` se una
        # rilevazione parte davvero.
        self._motori_da_caricare = True
        self._rileva(None)

    def _su_allineatore_scelto(self, _indice):
        """L'allineatore non sceglie il rettangolo: si resta su quello
        corrente e si guardano i landmark nuovi sullo stesso volto -- che
        e' il confronto per cui questi controlli esistono."""
        self._allineatore = _chiave_valida(self.selettore_allineatore.currentData(),
                                           MotoriCatalog.CHIAVI_ALLINEATORI,
                                           MotoriCatalog.DEFAULT_ALLINEATORE)
        self._ricorda_motori()
        # Stessa attesa di cambiare rilevatore: un modello nuovo da caricare.
        self._motori_da_caricare = True
        self._rileva(self.tela.rettangolo())

    def _su_memoria_scelta(self, acceso):
        """Togliere la spunta libera SUBITO i motori non correnti, non alla
        prossima scelta.

        Passa dall'operazione `libera`, non da un `rileva`: la politica
        viaggia anche sul comando `rileva`, ma `_rileva` esce subito senza
        `_frame_corrente` (una pellicola filtrata a vuoto, l'istante prima
        del primo fotogramma) e li' non partiva NIENTE -- proprio nello
        stato in cui uno toglie la spunta per fare posto a un training, con
        il testo della spunta che promette "right away". `libera` non ha
        bisogno di nessun fotogramma e non ridisegna niente sotto le mani
        dell'utente, quindi non serve nemmeno ri-rilevare.

        Rimetterla non ha niente da mandare: nessun motore da liberare, e
        la politica nuova arriva col prossimo `rileva`."""
        self._tieni_in_memoria = bool(acceso)
        self._ricorda_motori()
        if self.servizio is None or self._tieni_in_memoria:
            return
        self.servizio.libera_altri(self._rilevatore, self._allineatore,
                                   self._face_type_corrente())

    def _ricorda_motori(self):
        """Le tre scelte nella memoria del progetto, che le FONDE con le
        altre risposte invece di sostituirle (gui/progetti.py). Protetta
        come ogni scrittura di project.json fatta da uno slot: PyQt5
        trasforma un'eccezione qui in qFatal e si porta via il processo con
        dentro ogni altro lavoro aperto."""
        if self._progetto is None or self._passo_manuale is None:
            return
        try:
            ricorda_risposte(self._progetto, self._passo_manuale, {
                CHIAVE_MEMORIA_RILEVATORE: self._rilevatore,
                CHIAVE_MEMORIA_ALLINEATORE: self._allineatore,
                CHIAVE_MEMORIA_TIENI: self._tieni_in_memoria})
        except Exception as errore:
            # UNA volta per sessione. Questo slot parte a ogni cambio di
            # tendina e a ogni click sulla spunta: con la cartella del
            # progetto in sola lettura, un dialogo modale per gesto rende
            # i controlli inutilizzabili proprio mentre l'utente li sta
            # usando. Il guasto e' lo stesso a ogni tentativo -- dirlo la
            # seconda volta non aggiunge niente e toglie la pagina.
            if self._avvisato_memoria_non_scritta:
                return
            self._avvisato_memoria_non_scritta = True
            QMessageBox.warning(self, testi.TITLE_PROJECT_ACTION_FAILED,
                                testi.msg_project_action_failed(str(errore)))

    def _applica_motori_ricordati(self, nome_passo):
        """Le tre scelte che il progetto ricorda, dentro lo stato e dentro
        i controlli -- a segnali BLOCCATI: siamo prima del primo
        `_prossimo_frame`, quindi uno slot che partisse da qui chiamerebbe
        `_rileva` senza frame corrente (innocuo) e soprattutto
        riscriverebbe la memoria del progetto durante la sua stessa
        lettura."""
        try:
            ricordate = risposte_ricordate(self._progetto, nome_passo)
        except Exception:
            ricordate = {}
        # Un motore ricordato i cui pesi mancano ricade sul default,
        # esattamente come una chiave sconosciuta: "non selezionabile"
        # vale anche quando la selezione arriverebbe dalla memoria del
        # progetto, non solo dal click sulla tendina -- altrimenti la
        # sessione entrerebbe gia' con un motore che poi fallisce a
        # caricarsi, la stessa sorpresa che questo lavoro deve togliere.
        self._rilevatore = _chiave_valida(
            ricordate.get(CHIAVE_MEMORIA_RILEVATORE),
            tuple(k for k in MotoriCatalog.CHIAVI_RILEVATORI
                 if k not in self._pesi_mancanti_rilevatori),
            MotoriCatalog.DEFAULT_RILEVATORE)
        self._allineatore = _chiave_valida(
            ricordate.get(CHIAVE_MEMORIA_ALLINEATORE),
            tuple(k for k in MotoriCatalog.CHIAVI_ALLINEATORI
                 if k not in self._pesi_mancanti_allineatori),
            MotoriCatalog.DEFAULT_ALLINEATORE)
        self._tieni_in_memoria = _spunta_ricordata(ricordate.get(CHIAVE_MEMORIA_TIENI))
        self._mostra_motori()

    def _mostra_motori(self):
        """Lo stato dentro i tre controlli, a segnali bloccati (vedi
        `_applica_motori_ricordati`)."""
        for controllo, valore in ((self.selettore_rilevatore, self._rilevatore),
                                  (self.selettore_allineatore, self._allineatore)):
            controllo.blockSignals(True)
            controllo.setCurrentIndex(max(controllo.findData(valore), 0))
            controllo.blockSignals(False)
        self.spunta_memoria.blockSignals(True)
        self.spunta_memoria.setChecked(self._tieni_in_memoria)
        self.spunta_memoria.blockSignals(False)

    def _su_volti(self, frame, rect_chiesto, volti):
        self.pila.togli_avvio()
        if self.servizio is None or frame != self._frame_corrente:
            return          # consegna in ritardo, o sessione gia' chiusa
        if not volti:
            self._landmarks_correnti = None
            self.tela.imposta_landmarks(None)
            errore = self.servizio.ultimo_errore
            self._aggiorna_stato_manuale(
                testi.estrazione_servizio_guasto(errore) if errore
                else testi.ESTRAZIONE_NESSUN_VOLTO)
            # Il traceback vero va nel tooltip della stessa etichetta, non
            # in una superficie nuova. `getattr` con ripiego, in una riga a
            # parte: un servizio finto piu' vecchio di questo lavoro (nei
            # test, e chiunque implementi il protocollo altrove) puo' non
            # avere `ultimo_stderr` affatto, e questo e' chiamato da uno
            # slot Qt -- un'eccezione li' chiama qFatal e si porta via il
            # processo con dentro ogni training aperto.
            righe_stderr = getattr(self.servizio, _ATTRIBUTO_ULTIMO_STDERR, [])
            self.etichetta_stato.setToolTip(
                testi.estrazione_servizio_guasto_tooltip(righe_stderr)
                if errore else "")
            self._rigenera_comandi()
            return
        rect, landmarks = volti[0]
        self._landmarks_correnti = landmarks
        self.tela.imposta_landmarks(landmarks)
        if rect_chiesto is None:
            # Solo il rilevamento AUTOMATICO puo' spostare il rettangolo:
            # con un rect esplicito la risposta ne e' l'eco, e riapplicarla
            # farebbe saltare indietro il rettangolo che l'utente sta
            # ancora muovendo (la risposta arriva dopo il movimento dopo).
            self._rect_corrente = tuple(rect)
            self.tela.imposta_rettangolo(self._rect_corrente)
            self.tela.blocca(True)
        self._aggiorna_stato_manuale(testi.ESTRAZIONE_VOLTO_TROVATO)
        self.etichetta_stato.setToolTip("")  # ripulisce un guasto precedente
        self._rigenera_comandi()

    def _aggiorna_stato_manuale(self, stato_rilevamento=None):
        """L'aiutante unico per la riga di stato della sessione manuale:
        compone la nota di salvataggio eventuale
        (mostrata una volta sola e poi consumata), il nome del fotogramma
        corrente (che la revisione mostra gia' fuori da qui, e qui si
        perdeva) e lo stato del rilevamento. Chiamata SOLO da `_su_volti`,
        mai da `_carica_frame`: quest'ultima non conosce ancora l'esito
        della richiesta appena partita, e comporre la riga li' consumerebbe
        la nota di salvataggio prima che l'utente la veda mai -- la prima
        versione di questa correzione faceva esattamente questo errore."""
        if self._frame_corrente is None:
            return
        nota = self._nota_salvataggio
        self._nota_salvataggio = None
        self.etichetta_stato.setText(testi.estrazione_stato_manuale(
            nota, self._frame_corrente.name, stato_rilevamento))

    def _su_rettangolo_cambiato(self, rect):
        self._rect_corrente = rect
        self._rileva(rect)

    def _su_blocco_cambiato(self, _acceso):
        self._rigenera_comandi()

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
            # NON accetta -- solleverebbe ValueError dentro _su_confermato --
            # raggiunto dalla QAction `conferma`,
            # tela.confermato essendo sparito -- ed e' uno slot: se solleva
            # chiama qFatal e si porta via il processo con dentro ogni
            # training aperto (N1 del ledger). Il try/except intorno al solo
            # int() e' la seconda guardia: un nome che superasse comunque
            # isdecimal() per una ragione che non ho previsto non deve
            # fermare gli altri file.
            if coda.isdecimal():
                try:
                    trovati.append(int(coda))
                except ValueError:
                    continue
        return max(trovati) + 1

    def _salva_corrente(self):
        """Il corpo di sempre di `_su_confermato`, senza lo sfogliamento.

        Tre esiti, non due: `True` ha scritto un
        file, `None` non c'era niente da salvare (nessun rettangolo
        bloccato -- il caso normale di `_cmd_salta`/di un frame senza
        volto), `False` ha TENTATO e il servizio ha rifiutato (riavviato a
        meta', disco pieno, permessi). In precedenza i due fallimenti
        tornavano entrambi un `False`/`None` indistinguibile e i chiamanti
        sotto sfogliavano comunque -- un salvataggio perso spariva senza
        una riga, proprio nella finestra che il sorvegliante rende
        ordinaria (cinque minuti di pausa, non solo un crash)."""
        if (self.servizio is None or self._frame_corrente is None
                or self._rect_corrente is None or self._landmarks_correnti is None):
            return None
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
        # I fratelli gia' noti, per la voce di rapporto che il servizio
        # scrive insieme al file (mainscripts/ExtractManual.py::_op_salva):
        # elenca i volti gia' allineati di QUESTO fotogramma, mai il file
        # che sta per nascere -- `self._mappa` e' la foto scattata
        # all'apertura della cartella, quindi non puo' contenere un file
        # che a quel momento non esisteva ancora. Il filtro `exists()` copre
        # la direzione opposta: un fratello cancellato DOPO la foto (a mano,
        # da un sort) libera il suo indice, e `_prossimo_face_idx_libero`
        # -- che guarda il disco vero, non la mappa -- lo riassegnerebbe al
        # file che sta per nascere. Mandarlo lo stesso duplicherebbe la voce
        # nel rapporto: lo stesso nome una volta da `fratelli` (stantio) e
        # una da `scritto` (mainscripts/ExtractManual.py::_op_salva).
        fratelli = [str(q) for q in (self._mappa.get(self._frame_corrente.name) or [])
                   if q.exists()]
        nome_file = self.servizio.salva(
            path=str(self._frame_corrente),
            output_dir=str(self._cartella / "aligned"),
            face_idx=face_idx,
            rect=self._rect_corrente,
            landmarks=self._landmarks_correnti,
            face_type=_FACE_TYPE_LUNGO.get(valori.get("face-type"), "whole_face"),
            image_size=valori.get("image-size", 512),
            jpeg_quality=valori.get("jpeg-quality", 90),
            report_dir=str(self._cache_dir()),
            fratelli=fratelli)
        if nome_file is None:
            # _invia valorizza
            # ultimo_errore in QUASI ogni fallimento, ma non e' garantito --
            # una risposta senza il campo "file" con _invia comunque
            # RIUSCITA (ultimo_errore rimesso a None) e' un "None" da qui
            # senza motivo. Oggi _op_salva scrive sempre "file", quindi
            # irraggiungibile -- ma "Save failed: None" e' un testo peggiore
            # di uno generico, ed e' un ripiego a costo zero.
            motivo = self.servizio.ultimo_errore or testi.ESTRAZIONE_MOTIVO_SALVATAGGIO_IGNOTO
            self._nota_salvataggio = testi.estrazione_salvataggio_fallito(motivo)
            return False
        self._salvati_per_frame[self._frame_corrente] = face_idx + 1
        # Non si scrive piu' qui direttamente --
        # lo sfogliamento che segue SEMPRE un salvataggio (_vai_a) la
        # cancellerebbe prima che l'utente la legga. _aggiorna_stato_manuale
        # la consuma al prossimo ridisegno.
        self._nota_salvataggio = testi.estrazione_volto_salvato(nome_file)
        return True

    def _su_confermato(self):
        # Un salvataggio TENTATO e fallito
        # (`False`, non `None`) non deve avanzare -- avanzare vorrebbe dire
        # perdere il volto in silenzio, esattamente il buco che il
        # sorvegliante ha reso raggiungibile da una pausa caffe' invece che
        # solo da un crash.
        if self._salva_corrente() is False:
            self._aggiorna_stato_manuale()
            return
        self._vai_a(+1)

    def _cmd_conferma(self):
        self._su_confermato()

    def _cmd_salta(self):
        self._vai_a(+1)

    def _cmd_successivo(self):
        if self.tela.bloccato() and self._salva_corrente() is False:
            self._aggiorna_stato_manuale()
            return
        self._vai_a(+1)

    def _cmd_precedente(self):
        if self.tela.bloccato() and self._salva_corrente() is False:
            self._aggiorna_stato_manuale()
            return
        self._vai_a(-1)

    def _cmd_salta_restanti(self):
        # `q` in Extractor.py (righe 521-526)
        # salva il volto bloccato prima di uscire -- il brief l'aveva
        # ridefinito come "esce e basta", ma per chi ha le dita sui tasti
        # di cv2 quello e' una perdita di dato silenziosa: un rettangolo
        # gia' agganciato e bloccato, buttato via invece di scritto.
        #
        # N3 del round 2: la nota di salvataggio va mostrata PRIMA di
        # ferma_servizio(), che azzera self._nota_salvataggio insieme al
        # resto dello stato manuale -- senza questa riga il file finisce
        # comunque su disco (il dato non si perde), ma l'utente chiude la
        # sessione senza mai vedere la conferma dell'ultimo volto.
        #
        # `is not None`, non un semplice if: qui
        # si esce comunque, quindi un salvataggio FALLITO non ha una
        # richiesta successiva che possa mostrarne l'errore -- e' questa
        # riga o mai piu'. `_salva_corrente` ha gia' composto la nota
        # giusta in entrambi i casi (successo o fallimento).
        if self.tela.bloccato() and self._salva_corrente() is not None:
            self._aggiorna_stato_manuale()
        self.ferma_servizio()

    def _cmd_rileva(self):
        self._rileva(None)

    def _cmd_blocca(self):
        self.tela.blocca(not self.tela.bloccato())

    def _cmd_muovi_sinistra(self):
        self.tela.muovi(*_DELTA_FRECCE["muovi-sinistra"])

    def _cmd_muovi_destra(self):
        self.tela.muovi(*_DELTA_FRECCE["muovi-destra"])

    def _cmd_muovi_su(self):
        self.tela.muovi(*_DELTA_FRECCE["muovi-su"])

    def _cmd_muovi_giu(self):
        self.tela.muovi(*_DELTA_FRECCE["muovi-giu"])

    def _cmd_ingrandisci(self):
        self.tela.ridimensiona(self.tela.passo_ridimensiona())

    def _cmd_rimpicciolisci(self):
        self.tela.ridimensiona(-self.tela.passo_ridimensiona())

    def _cmd_accuratezza(self):
        self._accurato = not self._accurato
        self._rileva(self.tela.rettangolo())

    def _cmd_vettore(self):
        self.tela.imposta_modo_vettore(not self.tela.modo_vettore())

    # Una tabella di dispatch invece di un
    # if/elif -- costruita una volta, a definizione di classe, non ad ogni
    # chiamata. Prima la lista delle chiavi coperte (CHIAVI_INSTRADATE,
    # sotto la classe) era scritta a mano, indipendente da questa catena:
    # un ramo dimenticato nell'if/elif lasciava comunque la sua chiave nella
    # lista, e il test di copertura restava verde (misurato: dieci mutazioni
    # diverse, tutte "1902 passed"). Derivando CHIAVI_INSTRADATE dalle
    # chiavi di QUESTA tabella, una voce mancante qui e' una chiave mancante
    # li' -- il test di copertura diventa vero per costruzione, non piu'
    # decorativo.
    _DISPATCH = {
        "conferma": _cmd_conferma,
        "salta": _cmd_salta,
        "successivo": _cmd_successivo,
        "precedente": _cmd_precedente,
        "salta-restanti": _cmd_salta_restanti,
        "rileva": _cmd_rileva,
        "blocca": _cmd_blocca,
        "muovi-sinistra": _cmd_muovi_sinistra,
        "muovi-destra": _cmd_muovi_destra,
        "muovi-su": _cmd_muovi_su,
        "muovi-giu": _cmd_muovi_giu,
        "ingrandisci": _cmd_ingrandisci,
        "rimpicciolisci": _cmd_rimpicciolisci,
        "accuratezza": _cmd_accuratezza,
        "vettore": _cmd_vettore,
    }

    def _su_comando(self, chiave):
        """Instrada tramite `_DISPATCH`. Fuori dalla sessione manuale esce
        subito -- la colonna e' gia' spenta, ma
        un comando instradato (una scorciatoia da tastiera) non deve
        dipendere solo da un'abilitazione: `blocca`, per dire, non legge
        `self.servizio` da sola."""
        if self.servizio is None or chiave not in CHIAVI_INSTRADATE:
            return
        self._DISPATCH[chiave](self)
        self._rigenera_comandi()

    # -- cosa e' permesso ora ------------------------------------------------

    def _rigenera_comandi(self):
        libera = self._job_corrente is None
        pronto = self._progetto is not None
        manuale_attiva = self.servizio is not None
        # `pronto and libera`, non solo `pronto`: cambiare lato sotto un job
        # lascerebbe le barre di quel job su una pagina che non e' la sua, e
        # farebbe leggere al battito del rapporto (`_lettore_vivo`, ricreato
        # da `apri()` sulla cartella NUOVA mentre `_timer_rapporto` resta
        # acceso) la cartella sbagliata per tutta la durata del job.
        # `not manuale_attiva` (M4 della revisione finale): ogni ALTRO
        # comando qui sotto e' gia' gated cosi', ma questi due bottoni
        # restavano accesi durante la sessione manuale -- premerli chiama
        # `apri()` -> `ferma_servizio()` e distrugge un rettangolo gia'
        # agganciato e bloccato SENZA chiedere. Niente si perde su disco,
        # ma l'incoerenza fra i due gruppi di comandi era gratuita.
        self.bottone_src.setEnabled(pronto and libera and not manuale_attiva)
        self.bottone_dst.setEnabled(pronto and libera and not manuale_attiva)
        self.selettore_operazione.setEnabled(pronto and self.selettore_operazione.count() > 0)
        self.bottone_avvia.setEnabled(pronto and libera and not manuale_attiva
                                      and self._gestore is not None
                                      and self.selettore_operazione.count() > 0)
        self.bottone_manuale.setEnabled(pronto and libera)
        self.bottone_manuale.setText(
            testi.ESTRAZIONE_MANUALE_ESCI if manuale_attiva else testi.ESTRAZIONE_MANUALE)
        # I tre controlli dei motori riguardano SOLO la sessione manuale:
        # fuori non hanno niente su cui agire (i passi automatici hanno i
        # propri selettori nel form del dialogo), e lasciarli visibili
        # allargherebbe la barra per niente.
        self.riga_motori.setVisible(manuale_attiva)
        for controllo in self._controlli_motori:
            controllo.setEnabled(manuale_attiva)
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
        # Non `libera`: guardare i volti di un frame e' di sola lettura, e
        # funziona anche mentre un job gira.
        self.bottone_volti.setEnabled(self._frame_corrente is not None)
        # Fuori dalla sessione manuale la colonna resta visibile ma spenta:
        # si impara che i comandi esistono anche senza entrarci. Dentro,
        # tre famiglie dipendono da uno stato che non c'e' ancora appena
        # entrati (nessun frame caricato): le frecce e il ridimensionamento
        # vogliono un rettangolo BLOCCATO (l'affinamento ha senso solo su
        # "questo e' il volto"), "conferma" vuole i landmark VERI
        # dell'allineatore -- mai i due assi confusi, o le frecce
        # resterebbero accese su un rettangolo che nessuno ha ancora
        # scelto.
        bloccato = manuale_attiva and self.tela.bloccato()
        ha_rect = manuale_attiva and self.tela.rettangolo() is not None
        acceso = dict((c.chiave, manuale_attiva) for c in COMANDI)
        # Tela.blocca() non richiede un
        # rettangolo -- premere L senza averne uno prima accenderebbe le
        # frecce su un rettangolo che non c'e', che Tela.muovi scarterebbe
        # comunque all'ingresso. "bloccato" da solo non basta.
        for chiave in CHIAVI_FRECCE:
            acceso[chiave] = bloccato and ha_rect
        acceso["ingrandisci"] = acceso["rimpicciolisci"] = ha_rect
        acceso["conferma"] = manuale_attiva and self._landmarks_correnti is not None
        self.colonna.imposta_abilitati(acceso)
        self.colonna.imposta_spunta("accuratezza", self._accurato)
        self.colonna.imposta_spunta("vettore", manuale_attiva and self.tela.modo_vettore())
        self.colonna.imposta_spunta("blocca", bloccato)


# Derivata dalle chiavi di PaginaEstrazione._DISPATCH, non scritta a mano:
# vedi il commento sopra _DISPATCH per il perche'. Usata sia come guardia
# d'ingresso di _su_comando sia dal test
# di copertura (test_ogni_chiave_dei_comandi_e_instradata) -- la stessa
# lista in entrambi i posti, per costruzione.
CHIAVI_INSTRADATE = frozenset(PaginaEstrazione._DISPATCH)
