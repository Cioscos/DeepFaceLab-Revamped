"""La copia di un progetto, su un thread suo, con avanzamento annullabile.

Non gira sul thread dell'interfaccia perche' non e' un'operazione breve: un
model/ sono giga e un data_dst/ decine. Bloccare l'interfaccia per minuti
congelerebbe anche il pannello di un training in corso su un altro
progetto -- che e' esattamente cio' che questo ciclo rende possibile avere.

Il corpo di CopiaProgetto.run e' controllato da un test che ne cammina
l'AST (in tests_gui/test_duplicazione.py): riconosce la costruzione di un
nome a forma di classe Qt ("Q" maiuscola seguita da un'altra maiuscola) e
le chiamate di metodo su una variabile che ne ha ricevuta una -- non ogni
tocco possibile dell'interfaccia. Un alias di importazione o una chiamata
fatta con getattr sfuggono entrambi, dichiarato (non scoperto da soli) nel
docstring di quel test.
"""
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QInputDialog, QLabel, QMessageBox,
    QProgressDialog, QVBoxLayout,
)

from gui import testi
from gui.progetti import DuplicazioneIncompleta


class _SceltaCosa(QDialog):
    """Le spunte di cosa portare nella copia -- modello, dataset, video.

    OK resta disabilitato finche' non c'e' almeno una spunta: senza,
    premerlo a vuoto sarebbe indistinguibile da un annullamento (nessuna
    copia parte, nessuno spiega perche') -- lo stato del bottone lo dice
    da solo, senza bisogno di un messaggio in piu'.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(testi.TITLE_DUPLICATE_PROJECT)
        colonna = QVBoxLayout(self)
        colonna.addWidget(QLabel(testi.DUPLICATE_WHAT))
        self._modello = QCheckBox(testi.DUPLICATE_MODEL)
        self._dataset = QCheckBox(testi.DUPLICATE_DATASET)
        self._video = QCheckBox(testi.DUPLICATE_VIDEO)
        for casella in (self._modello, self._dataset, self._video):
            colonna.addWidget(casella)
        bottoni = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bottoni.accepted.connect(self.accept)
        bottoni.rejected.connect(self.reject)
        colonna.addWidget(bottoni)
        self._bottone_ok = bottoni.button(QDialogButtonBox.Ok)
        self._bottone_ok.setEnabled(False)
        for casella in (self._modello, self._dataset, self._video):
            casella.stateChanged.connect(self._aggiorna_ok)

    def _aggiorna_ok(self, _stato=None):
        self._bottone_ok.setEnabled(bool(self.cosa_scelta()))

    def cosa_scelta(self):
        cosa = set()
        if self._modello.isChecked():
            cosa.add("modello")
        if self._dataset.isChecked():
            cosa.add("dataset")
        if self._video.isChecked():
            cosa.add("video")
        return cosa


def dialogo_di_avanzamento(lavoratore, parent):
    """Un QProgressDialog legato al lavoratore. Vive sul thread dell'interfaccia."""
    dialogo = QProgressDialog(testi.COPYING_PROJECT, testi.CANCEL, 0, 1, parent)
    dialogo.setWindowTitle(testi.TITLE_DUPLICATE_PROJECT)
    dialogo.setAutoClose(True)
    dialogo.canceled.connect(lavoratore.annulla)

    def _avanzato(fatti, totali):
        dialogo.setMaximum(max(totali, 1))
        dialogo.setValue(fatti)

    lavoratore.avanzato.connect(_avanzato)
    lavoratore.finito.connect(lambda _p: dialogo.close())
    return dialogo


class DialogoDuplicazione:
    """Il percorso intero di Duplicate...: il nome, cosa copiare, e il
    dialogo di avanzamento annullabile attorno a CopiaProgetto.

    Non e' un QWidget: orchestra tre dialoghi in sequenza e torna il
    risultato a chi l'ha chiamato, che decide cosa farne -- qui, passare al
    progetto nuovo.
    """

    def __init__(self, archivio, progetto, parent=None):
        self._archivio = archivio
        self._progetto = progetto
        self._parent = parent

    def esegui(self):
        """None a ogni annullamento -- del nome, delle spunte, o della
        copia stessa -- il Progetto nuovo solo a copia riuscita."""
        nome, ok = QInputDialog.getText(
            self._parent, testi.TITLE_DUPLICATE_PROJECT, testi.DIALOG_NEW_PROJECT_NAME_LABEL)
        if not ok or not nome.strip():
            return None
        cosa = self._chiedi_cosa()
        if not cosa:
            return None
        return self._copia(nome.strip(), cosa)

    def _chiedi_cosa(self):
        dialogo = _SceltaCosa(self._parent)
        if dialogo.exec_() != QDialog.Accepted:
            return None
        return dialogo.cosa_scelta() or None

    def _copia(self, nome, cosa):
        """Avvia il lavoratore, mostra il progresso, aspetta -- e avvisa se
        l'annullamento e' riuscito ma ha lasciato spazzatura sul disco
        (DuplicazioneIncompleta, vedi CopiaProgetto.run) o se la copia stessa
        e' fallita per un errore del disco (CopiaProgetto.errore, C2): senza
        uno di questi due avvisi l'utente non saprebbe ne' di dover ripulire
        a mano ne' che la copia non e' mai arrivata in fondo.

        Il risultato si legge dagli attributi del lavoratore DOPO wait(),
        non da un secondo ascoltatore agganciato a `finito` insieme a
        quello che chiude il dialogo dentro dialogo_di_avanzamento: due
        ascoltatori sullo stesso segnale girano nell'ordine in cui si sono
        agganciati, e se quello che chiude il dialogo parte per primo puo'
        far uscire il ciclo annidato di exec_() prima che l'altro sia mai
        stato consegnato -- misurato, non supposto, chiudere il dialogo
        interrompe il resto della consegna in corso. wait() non ha invece
        questa corsa: torna solo quando il lavoratore ha gia' finito di
        scrivere i propri attributi.
        """
        lavoratore = CopiaProgetto(self._archivio, self._progetto, nome, cosa, parent=self._parent)
        progresso = dialogo_di_avanzamento(lavoratore, self._parent)
        lavoratore.start()
        progresso.exec_()
        lavoratore.wait()
        if lavoratore.errore is not None:
            QMessageBox.warning(
                self._parent, testi.TITLE_PROJECT_ACTION_FAILED,
                testi.msg_project_action_failed(str(lavoratore.errore)))
        elif lavoratore.destinazione_incompleta is not None:
            QMessageBox.warning(
                self._parent, testi.TITLE_DUPLICATE_INCOMPLETE,
                testi.msg_duplicate_incomplete(lavoratore.destinazione_incompleta))
        return lavoratore.risultato


class CopiaProgetto(QThread):
    """Copia in un thread di lavoro. `avanzato` e `incompleta` sono l'unico
    ponte verso l'interfaccia: `run` non tocca nessun widget, emette e
    basta -- un widget toccato da qui porterebbe via il processo, e con
    esso ogni training aperto nella stessa finestra.
    """
    avanzato = pyqtSignal(int, int)     # fatti, totali
    finito = pyqtSignal(object)         # il Progetto nuovo, o None se annullata o incompleta
    incompleta = pyqtSignal(object)     # la cartella di destinazione rimasta a meta'

    def __init__(self, archivio, sorgente, nome, cosa, parent=None):
        super().__init__(parent)
        self._archivio = archivio
        self._sorgente = sorgente
        self._nome = nome
        self._cosa = set(cosa)
        self._annullata = False
        # Scritti da run() sul thread di lavoro, letti da chi chiama SOLO
        # dopo wait(): quel wait() e' la sincronizzazione, non serve altro
        # -- vedi DialogoDuplicazione._copia sul perche' non ci si affida
        # invece a un secondo ascoltatore di `finito`.
        self.risultato = None
        self.destinazione_incompleta = None
        # Un disco pieno, un permesso negato, un handle aperto su Windows --
        # qualunque OSError che duplica() puo' sollevare al di fuori del
        # terzo esito gia' nominato sopra (DuplicazioneIncompleta). Vedi il
        # secondo except in run().
        self.errore = None

    def annulla(self):
        """Chiamabile dal thread dell'interfaccia mentre la copia va avanti.

        Un bool scritto da un thread e letto dall'altro: nessun lucchetto,
        perche' la sola transizione possibile e' False -> True e leggerlo un
        file piu' tardi del dovuto costa un file copiato in piu', non una
        corsa sui dati.
        """
        self._annullata = True

    def run(self):
        """Gira sul thread di lavoro: nessun nome di metodo dell'interfaccia
        qui dentro, solo attributi e i segnali sopra -- Qt consegna questi
        ultimi a coda al thread dell'interfaccia da solo.

        `DuplicazioneIncompleta` e' il terzo esito di
        `ArchivioProgetti.duplica`: l'annullamento e' riuscito ma la
        destinazione parziale non si e' potuta rimuovere. Non e' un guasto
        da lasciare risalire non gestito -- non porterebbe via il processo
        (questo thread non e' quello dell'interfaccia), ma lascerebbe
        `finito` senza emettitore: il dialogo di avanzamento resterebbe
        aperto in attesa di un segnale che non arriva piu', e chi ha
        chiamato non saprebbe che sul disco e' rimasta della spazzatura da
        controllare a mano.

        Il secondo `except`, aggiunto nella revisione finale del ciclo dei
        progetti multipli (C2), copre ogni altro esito di `duplica()`: un
        disco pieno, un permesso negato, un handle che un antivirus o
        Esplora risorse tengono aperto sulla cartella -- `create_workspace`,
        `Path.mkdir`, `shutil.copy2` e `scrivi_progetto` possono sollevare
        tutti da dentro `duplica()`, e senza questo `except` quell'eccezione
        risalirebbe non gestita fuori da `run()`. Qui non porterebbe via il
        processo (questo thread non e' quello dell'interfaccia -- stesso
        motivo del caso sopra), ma senza `finito` emesso comunque il
        dialogo di avanzamento resterebbe appeso per sempre: misurato
        eseguendo, non supposto, prima di questa correzione.
        """
        try:
            nuovo = self._archivio.duplica(
                self._sorgente, self._nome, self._cosa,
                avanzamento=lambda fatti, totali: self.avanzato.emit(fatti, totali),
                annullato=lambda: self._annullata)
        except DuplicazioneIncompleta as errore:
            self.destinazione_incompleta = errore.destinazione
            self.incompleta.emit(errore.destinazione)
            self.finito.emit(None)
            return
        except Exception as errore:
            self.errore = errore
            self.finito.emit(None)
            return
        self.risultato = nuovo
        self.finito.emit(nuovo)
