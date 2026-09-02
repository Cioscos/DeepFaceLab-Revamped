"""I preset di merge: la cfg corrente con un nome, in project.json sotto
memory["merge_presets"], via ricorda_risposte (che FONDE: si riscrive
l'intero dizionario dei preset letto prima, o `extra` sovrascriverebbe la
chiave con il solo preset nuovo). Un preset porta i quindici campi del
nucleo e non face_type; alla lettura passa da imposta_cfg del pannello,
che valida e clippa: un valore fuori limite arriva clippato, una chiave
sconosciuta viene ignorata."""
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QHBoxLayout, QInputDialog, QLabel, QMessageBox,
                             QPushButton, QWidget)

from gui import progetti, testi, theme
from gui.fusione.pannello_parametri import CHIAVE_PER_CAMPO

CHIAVE_MEMORIA = "merge_presets"
CAMPI = frozenset(CHIAVE_PER_CAMPO.values())


def leggi(cartella):
    """I preset del progetto in `cartella`, o `{}` se non c'e' un progetto
    o la chiave e' scritta male (una stringa, una lista...)."""
    progetto = progetti.leggi_progetto(cartella)
    if progetto is None:
        return {}
    presets = progetto.memoria.get(CHIAVE_MEMORIA)
    if not isinstance(presets, dict):
        return {}
    return {nome: dict(cfg) for nome, cfg in presets.items()
            if isinstance(nome, str) and isinstance(cfg, dict)}


def salva(cartella, nome, cfg):
    """Salva `cfg` (solo i quindici campi del nucleo) sotto `nome`,
    fondendo con i preset gia' presenti. `False` senza un progetto."""
    if progetti.leggi_progetto(cartella) is None:
        return False
    presets = leggi(cartella)
    presets[str(nome)] = {k: v for k, v in dict(cfg).items() if k in CAMPI}
    return progetti.ricorda_risposte(cartella, "merge preset", {}, extra={CHIAVE_MEMORIA: presets})


def cancella(cartella, nome):
    """Toglie il preset `nome`. `False` se non c'era."""
    presets = leggi(cartella)
    if nome not in presets:
        return False
    del presets[nome]
    return progetti.ricorda_risposte(cartella, "merge preset", {}, extra={CHIAVE_MEMORIA: presets})


class BarraPreset(QWidget):
    """La tendina dei preset del progetto aperto, con salva e cancella.

    `chiedi_nome` e `conferma` sono iniettabili: i test non aprono nessun
    dialogo vero."""
    preset_scelto = pyqtSignal(dict)
    salvato = pyqtSignal(str)
    cancellato = pyqtSignal(str)

    def __init__(self, cfg_corrente=None, parent=None):
        super().__init__(parent)
        self._cartella = None
        self._presets = {}
        self._cfg_corrente = cfg_corrente or (lambda: {})
        self.chiedi_nome = self._chiedi_nome_con_dialogo
        self.conferma = self._conferma_con_dialogo
        riga = QHBoxLayout(self)
        riga.setContentsMargins(0, 0, 0, 0)
        riga.addWidget(QLabel(testi.FUSIONE_PRESET))
        self.tendina = theme.tendina()
        self.tendina.currentTextChanged.connect(self._su_scelta)
        riga.addWidget(self.tendina, 1)
        self.bottone_salva = QPushButton(testi.FUSIONE_PRESET_SAVE)
        self.bottone_salva.clicked.connect(self._salva)
        riga.addWidget(self.bottone_salva)
        self.bottone_cancella = QPushButton(testi.FUSIONE_PRESET_DELETE)
        self.bottone_cancella.clicked.connect(self._cancella)
        riga.addWidget(self.bottone_cancella)
        self._ricostruisci()

    def imposta(self, cartella):
        self._cartella = cartella
        self.ricarica()

    def ricarica(self):
        self._presets = leggi(self._cartella) if self._cartella is not None else {}
        self._ricostruisci()

    def nomi(self):
        return sorted(self._presets)

    def _ricostruisci(self):
        self.tendina.blockSignals(True)
        try:
            self.tendina.clear()
            self.tendina.addItem(testi.FUSIONE_PRESET_NONE)
            for nome in self.nomi():
                self.tendina.addItem(nome)
        finally:
            self.tendina.blockSignals(False)
        self.bottone_cancella.setEnabled(False)

    def _su_scelta(self, nome):
        cfg = self._presets.get(nome)
        self.bottone_cancella.setEnabled(cfg is not None)
        if cfg is not None:
            self.preset_scelto.emit(dict(cfg))

    def _chiedi_nome_con_dialogo(self):
        return QInputDialog.getText(self, testi.FUSIONE_PRESET_SAVE, testi.FUSIONE_PRESET_NAME)

    def _conferma_con_dialogo(self, testo):
        return QMessageBox.question(self, testi.FUSIONE_PRESET_SAVE, testo) == QMessageBox.Yes

    def _salva(self):
        if self._cartella is None:
            return
        nome, ok = self.chiedi_nome()
        nome = str(nome).strip()
        if not ok or not nome:
            return
        if nome in self._presets and not self.conferma(testi.FUSIONE_PRESET_OVERWRITE % nome):
            return
        if not salva(self._cartella, nome, self._cfg_corrente()):
            return
        self.ricarica()
        self.tendina.blockSignals(True)
        self.tendina.setCurrentText(nome)
        self.tendina.blockSignals(False)
        self.bottone_cancella.setEnabled(True)
        self.salvato.emit(nome)

    def _cancella(self):
        nome = self.tendina.currentText()
        if self._cartella is None or nome not in self._presets:
            return
        if cancella(self._cartella, nome):
            self.ricarica()
            self.cancellato.emit(nome)
