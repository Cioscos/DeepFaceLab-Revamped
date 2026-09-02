"""L'export del video dalla pagina di fusione: nessun comando nuovo, i
quattro passi «8) merged to …» del catalogo lanciati con JobManager (cosi'
il job sta nel pannello dei job e conflicts.py lo vede), e l'esito letto
dal file a job finito -- dimensione sempre, durata e fps da ffprobe se
c'e'. Le coppie (contenitore, lossless) offerte sono esattamente quelle
del catalogo: mov non lossless e avi lossless non esistono.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QFormLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox,
                             QWidget)

from gui import numeri, testi, theme
from gui.catalog import video_output

COPPIE = {("mp4", False): "8) merged to mp4",
          ("mp4", True): "8) merged to mp4 lossless",
          ("mov", True): "8) merged to mov lossless",
          ("avi", False): "8) merged to avi"}
CONTENITORI = ("mp4", "mov", "avi")
CHIAVE_BITRATE = "bitrate-of-output-file-in-mbs"
_TIMEOUT_FFPROBE_S = 20


def passo_per(contenitore, lossless):
    """Il passo del catalogo per quella coppia, o None se non esiste."""
    nome = COPPIE.get((contenitore, bool(lossless)))
    for passo in video_output.STEPS:
        if passo.name == nome:
            return passo
    return None


def risposte_per(passo, bitrate):
    """Il bitrate va nelle risposte solo se il passo lo chiede: i due
    passi lossless non hanno quel campo."""
    if any(campo.key == CHIAVE_BITRATE for campo in passo.fields):
        return {CHIAVE_BITRATE: int(bitrate)}
    return {}


def _default_bitrate():
    for campo in video_output.STEPS[-1].fields:
        if campo.key == CHIAVE_BITRATE and campo.default is not None:
            return int(campo.default)
    return 16


class DialogoExport(QDialog):
    """Contenitore, spunta lossless (vincolata al catalogo) e bitrate: i
    soli dati che il passo scelto chiede.

    Il riferimento audio si mostra ma non si modifica: il catalogo cabla
    `--reference-file` su `{WORKSPACE}/data_dst.*` e non ha quel campo,
    quindi un valore diverso digitato qui verrebbe ignorato in silenzio."""

    def __init__(self, riferimento, parent=None):
        super().__init__(parent)
        self.setWindowTitle(testi.FUSIONE_EXPORT_TITLE)
        form = QFormLayout(self)
        self.tendina_contenitore = theme.tendina()
        self.tendina_contenitore.addItems(list(CONTENITORI))
        form.addRow(testi.FUSIONE_EXPORT_CONTAINER, self.tendina_contenitore)
        self.spunta_lossless = QCheckBox(testi.FUSIONE_EXPORT_LOSSLESS)
        form.addRow("", self.spunta_lossless)
        self.campo_bitrate = QSpinBox()
        self.campo_bitrate.setRange(1, 200)
        self.campo_bitrate.setValue(_default_bitrate())
        form.addRow(testi.FUSIONE_EXPORT_BITRATE, self.campo_bitrate)
        self.campo_riferimento = QLineEdit(str(riferimento))
        self.campo_riferimento.setReadOnly(True)      # informativo: vedi il docstring
        form.addRow(testi.FUSIONE_EXPORT_REFERENCE, self.campo_riferimento)
        bottoni = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bottoni.accepted.connect(self.accept)
        bottoni.rejected.connect(self.reject)
        form.addRow(bottoni)
        self.tendina_contenitore.currentTextChanged.connect(lambda _t: self._vincola())
        self.spunta_lossless.toggled.connect(lambda _v: self._vincola())
        self._vincola()

    def _vincola(self):
        """mov esiste solo lossless, avi solo non lossless: la spunta segue
        il catalogo e si blocca; mp4 la lascia libera."""
        contenitore = self.tendina_contenitore.currentText()
        libera = (contenitore, True) in COPPIE and (contenitore, False) in COPPIE
        if not libera:
            self.spunta_lossless.setChecked((contenitore, True) in COPPIE)
        self.spunta_lossless.setEnabled(libera)
        self.campo_bitrate.setEnabled(not self.spunta_lossless.isChecked())

    def scelta(self):
        return (self.tendina_contenitore.currentText(), self.spunta_lossless.isChecked(),
                self.campo_bitrate.value())


def trova_ffprobe():
    """Cerca l'eseguibile prima in FFMPEG_PATH, poi nel PATH di sistema."""
    cartella = os.environ.get("FFMPEG_PATH")
    if cartella:
        trovato = shutil.which("ffprobe", path=cartella)
        if trovato:
            return trovato
    return shutil.which("ffprobe")


def _fps_da(razionale):
    try:
        num, den = str(razionale).split("/")
        den = float(den)
        return float(num) / den if den else None
    except (ValueError, ZeroDivisionError):
        return None


def esito_del_file(percorso, ffprobe=None):
    """None se il file non c'e'. Altrimenti dimensione sempre; durata e fps
    da ffprobe se lo si trova e risponde JSON; se no restano None.

    Non solleva mai: ffprobe e' un processo esterno che puo' mancare,
    stampare spazzatura o non finire, e il codice d'uscita del job che
    chiama questa funzione viene da un altro processo ancora."""
    percorso = Path(percorso)
    if not percorso.is_file():
        return None
    esito = {"dimensione": percorso.stat().st_size, "durata_s": None, "fps": None,
             "ffprobe_trovato": False}
    eseguibile = ffprobe if ffprobe is not None else trova_ffprobe()
    if not eseguibile or not Path(eseguibile).is_file():
        return esito
    esito["ffprobe_trovato"] = True
    try:
        uscita = subprocess.run([eseguibile, "-v", "quiet", "-print_format", "json",
                                 "-show_format", "-show_streams", str(percorso)],
                                capture_output=True, text=True, timeout=_TIMEOUT_FFPROBE_S)
        dati = json.loads(uscita.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return esito
    try:
        durata = float(dati.get("format", {}).get("duration"))
        esito["durata_s"] = durata if numeri.numero_finito(durata) else None
    except (TypeError, ValueError, AttributeError):
        pass
    for flusso in dati.get("streams") or []:
        if isinstance(flusso, dict) and flusso.get("codec_type") == "video":
            fps = _fps_da(flusso.get("r_frame_rate"))
            esito["fps"] = fps if fps is not None and numeri.numero_finito(fps) else None
            break
    return esito


class PannelloEsito(QWidget):
    """Dimensione, durata e fps del file esportato, con un bottone che lo
    apre con l'applicazione di sistema."""

    def __init__(self, parent=None):
        super().__init__(parent)
        riga = QHBoxLayout(self)
        riga.setContentsMargins(0, 0, 0, 0)
        self.etichetta = QLabel("")
        riga.addWidget(self.etichetta, 1)
        self.bottone_apri = QPushButton(testi.FUSIONE_EXPORT_OPEN)
        self.bottone_apri.clicked.connect(self._apri)
        riga.addWidget(self.bottone_apri)
        self._percorso = None
        self.imposta(None, None)

    def imposta(self, esito, percorso):
        self._percorso = Path(percorso) if percorso is not None else None
        if not isinstance(esito, dict):
            self.etichetta.setText("")
            self.bottone_apri.setEnabled(False)
            return
        dimensione = esito.get("dimensione")
        mb = dimensione / (1024.0 * 1024.0) if numeri.numero_finito(dimensione) else None
        durata = esito.get("durata_s") if numeri.numero_finito(esito.get("durata_s")) else None
        fps = esito.get("fps") if numeri.numero_finito(esito.get("fps")) else None
        testo = testi.fusione_esito_export(mb, durata, fps)
        if not esito.get("ffprobe_trovato"):
            testo = "%s %s" % (testo, testi.FUSIONE_EXPORT_NO_FFPROBE)
        self.etichetta.setText(testo)
        self.bottone_apri.setEnabled(self._percorso is not None and self._percorso.is_file())

    def _apri(self):
        if self._percorso is not None and self._percorso.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._percorso)))
