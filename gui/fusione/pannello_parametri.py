"""I quindici parametri del merger, resi dai FieldDef del catalogo.

Il catalogo (gui/catalog/merging.py) porta gia' range, scelte, aiuto per
voce e sezioni: qui non si riscrive niente, si costruisce ogni controllo
con gui.forms._build_control e si traduce fra le chiavi dei prompt e i
campi del nucleo (CHIAVE_PER_CAMPO). Ogni cambio parte con un debounce:
tenere premuta una freccia su uno spinbox non deve mandare un comando per
tick al pool.
"""
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import QFormLayout, QLabel, QVBoxLayout, QWidget

from gui import fascia_aiuto, forms
from gui.catalog.merging import _SETTINGS_BATTERY, _SEZIONI_MERGE_SAEHD
from gui.catalog.model import FIELD_BOOL, FIELD_CHOICE

DEBOUNCE_MS = 150

CHIAVE_PER_CAMPO = {
    "choose-mode": "mode",
    "masked-hist-match": "masked_hist_match",
    "hist-match-threshold": "hist_match_threshold",
    "choose-mask-mode": "mask_mode",
    "choose-erode-mask-modifier": "erode_mask_modifier",
    "choose-blur-mask-modifier": "blur_mask_modifier",
    "choose-motion-blur-power": "motion_blur_power",
    "choose-output-face-scale-modifier": "output_face_scale",
    "color-transfer-to-predicted-face": "color_transfer_mode",
    "choose-sharpen-mode": "sharpen_mode",
    "choose-blursharpen-amount": "blursharpen_amount",
    "choose-super-resolution-power": "super_resolution_power",
    "choose-image-degrade-by-denoise-power": "image_denoise_power",
    "choose-image-degrade-by-bicubic-rescale-power": "bicubic_degrade_power",
    "degrade-color-power-of-final-image": "color_degrade_power",
}

# I default della PRIMA sessione interattiva (il costruttore di
# MergerConfigMasked), non quelli di ask_settings che il catalogo porta:
# la pagina parte come partiva la finestra cv2.
_DEFAULT_INTERATTIVI = {"mode": "overlay", "masked_hist_match": True, "hist_match_threshold": 238,
                        "mask_mode": 4, "erode_mask_modifier": 0, "blur_mask_modifier": 0,
                        "motion_blur_power": 0, "output_face_scale": 0, "color_transfer_mode": 1,
                        "super_resolution_power": 0, "image_denoise_power": 0,
                        "bicubic_degrade_power": 0, "color_degrade_power": 0,
                        "sharpen_mode": 0, "blursharpen_amount": 0}

_SEZIONI_PANNELLO = tuple((titolo, chiavi) for titolo, chiavi in _SEZIONI_MERGE_SAEHD if titolo != "Output") + \
    (("Output", ("choose-mode", "choose-output-face-scale-modifier")),)


def _verso_nucleo(field, valore):
    """Il valore del widget -> il valore che il nucleo vuole."""
    if field.kind == FIELD_BOOL:
        return bool(valore)
    if field.kind == FIELD_CHOICE:
        # 'mode' resta la stringa scelta cosi' com'e' -- a differenza degli
        # altri campi a scelta, il nucleo non vuole l'indice qui.
        if field.key == "choose-mode":
            return valore
        if field.choice_values:
            return field.choice_values[field.choices.index(valore)] if valore in field.choices else field.choice_values[0]
        if field.key == "color-transfer-to-predicted-face":
            return 0 if valore is None else 1 + field.choices.index(valore)
        return valore
    return int(valore) if valore is not None else 0


def _verso_widget(field, valore):
    if field.kind == FIELD_CHOICE:
        if field.key == "choose-mode":
            return valore
        if field.choice_values:
            return field.choices[field.choice_values.index(valore)] if valore in field.choice_values else field.choices[0]
        if field.key == "color-transfer-to-predicted-face":
            return None if not valore else field.choices[int(valore) - 1]
    return valore


class PannelloParametri(QWidget):
    cfg_cambiata = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._campi = {f.key: f for f in _SETTINGS_BATTERY}
        self._controlli = {}        # campo del nucleo -> (widget, get, set_, field)
        self._silenzio = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(DEBOUNCE_MS)
        self._timer.timeout.connect(self._emetti)
        colonna = QVBoxLayout(self)
        for titolo, chiavi in _SEZIONI_PANNELLO:
            etichetta = QLabel(titolo)
            etichetta.setProperty("ruolo", "sezione")
            colonna.addWidget(etichetta)
            form = QFormLayout()
            for chiave in chiavi:
                field = self._campi[chiave]
                layout_widget, value_widget, get, set_, segnale = forms._build_control(field)
                if field.help:
                    value_widget.setToolTip(field.help)
                campo = CHIAVE_PER_CAMPO[chiave]
                self._controlli[campo] = (value_widget, get, set_, field)
                segnale.connect(self._su_cambio)
                form.addRow(QLabel(field.label), layout_widget)
            colonna.addLayout(form)
        colonna.addStretch(1)
        self.imposta_cfg(_DEFAULT_INTERATTIVI)

    def collega_fascia(self, fascia):
        for campo, (widget, _g, _s, field) in self._controlli.items():
            fascia_aiuto.osserva(widget, fascia, field.label, field.help or "",
                                 per_voce=tuple(field.choice_help or ()))

    def controllo(self, campo):
        return self._controlli[campo][0]

    def cfg(self):
        out = {}
        for campo, (_w, get, _s, field) in self._controlli.items():
            out[campo] = _verso_nucleo(field, get())
        return out

    def imposta_cfg(self, cfg):
        self._silenzio = True
        try:
            for campo, valore in cfg.items():
                if campo in self._controlli:
                    _w, _g, set_, field = self._controlli[campo]
                    set_(_verso_widget(field, valore))
        finally:
            self._silenzio = False
        self._timer.stop()

    def abilita(self, acceso):
        for widget, _g, _s, _f in self._controlli.values():
            widget.setEnabled(bool(acceso))

    def _su_cambio(self, *_args):
        if self._silenzio:
            return
        self._timer.start()

    def _emetti(self):
        cfg = self.cfg()
        self.cfg_cambiata.emit(cfg)
