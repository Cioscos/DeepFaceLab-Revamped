"""Le preferenze d'aspetto, ricordate fra una sessione e l'altra.

Stesso backend e stessa forma di `RecentWorkspaces`: un oggetto con
`value`/`setValue` iniettabile, e `QSettings` importato pigramente dentro il
costruttore, cosi' importare questo modulo non richiede che Qt sia stato
inizializzato.
"""
from gui.theme import SCALE_FACTORS, SCALE_NAMES

_CHIAVE = "textScale"


class ScalaTesto:
    """Il nome della scala scelta, e il fattore che ne consegue.

    Un valore illeggibile sul disco vale come "normale": la finestra deve
    aprirsi comunque. Un nome sbagliato passato da codice e' invece un
    errore di chi chiama, e si sente subito.
    """

    def __init__(self, settings=None):
        if settings is None:
            from PyQt5.QtCore import QSettings
            settings = QSettings("DeepFaceLab", "gui")
        self._settings = settings

    def nome(self):
        valore = self._settings.value(_CHIAVE, SCALE_NAMES[0])
        return valore if valore in SCALE_NAMES else SCALE_NAMES[0]

    def fattore(self):
        return SCALE_FACTORS[self.nome()]

    def imposta(self, nome):
        if nome not in SCALE_NAMES:
            raise ValueError("scala sconosciuta: %r" % (nome,))
        self._settings.setValue(_CHIAVE, nome)
