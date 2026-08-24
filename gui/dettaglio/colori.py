"""I colori dei sette gruppi, ricordati fra una sessione e l'altra.

In QSettings e non in project.json: un colore e' una preferenza della
PERSONA, non un dato del progetto, e chi cambia progetto non vuole rifare
la tavolozza.

Stessa forma di gui/preferenze.py::ScalaTesto: un oggetto con
value/setValue/remove iniettabile, e QSettings importato pigramente dentro
il costruttore, cosi' importare questo modulo non richiede che Qt sia
stato inizializzato.
"""
from PyQt5.QtGui import QColor

from gui.dettaglio.gruppi import COLORI_PREDEFINITI, NOMI

_PREFISSO = "dettaglio/colore/"


class ColoriGruppi:
    def __init__(self, settings=None):
        if settings is None:
            from PyQt5.QtCore import QSettings
            settings = QSettings("DeepFaceLab", "gui")
        self._settings = settings

    def _chiave(self, nome):
        # Un nome sbagliato e' un errore di chi chiama, non un dato
        # dubbio che arriva da fuori: si sente subito. Il valore SUL
        # DISCO invece si ripiega in silenzio -- e' l'asimmetria di
        # ScalaTesto, e vale identica qui.
        if nome not in NOMI:
            raise ValueError("gruppo sconosciuto: %r" % (nome,))
        return _PREFISSO + nome

    def colore(self, nome):
        """Il colore del gruppo, o il predefinito se sul disco non c'e'
        niente di leggibile. Non solleva per il contenuto del disco."""
        grezzo = self._settings.value(self._chiave(nome), None)
        if isinstance(grezzo, str):
            candidato = QColor(grezzo)
            if candidato.isValid():
                return candidato
        return QColor(*COLORI_PREDEFINITI[nome])

    def imposta(self, nome, colore):
        chiave = self._chiave(nome)
        if not isinstance(colore, QColor) or not colore.isValid():
            raise ValueError("colore non valido per %r" % (nome,))
        # `#rrggbb` e non l'oggetto QColor: quello che QSettings scrive per
        # un QColor e' illeggibile in un file di configurazione e cambia
        # forma fra le piattaforme. `QColor.name()` scarta pero' il canale
        # alpha: un colore semitrasparente rilegge sempre opaco dopo un
        # giro di scrittura.
        self._settings.setValue(chiave, colore.name())

    def azzera(self, nome):
        self._settings.remove(self._chiave(nome))

    def tutti(self):
        return dict((nome, self.colore(nome)) for nome in NOMI)
