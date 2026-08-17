"""Il modello dei frame: percorsi e voci, mai pixel.

Un frame comparso dopo l'ultima indicizzazione non ha voce, e resta
visibile lo stesso: sparire senza spiegazione e' peggio che comparire
senza esito.
"""
from PyQt5 import QtCore

from gui import testi
from gui.estrazione import indice

RUOLO_PERCORSO = QtCore.Qt.UserRole + 1
RUOLO_VOCE = QtCore.Qt.UserRole + 2

_PREDICATI = dict((chiave, predicato) for chiave, _, predicato in indice.FILTRI)


class ModelloFrame(QtCore.QAbstractListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._tutti = []
        self._visibili = []
        self._filtro = "tutti"

    def imposta(self, percorsi, voci):
        per_nome = dict((v.get("nome"), v) for v in voci
                        if isinstance(v, dict) and isinstance(v.get("nome"), str))
        self.beginResetModel()
        self._tutti = [(p, per_nome.get(p.name)) for p in percorsi]
        self._ricalcola()
        self.endResetModel()

    def applica_filtro(self, chiave):
        self.beginResetModel()
        self._filtro = chiave if chiave in _PREDICATI else "tutti"
        self._ricalcola()
        self.endResetModel()

    def _ricalcola(self):
        predicato = _PREDICATI.get(self._filtro, lambda v: True)
        if self._filtro == "tutti":
            self._visibili = list(self._tutti)
        else:
            self._visibili = [(p, v) for p, v in self._tutti
                              if v is not None and predicato(v)]

    def rowCount(self, _parent=QtCore.QModelIndex()):
        return len(self._visibili)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._visibili)):
            return None
        percorso, voce = self._visibili[index.row()]
        if role == RUOLO_PERCORSO:
            return percorso
        if role == RUOLO_VOCE:
            return voce
        if role == QtCore.Qt.DisplayRole:
            return percorso.name
        if role == QtCore.Qt.ToolTipRole:
            # Il motore che ha prodotto il frame, sotto al mouse in
            # pellicola: e' li' che l'utente guarda i frame, e senza
            # questo campo lo stesso 'aligned/' con volti di motori
            # diversi (ri-estrazione dei mancati con un motore diverso)
            # diventa indistinguibile.
            motore = indice.motore_di(voce) if voce is not None else None
            return testi.estrazione_motore_tooltip(motore)
        return None
