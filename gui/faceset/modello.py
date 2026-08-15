"""Il modello della griglia: percorsi e voci d'indice, mai immagini.

Con 50 000 elementi Qt costruisce e disegna solo le celle visibili, ma
solo se il modello non tiene i pixel: quelli li chiede il delegato al
decodificatore, e la loro vita e' della LRU, non del modello.
"""
from PyQt5.QtCore import QAbstractListModel, QModelIndex, Qt

RUOLO_PERCORSO = Qt.UserRole
RUOLO_VOCE = Qt.UserRole + 1


class ModelloVolti(QAbstractListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._tutti = []
        self._voci = {}
        self._filtro = None
        self._visibili = []

    def imposta(self, percorsi, abbinati):
        """I percorsi della cartella e le voci d'indice che li accompagnano."""
        self.beginResetModel()
        self._tutti = list(percorsi)
        self._voci = dict(abbinati)
        self._ricalcola()
        self.endResetModel()

    def imposta_filtro(self, percorsi_o_none):
        self.beginResetModel()
        self._filtro = None if percorsi_o_none is None else set(percorsi_o_none)
        self._ricalcola()
        self.endResetModel()

    def _ricalcola(self):
        if self._filtro is None:
            self._visibili = list(self._tutti)
        else:
            self._visibili = [p for p in self._tutti if p in self._filtro]

    #override
    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._visibili)

    #override
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or index.row() >= len(self._visibili):
            return None
        percorso = self._visibili[index.row()]
        if role == RUOLO_PERCORSO:
            return percorso
        if role == RUOLO_VOCE:
            return self._voci.get(percorso)
        if role == Qt.DisplayRole:
            return percorso.name
        return None

    def percorso(self, row):
        return self._visibili[row]

    def voce(self, row):
        return self._voci.get(self._visibili[row])

    def percorsi_visibili(self):
        return list(self._visibili)

    def totali(self):
        """Quanti volti ha la cartella, filtro o non filtro.

        `rowCount()` conta i visibili: da solo non basta a dire «3 di 96»,
        che e' l'unica forma in cui un filtro si legge come una fetta e non
        come una cartella vuota.
        """
        return len(self._tutti)

    def righe_di(self, percorsi):
        cercati = set(percorsi)
        return [i for i, p in enumerate(self._visibili) if p in cercati]
