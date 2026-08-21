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

    def aggiorna_voci(self, voci):
        """Le voci arrivate mentre il job gira. Torna quante righe VISIBILI
        ha toccato.

        Nessun `beginResetModel`: un reset al secondo perderebbe selezione e
        scorrimento. E l'insieme visibile NON si ricalcola -- una voce nuova
        che entrerebbe nel filtro acceso resta fuori finche' l'utente non
        tocca un filtro o il job non finisce. Il contatore del filtro sale
        lo stesso (lo calcola la pagina da `_voci`, non da qui): si vede che
        il lavoro procede senza che la lista si riordini sotto le mani.
        """
        per_nome = dict((v.get("nome"), v) for v in voci
                        if isinstance(v, dict) and isinstance(v.get("nome"), str))
        if not per_nome:
            return 0
        for i, (percorso, _voce) in enumerate(self._tutti):
            nuova = per_nome.get(percorso.name)
            if nuova is not None:
                self._tutti[i] = (percorso, nuova)
        toccate = 0
        for riga, (percorso, _voce) in enumerate(self._visibili):
            nuova = per_nome.get(percorso.name)
            if nuova is None:
                continue
            self._visibili[riga] = (percorso, nuova)
            indice = self.index(riga, 0)
            self.dataChanged.emit(indice, indice)
            toccate += 1
        return toccate

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

    def totale(self):
        """Quanti frame ha la cartella, filtro ignorato: `rowCount` conta i
        VISIBILI, e i due numeri divergono appena un filtro e' acceso --
        che e' proprio il caso in cui la pagina deve scegliere quale dei due
        messaggi vuoti mostrare."""
        return len(self._tutti)

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
            # Due assenze diverse: la voce che manca del tutto (frame mai
            # estratto) e la voce senza motore (rapporto ricostruito, o piu'
            # vecchio del campo). Prima dicevano la stessa cosa.
            if voce is None:
                return testi.ESTRAZIONE_FRAME_NON_NEL_RAPPORTO
            return testi.estrazione_motore_tooltip(indice.motore_di(voce))
        return None
