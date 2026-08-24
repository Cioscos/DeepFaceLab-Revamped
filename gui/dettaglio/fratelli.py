"""I volti dello STESSO fotogramma, in una striscia di miniature.

Non e' l'insieme che sfogliano le frecce. Quello lo decide la pagina che
apre la finestra -- l'intera griglia dalla cura, i soli fratelli
dall'estrazione -- e resta com'e'. Questa striscia guarda una relazione
sola, `source_filename`, e la guarda sempre: dalla pagina di cura, dopo un
riordino, i fratelli sono a migliaia di celle di distanza nella griglia, e
questo e' l'unico posto in cui si vedono insieme.
"""
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets

from gui import testi


class StrisciaFratelli(QtWidgets.QWidget):
    """Chi la possiede deve richiamare `imposta_fratelli` in due momenti:
    quando riceve `scelto`, per confermare o correggere la marcatura
    ottimistica del click; e quando le frecce sfogliano a un fotogramma
    diverso, perche' la striscia non ascolta nessun altro canale da sola.
    """
    scelto = QtCore.pyqtSignal(str)
    LATO_MINIATURA = 64

    def __init__(self, parent=None):
        super().__init__(parent)
        self._percorsi = []
        self._corrente = None
        self.bottoni = []
        self._riga = QtWidgets.QHBoxLayout(self)
        self._riga.setContentsMargins(0, 0, 0, 0)
        self._riga.addStretch(1)
        self.setToolTip(testi.DETTAGLIO_FRATELLI_TIP)
        self.setVisible(False)

    def percorsi(self):
        return list(self._percorsi)

    def corrente(self):
        return self._corrente

    def _svuota(self):
        for bottone in self.bottoni:
            self._riga.removeWidget(bottone)
            bottone.setParent(None)
            bottone.deleteLater()
        self.bottoni = []

    def imposta_fratelli(self, percorsi, corrente):
        """Con meno di due volti la striscia sparisce: una striscia da un
        elemento promette una scelta che non c'e'."""
        self._svuota()
        percorsi = [str(p) for p in (percorsi or [])]
        if len(percorsi) < 2:
            self._percorsi = []
            self._corrente = None
            self.setVisible(False)
            return
        self._percorsi = percorsi
        self._corrente = str(corrente) if corrente is not None else None
        for percorso in percorsi:
            self.bottoni.append(self._bottone(percorso))
        self.setVisible(True)

    def _bottone(self, percorso):
        bottone = QtWidgets.QToolButton()
        bottone.setCheckable(True)
        bottone.setChecked(percorso == self._corrente)
        bottone.setToolTip(Path(percorso).name)
        bottone.setIconSize(QtCore.QSize(self.LATO_MINIATURA, self.LATO_MINIATURA))
        pixmap = QtGui.QPixmap(percorso)
        if not pixmap.isNull():
            bottone.setIcon(QtGui.QIcon(pixmap.scaled(
                self.LATO_MINIATURA, self.LATO_MINIATURA,
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)))
        else:
            # Il servizio di dettaglio scarta gia' i file rotti prima di
            # annunciarli (operazione "frame"): questo ramo difende da un
            # file che diventa illeggibile DOPO quella risposta, o da un
            # chiamante che passa un elenco non filtrato. Si mostra la coda
            # dello stelo, non la testa: l'estrazione nomina i fratelli
            # `<stelo>_<indice>.jpg`, e la testa e' identica per tutti i
            # fratelli dello stesso fotogramma.
            bottone.setText(Path(percorso).stem[-6:])
        bottone.clicked.connect(lambda _v, p=percorso: self._su_click(p))
        self._riga.insertWidget(len(self.bottoni), bottone)
        return bottone

    def _su_click(self, percorso):
        # Riaprire il volto gia' aperto costerebbe un giro al servizio e
        # butterebbe la pila di undo per niente.
        if percorso == self._corrente:
            for bottone, p in zip(self.bottoni, self._percorsi):
                bottone.setChecked(p == self._corrente)
            return
        # Marcatura ottimistica: il servizio di dettaglio e' asincrono, e
        # senza marcare subito il bottone appena scelto la striscia
        # mostrerebbe due volti marcati -- quello vecchio e quello nuovo --
        # per tutto il giro di andata e ritorno. Aggiornare anche
        # `_corrente` disarma la guardia sul click ripetuto qui sopra: senza
        # questa riga un secondo click sullo stesso bottone ripeterebbe il
        # giro al servizio e la cancellazione dell'undo che quella guardia
        # esiste per evitare. Chi ascolta `scelto` conferma o corregge
        # entrambi richiamando `imposta_fratelli` col volto davvero aperto.
        for bottone, p in zip(self.bottoni, self._percorsi):
            bottone.setChecked(p == percorso)
        self._corrente = percorso
        self.scelto.emit(percorso)
