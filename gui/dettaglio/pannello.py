"""I sette pulsanti area: un interruttore e una pastiglia per gruppo.

L'interruttore decide se l'area PARTECIPA ALLA SELEZIONE -- spenta, i suoi
punti si disegnano smorzati e nessun gesto li prende. La pastiglia apre
QColorDialog e cambia il colore del gruppo.

La fabbrica del dialogo si inietta: QColorDialog e' modale e in un test
resterebbe aperto per sempre. E' la stessa forma di `trasporto` in
ClienteDettaglio e di `settings` in ScalaTesto.

**Nasce con le sette aree accese ma non annuncia niente alla
costruzione**: nessun ascoltatore c'e' ancora. Chi monta la finestra deve
spingere una volta `aree_attive()` e `colori()` dentro la Tela dopo aver
collegato i segnali, altrimenti la tela disegna tutto smorzato mentre il
pannello mostra sette spunte accese.

**Niente larghezza fissa sull'interruttore o sulla riga**: alla scala
tipografica xlarge l'etichetta cresce con il font dello stylesheet
applicato (gui/theme.py::stylesheet), e QCheckBox calcola da solo il suo
sizeHint sul font corrente. Fissare una larghezza qui tronca il testo
piu' lungo proprio dove il criterio di verifica guarda -- l'unica
larghezza fissa e' quella della pastiglia, che non porta testo.
"""
from PyQt5 import QtCore, QtWidgets

from gui import testi
from gui.dettaglio import gruppi as gruppi_mod


def _apri_dialogo(iniziale, parent):
    # Nessuna opzione ShowAlphaChannel: ColoriGruppi.imposta scrive sempre
    # `QColor.name()`, che scarta l'alpha in scrittura -- offrire un
    # cursore che poi viene buttato via silenziosamente sarebbe una scelta
    # peggiore di non offrirlo affatto.
    colore = QtWidgets.QColorDialog.getColor(
        iniziale, parent, testi.DETTAGLIO_COLORE_TITOLO)
    return colore if colore.isValid() else None


class PannelloAree(QtWidgets.QWidget):
    aree_cambiate = QtCore.pyqtSignal(object)
    colori_cambiati = QtCore.pyqtSignal(object)

    def __init__(self, colori, scegli_colore=None, parent=None):
        super().__init__(parent)
        self._colori = colori
        self._scegli = scegli_colore or _apri_dialogo
        self._zitto = False
        self.spunte = {}
        self.pastiglie = {}
        layout = QtWidgets.QVBoxLayout(self)
        titolo = QtWidgets.QLabel(testi.DETTAGLIO_AREE_TITOLO)
        titolo.setToolTip(testi.DETTAGLIO_AREE_TIP)
        layout.addWidget(titolo)
        for nome in gruppi_mod.NOMI:
            riga = QtWidgets.QHBoxLayout()
            spunta = QtWidgets.QCheckBox(testi.DETTAGLIO_AREE_NOMI[nome])
            spunta.setChecked(True)
            spunta.setToolTip(testi.DETTAGLIO_AREE_TIP)
            spunta.toggled.connect(self._su_spunta)
            pastiglia = QtWidgets.QPushButton()
            # Solo la larghezza e' fissa: l'altezza segue la riga, che
            # cresce con l'interruttore alla scala tipografica -- la
            # pastiglia smette quindi di essere quadrata alle scale
            # maggiori, ma resta un piccolo quadrato di colore, non testo
            # da troncare.
            pastiglia.setFixedWidth(28)
            pastiglia.setToolTip(testi.DETTAGLIO_AREE_TIP)
            pastiglia.clicked.connect(lambda _v, n=nome: self._su_pastiglia(n))
            self.spunte[nome] = spunta
            self.pastiglie[nome] = pastiglia
            self._dipingi(nome)
            riga.addWidget(spunta)
            riga.addWidget(pastiglia)
            layout.addLayout(riga)
        layout.addStretch(1)

    def _dipingi(self, nome):
        # QColor.name() scarta l'alpha: un colore semitrasparente scritto
        # a mano nel file di impostazioni si vede opaco qui, e la prima
        # ripicchiata dell'utente lo appiattisce anche sul disco.
        self.pastiglie[nome].setStyleSheet(
            "background: %s;" % self._colori.colore(nome).name())

    def aree_attive(self):
        """SEMPRE nell'ordine del volto, non in quello in cui si sono
        accese: chi la usa per disegnare disegnerebbe in ordine diverso a
        ogni giro."""
        return tuple(n for n in gruppi_mod.NOMI if self.spunte[n].isChecked())

    def imposta_aree_attive(self, nomi):
        # Una stringa e' una sequenza di caratteri: senza questa guardia
        # "naso" diventerebbe silenziosamente {'n', 'a', 's', 'o'}, che
        # spegne tutto.
        if isinstance(nomi, str):
            raise TypeError("nomi deve essere una sequenza di nomi di gruppo, non una stringa")
        nomi = set(nomi)
        self._zitto = True
        try:
            for nome in gruppi_mod.NOMI:
                self.spunte[nome].setChecked(nome in nomi)
        finally:
            self._zitto = False
        self._su_spunta()

    def colori(self):
        return self._colori.tutti()

    def _su_spunta(self, _valore=None):
        # Un annuncio per gesto, non uno per spunta: imposta_aree_attive ne
        # tocca sette, e sette annunci farebbero fare a chi ascolta sette
        # volte il lavoro -- fra cui, nella finestra, sette update().
        if self._zitto:
            return
        aree = self.aree_attive()
        self.aree_cambiate.emit(aree)

    def _su_pastiglia(self, nome):
        scelto = self._scegli(self._colori.colore(nome), self)
        if scelto is None:
            return
        self._colori.imposta(nome, scelto)
        self._dipingi(nome)
        self.colori_cambiati.emit(self._colori.tutti())
