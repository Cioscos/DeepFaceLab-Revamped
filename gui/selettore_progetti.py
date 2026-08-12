"""Il selettore di progetti: quale progetto guarda la finestra, e chi e' occupato.

Un pulsante col menu invece di una QComboBox: la lista cambia sotto (un
progetto creato, uno eliminato) e una combo che si ricostruisce mentre e'
aperta e' una fonte di sorprese. Il pallino accanto a un nome dice che
quel progetto ha passi in corso, che dopo questo lavoro puo' succedere
mentre se ne guarda un altro.
"""
from collections import Counter

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QAction, QMenu, QToolButton

from gui import testi

PALLINO = "●"


class SelettoreProgetti(QToolButton):
    progetto_scelto = pyqtSignal(object)

    def __init__(self, archivio, parent=None):
        super().__init__(parent)
        self._archivio = archivio
        self._progetti = []
        self._occupati = set()
        self.setToolTip(testi.PROJECT_SELECTOR_TIP)
        self.setPopupMode(QToolButton.InstantPopup)
        self._menu = QMenu(self)
        self.setMenu(self._menu)
        self.setText(testi.project_button(testi.NO_PROJECT))

    def aggiorna(self, progetti, occupati):
        """Ricostruisce l'elenco. `occupati` e' un iterabile di cartelle, una
        per job attivo -- non un insieme: due job sullo stesso progetto
        devono contare due, non collassare in una sola voce (il tooltip
        deve mostrare il conteggio vero; prima di questa correzione
        diceva sempre "1 running" qualunque fosse il numero
        vero, perche' un insieme non porta i duplicati). Un Counter accetta
        anche un insieme vero passato dai test piu' vecchi -- ogni voce
        conta 1, comportamento identico a prima per loro."""
        self._progetti = list(progetti)
        self._occupati = Counter(occupati)
        self._menu.clear()
        for progetto in self._progetti:
            quanti = self._occupati_quanti(progetto)
            etichetta = progetto.nome if not quanti else (
                "%s  %s" % (PALLINO, progetto.nome))
            action = QAction(etichetta, self._menu)
            if quanti:
                action.setToolTip(testi.project_with_jobs(progetto.nome, quanti))
            action.triggered.connect(
                lambda _c=False, p=progetto: self.progetto_scelto.emit(p))
            self._menu.addAction(action)

    def _occupati_quanti(self, progetto):
        # Confronta percorsi, non l'identita' del filesystem: un pallino
        # mancante sarebbe un difetto cosmetico, non una corsa sui dati, e
        # non vale il costo di uno stat per progetto a ogni ridisegno --
        # vedi gui/main_window.py::_aggiorna_selettore. Il conteggio (non
        # solo la presenza) e' quello che project_with_jobs mette nel
        # tooltip.
        return self._occupati.get(progetto.cartella, 0)

    def imposta_corrente(self, progetto):
        self.setText(testi.project_button(
            progetto.nome if progetto is not None else testi.NO_PROJECT))

    def nomi_mostrati(self):
        return [p.nome for p in self._progetti]

    def e_segnato(self, nome):
        for action in self._menu.actions():
            if action.text().endswith(nome):
                return action.text().startswith(PALLINO)
        return False

    def scegli(self, nome):
        for progetto in self._progetti:
            if progetto.nome == nome:
                self.progetto_scelto.emit(progetto)
                return
