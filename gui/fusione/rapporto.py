"""Il rapporto di fine fusione: cio' che Merger.main stampava in console e
perdeva -- i frame copiati senza volto e quelli con piu' allineati -- piu' i
millisecondi per frame del batch appena finito.

`indice_di` arriva dalla pagina perche' il rapporto porta NOMI di file --
non indici, a differenza delle risposte ai comandi -- e chi sa tradurre un
nome in una riga della pellicola e' la pagina, non questo widget.

Ogni valore si valida prima di toccare un widget: l'evento e' JSON scritto
da un altro processo, e questo e' uno slot.
"""
import json
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from gui import numeri, testi

RUOLO_INDICE = Qt.UserRole + 1

# Quante righe di elenco si vedono senza scorrere. Un tetto ci vuole: senza,
# un rapporto con cinquanta frame senza volto chiederebbe piu' altezza di
# quanta la finestra ne abbia -- misurato offscreen, la pagina in stato
# `done` ne chiedeva 1142 px di minimo contro gli ~800 disponibili in una
# finestra da 900, e cio' che sporgeva era la fine dell'elenco.
RIGHE_VISIBILI = 4


def _intestazione(testo):
    """L'etichetta di un elenco, che va a capo invece di allargare.

    La seconda frase e' lunga (parla del motion blur spento per tutto il
    video), e un'etichetta che non va a capo impone la propria larghezza a
    tutta la colonna: misurato offscreen, la pagina in stato `done` chiedeva
    227 px di larghezza minima in piu' di quanti ne chiede in `tuning`."""
    etichetta = QLabel(testo)
    etichetta.setWordWrap(True)
    return etichetta


def _limita_altezza(lista):
    """L'elenco si ferma a RIGHE_VISIBILI righe e poi scorre. L'altezza
    della riga viene dalle metriche del font, non da un numero fisso: alla
    scala tipografica piu' grande un numero fisso mostrerebbe meno righe di
    quante ne dichiara."""
    riga = lista.fontMetrics().height() + 6
    lista.setMinimumHeight(riga * 2)
    lista.setMaximumHeight(riga * RIGHE_VISIBILI)


class PannelloRapporto(QWidget):
    frame_scelto = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._evento = None
        self._indice_di = None
        colonna = QVBoxLayout(self)
        self.titolo = QLabel(testi.FUSIONE_REPORT)
        self.titolo.setProperty("ruolo", "sezione")
        colonna.addWidget(self.titolo)
        self.etichetta_ms = QLabel("")
        colonna.addWidget(self.etichetta_ms)
        colonna.addWidget(_intestazione(testi.FUSIONE_REPORT_NO_FACE))
        self.lista_senza_volto = QListWidget()
        _limita_altezza(self.lista_senza_volto)
        self.lista_senza_volto.itemClicked.connect(self._su_click)
        colonna.addWidget(self.lista_senza_volto)
        colonna.addWidget(_intestazione(testi.FUSIONE_REPORT_MULTI))
        self.lista_multipli = QListWidget()
        _limita_altezza(self.lista_multipli)
        self.lista_multipli.itemClicked.connect(self._su_click)
        colonna.addWidget(self.lista_multipli)

    @staticmethod
    def leggi_da_file(percorso):
        """Il rapporto scritto a fine batch, o None se non si legge.

        Serve a pagina riaperta: l'evento passa una volta sola, il file
        resta. Chi chiude la GUI a batch acceso non ha nessun file --
        merge_report.json lo scrive solo la fine spontanea."""
        try:
            dati = json.loads(Path(percorso).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return dati if isinstance(dati, dict) else None

    def imposta(self, evento, indice_di):
        """Riempie le due liste. `evento` puo' essere None (nessun rapporto
        da mostrare) o un dizionario di qualunque forma."""
        self._evento = evento if isinstance(evento, dict) else None
        self._indice_di = indice_di
        self.lista_senza_volto.clear()
        self.lista_multipli.clear()
        dati = self._evento or {}
        senza_volto = dati.get("senza_volto")
        for nome in senza_volto if isinstance(senza_volto, list) else ():
            if isinstance(nome, str):
                self._aggiungi(self.lista_senza_volto, nome, self._indice(nome))
        multipli = dati.get("multipli")
        for voce in multipli if isinstance(multipli, list) else ():
            if isinstance(voce, (list, tuple)) and len(voce) == 2 and isinstance(voce[0], str):
                nomi = voce[1] if isinstance(voce[1], list) else []
                self._aggiungi(self.lista_multipli,
                               testi.fusione_voce_multipli(voce[0], len(nomi)),
                               self._indice(voce[0]))
        ms = dati.get("ms_per_frame")
        self.etichetta_ms.setText(testi.fusione_ms_per_frame(float(ms))
                                  if numeri.numero_finito(ms) else "")

    def _indice(self, nome):
        """La riga della pellicola per quel nome. Il traduttore e' della
        pagina: se non c'e' o inciampa, la voce resta in elenco e non e'
        cliccabile -- un rapporto e' comunque da leggere."""
        if self._indice_di is None:
            return None
        try:
            return self._indice_di(nome)
        except Exception:
            return None

    def _aggiungi(self, lista, testo, indice):
        voce = QListWidgetItem(testo)
        # Il testo intero anche a colonna stretta: alla scala tipografica
        # piu' grande l'elenco sta in poco piu' di 200 px e un nome lungo
        # esce dal bordo (si legge scorrendo, o col suggerimento).
        voce.setToolTip(testo)
        if isinstance(indice, int) and numeri.intero_qt_utilizzabile(indice):
            voce.setData(RUOLO_INDICE, indice)
        lista.addItem(voce)

    def _su_click(self, voce):
        indice = voce.data(RUOLO_INDICE)
        if isinstance(indice, int):
            self.frame_scelto.emit(indice)
