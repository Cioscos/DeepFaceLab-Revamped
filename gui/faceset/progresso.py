"""La pila delle barre di avanzamento di un job.

Le barre del sorting sono CONSECUTIVE -- interact ne ammette una sola per
volta -- quindi la pila non e' una lista di barre parallele: e' una
cronologia, in cui quelle chiuse restano come consuntivo e solo l'ultima
si muove. Una nuova apertura chiude implicitamente qualunque barra ancora
aperta, anche se oggi il canale non produce mai un `open` senza un `close`
prima (`ProgressLog.apri` lo garantisce lato scrittore) -- questo widget
consuma l'uscita di un altro processo, e la sua invariante non deve
dipendere dalla disciplina di chi scrive.

Ogni numero qui viene da un altro processo, quindi passa da gui/numeri.py
prima di toccare un widget: un total a zero e' una divisione per zero, e
un NaN attraversa somme e confronti senza sollevare per morire molto piu'
tardi dentro un paintEvent -- che, sollevando, si porta via il processo
intero con dentro ogni altro training aperto.

Il predicato e' `intero_qt_utilizzabile` e non il solo `numero_finito`:
`setMaximum` e `setValue` prendono un `int` a 32 bit, e un `1e300` --
finito, quindi accettato da un controllo di sola finitezza -- solleva
`OverflowError` dentro `applica`. E `applica` e' chiamata da uno SLOT, non
da un paintEvent: la differenza non conta, perche' su questo PyQt5
un'eccezione che risale da uno slot fa abortire il processo esattamente
come una che risale da un paintEvent (misurato, EXIT=134).

`barre()` legge valore e totale dal widget vero (`QProgressBar.value()`/
`.maximum()`), mai da una copia parallela: due fonti per lo stesso fatto
sono la classe di difetto che questo modulo esiste per evitare.
"""
from PyQt5.QtWidgets import QProgressBar, QVBoxLayout, QWidget

from gui import testi
from gui.numeri import intero_qt_utilizzabile


def _totale_utilizzabile(valore):
    if not intero_qt_utilizzabile(valore):
        return None
    # Nessun `try` attorno all'`int()`: il predicato ha gia' garantito un
    # numero finito e dentro l'intervallo, quindi la conversione non ha piu'
    # modo di sollevare -- una cattura qui direbbe che ce l'ha.
    intero = int(valore)
    return intero if intero > 0 else None


class PilaProgresso(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._barre = {}   # id -> [QProgressBar, desc, chiusa]
        self._ordine = []
        self._avvio = None

    def mostra_avvio(self, desc):
        """Una barra che PULSA (minimo e massimo a zero, il modo
        indeterminato di Qt) fra il click e la prima riga del figlio.

        Serve perche' quell'intervallo non e' breve: il figlio importa torch
        e carica il modello, e sono 5-10 s in cui l'unico segnale era un
        bottone che si spegneva. Non e' una barra del protocollo: non ha id,
        non entra in `_barre` e non compare in `barre()`, cosi' nessun test
        del canale la conta per sbaglio.
        """
        if self._avvio is not None:
            return
        barra = QProgressBar()
        barra.setMinimum(0)
        barra.setMaximum(0)
        barra.setFormat(testi.progress_bar_format(str(desc or "")))
        self._layout.addWidget(barra)
        self._avvio = barra

    def avvio_visibile(self):
        """M2 della revisione finale: fino a quel round questo predicato
        non aveva ne' un chiamante di produzione ne' un docstring che lo
        dichiarasse aiutante dei test (a differenza di
        `TrasportoAsincrono.consegna_tutto`, che lo dice di se'). Ora ne
        ha uno: `gui/estrazione/pagina.py::PaginaEstrazione._su_progresso`
        lo legge PRIMA e DOPO `applica()` per accorgersi della transizione
        dalla barra indeterminata alla prima barra vera (M5), senza
        dipendere dalla geometria di un widget -- una trappola misurata
        sotto la piattaforma offscreen. I test lo usano per la stessa
        ragione."""
        return self._avvio is not None

    def togli_avvio(self):
        """Il gemello pubblico di `_togli_avvio`, per chi la barra pulsante
        deve spegnerla a mano.

        `_apri` la spegne da sola alla prima riga `open` del canale di
        avanzamento, e per un job batch basta. La sessione manuale della
        pagina di estrazione pero' non ha nessun canale: parla col figlio a
        richiesta/risposta (gui/estrazione/trasporto.py), quindi nessun
        `open` arrivera' mai, e chi riceve la prima risposta deve poterla
        togliere. Chiamarla due volte non e' un errore."""
        self._togli_avvio()

    def _togli_avvio(self):
        if self._avvio is None:
            return
        self._layout.removeWidget(self._avvio)
        self._avvio.deleteLater()
        self._avvio = None

    def applica(self, riga):
        if not isinstance(riga, dict):
            return
        op = riga.get("op")
        if op == "open":
            self._apri(riga)
        elif op == "inc":
            self._inc(riga)
        elif op == "close":
            self._chiudi(riga)

    def _apri(self, riga):
        self._togli_avvio()
        ident = riga.get("id")
        if ident is None or ident in self._barre:
            return
        for voce in self._barre.values():
            if not voce[2]:
                self._chiudi_voce(voce)
        totale = _totale_utilizzabile(riga.get("total"))
        barra = QProgressBar()
        barra.setMinimum(0)
        barra.setMaximum(totale if totale is not None else 0)
        iniziale = riga.get("initial") or 0
        iniziale = int(iniziale) if intero_qt_utilizzabile(iniziale) else 0
        barra.setValue(iniziale)
        barra.setFormat(testi.progress_bar_format(str(riga.get("desc") or "")))
        self._layout.addWidget(barra)
        self._barre[ident] = [barra, str(riga.get("desc") or ""), False]
        self._ordine.append(ident)

    def _inc(self, riga):
        voce = self._barre.get(riga.get("id"))
        if voce is None:
            return
        n = riga.get("n")
        if not intero_qt_utilizzabile(n):
            return
        voce[0].setValue(int(n))

    def _chiudi(self, riga):
        voce = self._barre.get(riga.get("id"))
        if voce is None:
            return
        self._chiudi_voce(voce)

    def _chiudi_voce(self, voce):
        voce[2] = True
        totale = self._totale_di(voce[0])
        if totale is not None:
            voce[0].setValue(totale)

    @staticmethod
    def _totale_di(barra):
        massimo = barra.maximum()
        return None if massimo == 0 else massimo

    def barre(self):
        risultato = []
        for ident in self._ordine:
            barra, desc, chiusa = self._barre[ident]
            risultato.append((desc, barra.value(), self._totale_di(barra), chiusa))
        return risultato

    def pulisci(self):
        self._togli_avvio()
        for ident in self._ordine:
            barra = self._barre[ident][0]
            self._layout.removeWidget(barra)
            barra.deleteLater()
        self._barre.clear()
        self._ordine.clear()
