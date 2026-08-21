"""La colonna dei comandi della pagina di estrazione, che e' anche la sua
spiegazione.

Tre scelte, e ognuna ha una ragione misurata dietro.

**Fuori dal frame.** La finestra `cv2` che questa pagina sostituisce
disegnava sei righe di aiuto SOPRA il fotogramma (mainscripts/Extractor.py,
`get_draw_text_lines`), con un tasto per nasconderle: coprivano proprio la
parte alta dell'immagine, dove sta la testa. Qui i comandi stanno in una
colonna a lato, e il frame non si tocca.

**Il tasto e' scritto, non solo nel tooltip.** L'editor XSeg mette la
scorciatoia nel solo tooltip (`core/qtex/qtex.py::QActionEx`, con
`shortcut_in_tooltip`) e non ha nessun pannello d'aiuto -- il suo
`help_frame` e' un attributo inizializzato a None e mai costruito. Il
tooltip serve chi cerca; questa colonna serve chi non sa di dover cercare,
che e' la stessa ragione per cui esiste `gui/fascia_aiuto.py`.

**Le scorciatoie sono `Qt.WidgetWithChildrenShortcut`.** Misurato il
2026-08-17 con due schede in un QTabWidget: una pagina non corrente e'
nascosta e Qt non attiva le scorciatoie dei widget nascosti, quindi nessuno
dei tre contesti scatta da un'altra scheda -- il contesto stretto si sceglie
per il caso futuro di un campo di testo dentro questa pagina, a cui
`WindowShortcut` mangerebbe le lettere mentre ci si scrive. Il prezzo e' che
col focus sulla LINGUETTA della scheda non scattano: per questo la pagina
prende il focus quando la sua scheda diventa corrente
(gui/main_window.py::_su_scheda_cambiata).

**L'appartenenza (`addAction`) NON e' di questo modulo.** `WidgetWithChildrenShortcut`
scatta quando ad avere il focus e' il widget su cui l'azione e' stata
aggiunta con `addAction()`, o un suo discendente -- non il widget che ha
costruito il QAction. `ColonnaComandi` costruisce le azioni (le possiede
come QObject, per il ciclo di vita) ma non le aggiunge mai a se stessa: i
suoi bottoni sono `Qt.NoFocus`, quindi nessun discendente della colonna puo'
mai avere il focus, e un'azione aggiunta qui sarebbe silenziosamente morta.
Tocca a chi monta la colonna scegliere l'antenato giusto -- vedi
`gui/estrazione/pagina.py::PaginaEstrazione.__init__`, che aggiunge quasi
tutto a se stessa (tela, colonna e pellicola sono tutte sue discendenti) e
le quattro frecce alla `Tela` soltanto (vedi `CHIAVI_FRECCE` sotto).
Verificato con `QTest.keyClick` su una `MainWindow` vera, non solo letto: con
le azioni possedute dalla colonna, `pagina.setFocus()` (cio' che fa
main_window.py) e `tela.setFocus()` (dove clicca l'utente) non facevano
scattare NESSUNO dei comandi tranne "conferma", che scattava per un'altra
via (`Tela.keyPressEvent` intercetta Return da prima di questo modulo).
"""
import collections

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (QAction, QFrame, QLabel, QPushButton, QVBoxLayout,
                             QWidget)

from gui import testi

Comando = collections.namedtuple("Comando", "chiave etichetta tasto gruppo")

# I gruppi sono i riquadri della colonna, nell'ordine in cui compaiono.
GRUPPO_FRAME = "Frame"
GRUPPO_RETTANGOLO = "Rectangle"
GRUPPO_AZIONE = "Frame action"

# I tasti sono quelli della finestra `cv2` (Extractor.py:421-427) ovunque
# esista un equivalente: chi conosce DeepFaceLab li ha nelle dita, e questa
# pagina sostituisce quella finestra, non ne apre una diversa. Le frecce e
# `R` non hanno equivalente li' -- sono le due aggiunte.
COMANDI = (
    Comando("precedente",     "Previous frame",   ",",     GRUPPO_FRAME),
    Comando("successivo",     "Next frame",       ".",     GRUPPO_FRAME),
    Comando("salta-restanti", "Skip the rest",    "Q",     GRUPPO_FRAME),
    Comando("rileva",         "Detect face",      "R",     GRUPPO_FRAME),
    Comando("blocca",         "Lock rectangle",   "L",     GRUPPO_RETTANGOLO),
    Comando("muovi-sinistra", "Move left",        "Left",  GRUPPO_RETTANGOLO),
    Comando("muovi-destra",   "Move right",       "Right", GRUPPO_RETTANGOLO),
    Comando("muovi-su",       "Move up",          "Up",    GRUPPO_RETTANGOLO),
    Comando("muovi-giu",      "Move down",        "Down",  GRUPPO_RETTANGOLO),
    Comando("ingrandisci",    "Bigger",           "+",     GRUPPO_RETTANGOLO),
    Comando("rimpicciolisci", "Smaller",          "-",     GRUPPO_RETTANGOLO),
    Comando("accuratezza",    "Accurate landmarks", "A",   GRUPPO_RETTANGOLO),
    Comando("vettore",        "Vector fallback",  "V",     GRUPPO_RETTANGOLO),
    Comando("conferma",       "Confirm and save", "Return", GRUPPO_AZIONE),
    Comando("salta",          "Skip frame",       "Space", GRUPPO_AZIONE),
)

_ORDINE_GRUPPI = (GRUPPO_FRAME, GRUPPO_RETTANGOLO, GRUPPO_AZIONE)

# Le quattro chiavi che chi monta la colonna deve aggiungere alla TELA e non
# alla pagina: sono le uniche che una QListView (la pellicola) usa gia' per
# la propria navigazione. Con lo scope sulla pagina le ruberebbero appena il
# focus fosse sulla striscia; sulla tela scattano solo quando e' lei ad
# avere il focus, e la striscia si tiene le sue frecce senza dipendere da
# nessuna abilitazione.
CHIAVI_FRECCE = ("muovi-sinistra", "muovi-destra", "muovi-su", "muovi-giu")


class ColonnaComandi(QWidget):
    scelto = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._azioni = {}
        self._bottoni = {}
        radice = QVBoxLayout(self)
        radice.setContentsMargins(0, 0, 0, 0)
        for gruppo in _ORDINE_GRUPPI:
            riquadro = QFrame()
            riquadro.setFrameShape(QFrame.StyledPanel)
            dentro = QVBoxLayout(riquadro)
            titolo = QLabel(gruppo)
            titolo.setProperty("ruolo", "minore")
            dentro.addWidget(titolo)
            for c in (x for x in COMANDI if x.gruppo == gruppo):
                dentro.addWidget(self._costruisci(c))
            radice.addWidget(riquadro)
        radice.addStretch(1)

    def _costruisci(self, comando):
        # Il testo porta il tasto: e' la spiegazione fissa, non un tooltip.
        # Fra parentesi quadre e non dopo una tabulazione, che in un
        # QPushButton non si vede.
        bottone = QPushButton(testi.estrazione_comando_etichetta(comando.etichetta,
                                                                 comando.tasto))
        bottone.setProperty("tasto", comando.tasto)
        # NoFocus: un bottone che prende il focus al click lo toglie alla
        # tela, e il gesto dopo (una freccia) andrebbe altrove. Misurato.
        bottone.setFocusPolicy(Qt.NoFocus)
        # Il tooltip della QAction qui sotto non raggiunge il bottone: un
        # QPushButton non ha setDefaultAction, e clicked.connect non
        # trasporta niente. Sul bottone va il SOLO tasto -- chi ci passa
        # sopra cerca quello, non una ripetizione dell'etichetta che ha gia'
        # sotto gli occhi. Non e' un letterale: e' un dato di COMANDI.
        bottone.setToolTip(comando.tasto)
        azione = QAction(comando.etichetta, self)
        azione.setShortcut(QKeySequence(comando.tasto))
        azione.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        azione.setToolTip(testi.estrazione_comando_tip(comando.etichetta, comando.tasto))
        azione.triggered.connect(lambda _=False, k=comando.chiave: self.scelto.emit(k))
        bottone.clicked.connect(azione.trigger)
        # NIENTE self.addAction(azione) qui: vedi la nota in testa al modulo
        # -- l'appartenenza che rende la scorciatoia raggiungibile da
        # tastiera tocca a chi monta la colonna, non a chi costruisce
        # l'azione.
        self._azioni[comando.chiave] = azione
        self._bottoni[comando.chiave] = bottone
        return bottone

    def azione(self, chiave):
        return self._azioni[chiave]

    def bottone(self, chiave):
        return self._bottoni[chiave]

    def imposta_abilitati(self, mapping):
        """Accende e spegne insieme l'azione e il suo bottone.

        E' cio' che spegne l'intera colonna fuori dalla sessione manuale
        (vedi PaginaEstrazione._rigenera_comandi): un'azione disabilitata non
        scatta da tastiera ne' il suo bottone risponde al click. Le frecce
        NON dipendono da questo per lasciare in pace la navigazione della
        pellicola -- quello e' lo SCOPE (sono aggiunte alla Tela, non alla
        pagina: vedi CHIAVI_FRECCE), non l'abilitazione: una freccia
        disabilitata e con lo scope sulla pagina ruberebbe comunque la
        selezione della lista appena tornasse abilitata.
        """
        for chiave, acceso in mapping.items():
            if chiave not in self._azioni:
                continue
            self._azioni[chiave].setEnabled(bool(acceso))
            self._bottoni[chiave].setEnabled(bool(acceso))

    def imposta_spunta(self, chiave, acceso):
        """Il bottone di un comando a stato si vede premuto o no.

        Senza, l'unico modo di sapere se l'accuratezza e' attiva sarebbe
        premere e guardare cosa cambia -- e l'accuratezza cambia la
        precisione dei landmark, non qualcosa che si veda a colpo d'occhio.
        Il bottone diventa checkable alla prima chiamata invece che alla
        costruzione: cosi' la tabella COMANDI resta una tabella di dati e
        non deve portare una quarta colonna per tre voci su quindici.
        """
        if chiave not in self._bottoni:
            return
        bottone = self._bottoni[chiave]
        if not bottone.isCheckable():
            bottone.setCheckable(True)
        bottone.setChecked(bool(acceso))
