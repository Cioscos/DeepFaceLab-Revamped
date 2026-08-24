"""The shell's dark theme.

The colours are the mask editor's own (`core/qtex/qtex.py::QDarkPalette`),
duplicated here on purpose: the `gui` package imports nothing from the
application, not even its Qt helpers. A guard compares the two palettes
role by role, so the duplication cannot drift.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QComboBox, QStyledItemDelegate

from gui.workspace import STATE_BLOCKED, STATE_DONE, STATE_READY

TEXT = QColor(200, 200, 200)
WINDOW = QColor(53, 53, 53)
BASE = QColor(25, 25, 25)
ACCENT = QColor(42, 130, 218)

#Business colors for the three STATE_* a stage/step can be in -- not derived
#from the dark palette above, so they are spelled out here rather than
#composed from WINDOW/TEXT/ACCENT/BASE like everything else in this file.
#Single source for both the pipeline bar's pills (gui.main_window) and the
#step list's per-row badge (gui.delegato_passi), so the two can never drift
#apart the way a hand-copied hex string would let them.
STATO_COLORE = {
    STATE_DONE: "#2e7d32",
    STATE_READY: "#1565c0",
    STATE_BLOCKED: "#9e9e9e",
}

#The type scale, in one place. The points below are the ones at scale 1.0;
#`punti()` multiplies them by the factor the user picked. A new role is
#added here, not in the style sheet: that one only reads.
PUNTI = {
    "base":     11.0,   # labels, controls, lists
    "tessera":  16.0,   # the value shown on a status tile
    "passo":    12.5,   # a step's name in the list
    "sezione":  10.0,   # section titles, spaced small caps
    "minore":   10.0,   # descriptions, pills, badges
    "console":  10.0,   # the console's fixed-pitch font
}

SCALE_NAMES = ("normal", "large", "xlarge")
SCALE_FACTORS = {"normal": 1.0, "large": 1.15, "xlarge": 1.3}

#Kept as an alias: the console still reads a plain point size, and wiring
#the chosen scale into it is left for whoever reads the user's choice.
CONSOLE_FONT_POINT_SIZE = PUNTI["console"]


def punti(ruolo, scala=1.0):
    """The size of a role at the given scale, in points."""
    return round(PUNTI[ruolo] * scala, 1)


def scala_di(font):
    """The scale factor a widget's own font implies, read back from it.

    `QWidget { font-size: base*scale }` above is what the style sheet hands
    every widget, so a font that came through it already carries the user's
    choice -- dividing it by the role's size at scale 1.0 gets the factor
    back without asking anyone. That is the point: a painter that read the
    chosen scale from `gui.preferenze` instead would be a second source for
    the same fact, free to disagree with the sheet that is actually drawing
    everything else on screen.

    A font measured in pixels rather than points (`pointSizeF()` is -1 then)
    says nothing about the scale, and neither does a nonsensical one: both
    fall back to 1.0 rather than producing a size out of a negative number.
    """
    misura = font.pointSizeF()
    if misura <= 0:
        return 1.0
    return misura / PUNTI["base"]


def dark_palette():
    """The palette applied to the whole application."""
    p = QPalette()
    p.setColor(QPalette.Window, WINDOW)
    p.setColor(QPalette.WindowText, TEXT)
    p.setColor(QPalette.Base, BASE)
    p.setColor(QPalette.AlternateBase, WINDOW)
    p.setColor(QPalette.ToolTipBase, TEXT)
    p.setColor(QPalette.ToolTipText, TEXT)
    p.setColor(QPalette.Text, TEXT)
    p.setColor(QPalette.Button, WINDOW)
    p.setColor(QPalette.ButtonText, Qt.white)
    p.setColor(QPalette.BrightText, Qt.red)
    p.setColor(QPalette.Link, ACCENT)
    p.setColor(QPalette.Highlight, ACCENT)
    p.setColor(QPalette.HighlightedText, Qt.black)
    return p


_QSS = """
QWidget {
    font-size: %(base).1fpt;
}
QPushButton {
    padding: 5px 14px;
    border: 1px solid %(bordo)s;
    border-radius: 5px;
    background: %(finestra)s;
    color: %(testo)s;
}
QPushButton:hover {
    border-color: %(accento)s;
}
QPushButton:pressed {
    background: %(lieve)s;
}
QPushButton:checked {
    border-color: %(accento)s;
    color: %(accento)s;
}
QPushButton:disabled {
    color: %(testo_lieve)s;
}
QPushButton#start {
    background: %(accento)s;
    border-color: %(accento)s;
    color: %(testo)s;
    font-weight: bold;
}
QPushButton[ruolo="stop"] {
    border-color: palette(bright-text);
    color: palette(bright-text);
}
QPushButton[stato="done"] {
    background-color: %(stato_done)s;
    color: white;
}
QPushButton[stato="ready"] {
    background-color: %(stato_ready)s;
    color: white;
}
QPushButton[stato="blocked"] {
    background-color: %(stato_blocked)s;
    color: white;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: %(campo)s;
    border: 1px solid %(bordo)s;
    border-radius: 4px;
    padding: 3px 6px;
    color: %(testo)s;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: %(accento)s;
}
/* Come QPushButton:disabled sopra, e per lo stesso motivo. Senza questa
   riga il testo di un campo spento resta acceso: Qt ingrigisce da se' solo
   cio' che disegna lo stile nativo, e queste regole lo hanno gia' portato
   sul motore del foglio. Misurato sul selettore delle maschere della
   pagina di cura del faceset: 52 pixel chiari nella tendina
   attiva, gli stessi 52 in quella disabilitata -- accanto a un bottone
   correttamente grigio, cioe' due controlli spenti che si leggono in due
   modi diversi nella stessa barra. */
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
    color: %(testo_lieve)s;
}
QComboBox QAbstractItemView {
    background: %(campo)s;
    border: 1px solid %(bordo)s;
    selection-background-color: %(accento)s;
}
/* Le voci disegnate dal delegato di theme.tendina() non ereditano il
   `padding` della regola QComboBox qui sopra: senza questa riga l'altezza di
   una riga del popup scende, mentre una QComboBox nuda -- che il foglio non
   riesce a stilizzare -- resta piena. Misurato due volte offscreen su venti
   voci, in condizioni tipografiche diverse e non registrate, con la stessa
   differenza di 8 px per riga (il padding 4px x 2): 26->18 px per riga,
   popup 522->362 px (2026-08-21, col delegato); 29->21 px, popup 580->420
   (misura precedente, stessa data). Non e' una preferenza: e' il ripristino
   della densita' che il popup ha quando lo disegna lo stile nativo. */
QComboBox QAbstractItemView::item {
    padding: 4px 6px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
}
/* Stessa regola di QPushButton:disabled e dei campi qui sopra, e per la
   stessa ragione: la riga sull'indicatore porta gia' l'interruttore sul
   motore del foglio di stile, che per cio' che non trova scritto usa i
   propri default -- e il default del testo e' quello acceso. Misurato
   sulla barra della finestra del volto allineato: 207 pixel chiari
   nell'etichetta "Landmarks" con l'interruttore attivo, gli stessi 207
   con l'interruttore spento, accanto a quattro pulsanti correttamente
   grigi nella stessa barra. */
QCheckBox:disabled {
    color: %(testo_lieve)s;
}
QListWidget::item {
    padding: 4px 6px;
}
/* Ogni proprieta' di queste due regole va dichiarata, anche quelle che
   "si vedono gia' giuste": basta una riga in `QTabBar::tab` perche' Qt
   smetta di far disegnare le schede allo stile nativo e le disegni col
   motore del foglio di stile, che per cio' che non trova scritto usa i
   propri valori di default -- e il default del bordo e' il **bianco**.
   Era la cornice chiara attorno alle schede, che non veniva da nessun
   colore di questo file: veniva dal non averlo detto. Misurato: col
   foglio, 143 pixel bianchi puri nella barra delle schede; senza, zero. */
QTabWidget::pane {
    border: 1px solid %(bordo)s;
    background: %(finestra)s;
    top: -1px;
}
QTabBar::tab {
    padding: 6px 12px;
    background: %(finestra)s;
    color: %(testo_lieve)s;
    border: 1px solid %(bordo)s;
    border-bottom: 1px solid %(bordo)s;
    margin-right: 2px;
}
QTabBar::tab:hover {
    color: %(testo)s;
}
QTabBar::tab:selected {
    background: %(lieve)s;
    color: %(testo)s;
    border-bottom: 2px solid %(accento)s;
}
QLabel[ruolo="sezione"] {
    font-size: %(sezione).1fpt;
    font-weight: bold;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: %(testo_lieve)s;
}
/* Il titolo della heatmap e' un QToolButton, non una QLabel: e' il comando
   che la collassa. Senza questa regola resterebbe al font di sistema
   accanto agli altri titoli di sezione -- e col bordo del proprio stile,
   in una fascia dove nient'altro ne ha uno. */
QToolButton[ruolo="sezione"] {
    /* La freccia di un QToolButton e' la sua ICONA, e l'icona non segue il
       font: resta a 16 px mentre il testo del ruolo va da 17 a 22 px fra
       `normal` e `xlarge` (misurato). Qui la si lega alla scala come tutto
       il resto -- il foglio si rigenera a ogni cambio di «View > Text
       size», quindi la freccia lo segue senza nessun changeEvent. */
    qproperty-iconSize: %(freccia)dpx %(freccia)dpx;
    font-size: %(sezione).1fpt;
    font-weight: bold;
    letter-spacing: 1px;
    color: %(testo_lieve)s;
    border: none;
    background: transparent;
    padding: 2px 4px;
}
QToolButton[ruolo="sezione"]:hover {
    color: %(testo)s;
}
QLabel[ruolo="minore"] {
    font-size: %(minore).1fpt;
    color: %(testo_lieve)s;
}
QLabel[ruolo="pastiglia"] {
    font-size: %(minore).1fpt;
    background: %(lieve)s;
    border-radius: 8px;
    padding: 2px 8px;
}
QWidget[ruolo="fascia-aiuto"] {
    background: %(lieve)s;
    border-left: 3px solid %(accento)s;
    padding: 2px 8px;
}
QWidget[ruolo="tessera-riquadro"] {
    background: %(lieve)s;
    border: 1px solid %(bordo)s;
    border-radius: 6px;
    padding: 4px 10px;
}
QLabel[ruolo="tessera"] {
    font-size: %(tessera).1fpt;
    font-weight: bold;
    color: %(testo)s;
}
QLabel[ruolo="aiuto-titolo"] {
    font-size: %(passo).1fpt;
    font-weight: bold;
}
QLabel[ruolo="aiuto-testo"] {
    font-size: %(base).1fpt;
    color: %(testo_lieve)s;
}
QToolTip {
    background: %(campo)s;
    border: 1px solid %(bordo)s;
    padding: 4px 6px;
    color: %(testo)s;
}
QTextEdit#console {
    font-family: Consolas, "DejaVu Sans Mono", monospace;
    font-size: %(console).1fpt;
}
"""


def stylesheet(scala=1.0):
    """The shell's QSS, generated from the palette and the scale.

    No literal colour: whoever reads this file must be able to change the
    palette in one place and see the whole window follow it. The accent is
    the palette's own because the XSeg editor uses the same one, and two
    different accents in the same package would stand out.
    """
    bordo = WINDOW.lighter(130).name()
    sfondo_lieve = WINDOW.lighter(115).name()
    testo_lieve = TEXT.darker(140).name()
    return _QSS % {
        "base": punti("base", scala), "tessera": punti("tessera", scala),
        "passo": punti("passo", scala), "sezione": punti("sezione", scala),
        # NOTA: "tessera" era gia' qui, calcolato e non usato da nessuna
        # regola prima di questa -- lo consuma QLabel[ruolo="tessera"] sopra.
        "minore": punti("minore", scala), "console": punti("console", scala),
        # I punti sono tipografici, i pixel di un'icona no: 4/3 e' il
        # rapporto a 96 dpi, ed e' l'unico modo per far crescere la freccia
        # insieme alla scritta che le sta accanto.
        "freccia": round(punti("sezione", scala) * 4 / 3),
        "testo": TEXT.name(), "testo_lieve": testo_lieve,
        "finestra": WINDOW.name(), "campo": BASE.name(),
        "accento": ACCENT.name(), "bordo": bordo, "lieve": sfondo_lieve,
        "stato_done": STATO_COLORE[STATE_DONE],
        "stato_ready": STATO_COLORE[STATE_READY],
        "stato_blocked": STATO_COLORE[STATE_BLOCKED],
    }


def apply_dark_theme(app, scala=1.0):
    """Style, palette and style sheet. The palette is the XSeg editor's own."""
    app.setStyle("Fusion")
    app.setPalette(dark_palette())
    app.setStyleSheet(stylesheet(scala))


def tendina(parent=None):
    """Una QComboBox il cui popup si illumina al passaggio del mouse.

    Misurato il 2026-08-21 su questa palette, muovendo il mouse sulla terza
    voce e leggendo il pixel sotto il cursore: con la regola
    `QComboBox { background: ... }` di questo stesso foglio il popup resta
    #191919 -- il fondo -- benche' currentIndex e il modello di selezione
    seguano il mouse. E' un difetto di PITTURA, non di logica: stilizzare
    il controllo chiuso porta la tendina sul motore del foglio e il popup
    interno smette di disegnare l'evidenziazione. Aggiungere
    `::item:hover` non la riaccende e in piu' fa saltare la selezione alla
    riga 0.

    Rimedi che accendono il pixel ce ne sono due, e **non sono
    equivalenti**. Sostituire la vista interna (`setView(QListView())`) lo
    accende ma **spegne il segnale `highlighted`**: con una vista propria la
    QComboBox non lo emette piu', ne' col mouse ne' con le frecce (misurato:
    `[]` contro `[0, 2]` di una tendina nuda, a parita' di gesto). Quel
    segnale e' l'unico motore dell'aiuto per voce di
    gui/fascia_aiuto.py::osserva -- la spiegazione del valore che si sta
    scegliendo, in ogni campo a scelta di ogni dialogo della shell: barattarla
    per un colore vorrebbe dire pagare un difetto con uno piu' grande.

    Il delegato tiene tutti e due. `QStyledItemDelegate` e' il rimedio
    classico al popup che ignora il foglio di stile: la vista resta quella
    della QComboBox -- quindi `highlighted` continua a partire -- ma le voci
    le disegna il motore del foglio, che l'hover lo dipinge (#378add sotto il
    cursore, contro #191919 senza). I tooltip per voce (`Qt.ToolTipRole`, che
    il selettore dei motori dell'estrazione usa per dire quali pesi mancano)
    sopravvivono a entrambe le strade.

    Sta qui e non in gui/forms.py perche' il rimedio appartiene alla regola
    che causa il difetto, che e' in questo file; e perche' ogni tendina
    della shell deve passare di qui, non solo quelle dei form -- lo impone
    una guardia in tests_gui/test_theme.py.
    """
    combo = QComboBox(parent)
    combo.setItemDelegate(QStyledItemDelegate(combo))
    return combo
