"""Il riquadro che spiega cio' che si sta guardando.

Il tooltip e' per chi cerca; questa fascia e' per chi non sa di dover
cercare. Sta in fondo al form, e non si muove: altezza fissa (non solo
minima -- una riga di testo in piu' non deve spostare i campi sotto), testo
che va a capo entro un numero fisso di righe e viene elissato con "…" se
sfora quella soglia, cosi' il form non balla mentre il mouse attraversa i
campi ne' quando il testo piu' lungo del catalogo (il `help` di
`torch_compile`, sei-sette righe piene) finisce sotto il cursore.

Tre eventi la alimentano, tutti dallo stesso filtro: il mouse che entra, il
focus che arriva da tastiera, e la voce evidenziata dentro una tendina
aperta -- che e' il momento esatto in cui la spiegazione di **quel** valore
serve, mentre lo si sta scegliendo. Leave e FocusOut la riportano a riposo,
ma solo se chi esce e' ancora il proprietario di cio' che e' a schermo: se
il focus e' gia' passato altrove (tastiera) mentre il mouse indugia sul
campo di prima, il suo Leave tardivo non deve cancellare l'aiuto del campo
nuovo.
"""
from PyQt5.QtCore import QEvent, QObject, Qt
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

from gui import testi

# Quante righe di testo (oltre al titolo) entrano nella fascia. Il
# `choice_help` piu' lungo del catalogo dura due frasi; il piu' lungo in
# assoluto e' un `help` di checkbox (torch_compile, ~700 caratteri): con
# quattro righe la prima entra per intero, la seconda no, ed e' esattamente
# il caso che l'ellissi deve gestire senza far crescere il riquadro.
_RIGHE_TESTO = 4


def _spezza_e_ellissa(metrica, testo, larghezza, righe_massime):
    """`testo` spezzato per parola entro `larghezza` pixel, al massimo
    `righe_massime` righe; l'ultima porta un ellissi se resta fuori
    qualcosa. `larghezza <= 0` (widget non ancora disposto da un layout
    reale, es. un test che non lo mostra mai) rinuncia ad andare a capo e
    ritorna il testo intero: e' la scelta sicura, mai quella che spezza le
    parole a caso.
    """
    if larghezza <= 0 or righe_massime <= 0:
        return testo
    parole = testo.split()
    righe, corrente, indice = [], "", 0
    while indice < len(parole) and len(righe) < righe_massime:
        parola = parole[indice]
        prova = (corrente + " " + parola).strip()
        if corrente and metrica.horizontalAdvance(prova) > larghezza:
            righe.append(corrente)
            corrente = ""
            continue
        corrente = prova
        indice += 1
    troncato = indice < len(parole)
    if corrente:
        righe.append(corrente)
    if not troncato:
        return "\n".join(righe)
    ultima = righe[-1] if righe else ""
    while ultima and metrica.horizontalAdvance(ultima + "…") > larghezza:
        ultima = ultima.rsplit(" ", 1)[0] if " " in ultima else ultima[:-1]
    righe[-1:] = [(ultima + "…") if ultima else "…"]
    return "\n".join(righe)


class FasciaAiuto(QWidget):
    """Titolo + testo, altezza fissa. `mostra`/`riposo` sono l'unica API
    pubblica: chi la usa non sa (ne' deve sapere) che il testo mostrato e'
    elissato rispetto a quello passato -- `testo_corrente()` ritorna cio'
    che e' davvero a schermo, coerente con ogni altra `_corrente()` del
    pacchetto.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._riposo = ""
        self._proprietario = None
        self._titolo_intero = ""
        self._testo_intero = ""
        colonna = QVBoxLayout(self)
        self._titolo = QLabel()
        self._titolo.setProperty("ruolo", "aiuto-titolo")
        self._testo = QLabel()
        self._testo.setWordWrap(True)
        self._testo.setProperty("ruolo", "aiuto-testo")
        colonna.addWidget(self._titolo)
        colonna.addWidget(self._testo)
        self.setProperty("ruolo", "fascia-aiuto")
        #Senza questo attributo il foglio di stile non dipinge niente qui:
        #`background` e `border-left` di una regola QSS valgono su un QWidget
        #**diretto** ma non su una sua sottoclasse, che Qt lascia trasparente
        #a meno che non le si dica esplicitamente di farsi dipingere dallo
        #stile. E' il motivo per cui le tessere di stato -- QWidget() nudi --
        #si vedevano e questa fascia no: stessa forma di regola, esito
        #opposto, e nel foglio di stile non c'e' niente che lo lasci
        #sospettare. Misurato: dentro la fascia il pixel era (53,53,53), lo
        #stesso sfondo della finestra, e la banda d'accento a sinistra non
        #esisteva.
        self.setAttribute(Qt.WA_StyledBackground, True)
        #Fissa, non minima: il capitolato originale usava setMinimumHeight,
        #ma una riga di testo in piu' faceva crescere la fascia e con lei
        #ogni campo sotto -- esattamente il "ballare" che questo widget
        #deve evitare. Il budget viene dalla metrica vera dei due font (il
        #titolo su una riga, il testo su _RIGHE_TESTO), non da un fattore
        #inventato: cosi' segue davvero cio' che verra' disegnato, incluso
        #il caso in cui il foglio di stile dell'applicazione cambi la
        #dimensione del carattere prima che questo widget nasca.
        #`ensurePolished()` e' cio' che rende vera quell'ultima frase: senza,
        #`fontMetrics()` qui misura il font ereditato prima che il foglio di
        #stile applichi le regole per ruolo ("aiuto-titolo"/"aiuto-testo"),
        #un font piu' piccolo di quello vero -- il divario cresce con la
        #scala tipografica e a "xlarge" il budget non basta piu': il titolo
        #si sovrappone al testo e l'ultima riga viene tagliata invece di
        #ricevere l'ellissi.
        self._adegua_altezza()

    def _adegua_altezza(self):
        """L'altezza fissa, ricalcolata dai font veri di adesso.

        Non solo alla nascita: la scala tipografica si cambia da `View ›
        Text size` mentre un passo e' gia' aperto -- ed e' il percorso
        naturale, perche' e' proprio leggendo un testo troppo piccolo che
        viene voglia di ingrandirlo. Con il budget calcolato una volta sola
        una fascia nata a "normal" restava alta come allora e a "xlarge" il
        testo tornava tagliato, cioe' esattamente il difetto che l'altezza
        fissa esiste per non avere.

        `ensurePolished()` va rifatto ogni volta per la stessa ragione per
        cui serviva la prima: senza, `fontMetrics()` misura il font di
        prima che il foglio di stile nuovo applichi le regole per ruolo.
        """
        self._titolo.ensurePolished()
        self._testo.ensurePolished()
        margini = self.layout().contentsMargins()
        self.setFixedHeight(
            self._titolo.fontMetrics().height()
            + self.layout().spacing()      # fra titolo e testo: lo disegna il layout
            + self._testo.fontMetrics().lineSpacing() * _RIGHE_TESTO
            + margini.top() + margini.bottom())

    #override
    def changeEvent(self, evento):
        """Il font o lo stile sono cambiati: il budget non vale piu'.

        `setFixedHeight` non genera nessuno di questi due eventi, quindi
        non c'e' ricorsione da temere; `_ridisegna` segue perche' anche
        l'ellissi e' calcolata sulla metrica vecchia.
        """
        super().changeEvent(evento)
        if evento.type() in (QEvent.FontChange, QEvent.StyleChange):
            self._adegua_altezza()
            self._ridisegna()

    def mostra(self, titolo, testo):
        self._titolo_intero, self._testo_intero = titolo, testo
        self._ridisegna()

    def riposo(self, testo=None):
        """Torna al testo di riposo. Con `testo` dato, lo sostituisce
        prima -- e' cosi' che `StepForm` imposta il messaggio iniziale."""
        if testo is not None:
            self._riposo = testo
        self.mostra("", self._riposo)

    def titolo_corrente(self):
        return self._titolo.text()

    def testo_corrente(self):
        return self._testo.text()

    def resizeEvent(self, evento):
        super().resizeEvent(evento)
        self._ridisegna()

    def _larghezza_utile(self):
        margini = self.layout().contentsMargins()
        return self.width() - margini.left() - margini.right()

    def _ridisegna(self):
        larghezza = self._larghezza_utile()
        metrica_titolo = QFontMetrics(self._titolo.font())
        self._titolo.setText(
            metrica_titolo.elidedText(self._titolo_intero, Qt.ElideRight, larghezza)
            if larghezza > 0 else self._titolo_intero)
        self._titolo.setVisible(bool(self._titolo_intero))
        metrica_testo = QFontMetrics(self._testo.font())
        self._testo.setText(_spezza_e_ellissa(metrica_testo, self._testo_intero,
                                               larghezza, _RIGHE_TESTO))


class _Filtro(QObject):
    """Un filtro per controllo: tiene il testo e non tocca l'evento.

    `scarto` e' l'offset fra l'indice del combo e quello di `choice_help`:
    un campo senza default ha una voce vuota in testa, e senza questo la
    spiegazione sarebbe quella del valore precedente -- il modo piu' rapido
    per far mentire proprio la superficie che spiega.

    Diventa figlio del widget che osserva (vedi `osserva`): senza un
    riferimento vivo Qt lo raccoglierebbe e l'aggancio smetterebbe di
    funzionare **senza dirlo**, il modo peggiore in cui questo codice possa
    rompersi -- il filtro esisterebbe ancora sulla carta ma non riceverebbe
    piu' nessun evento.
    """

    def __init__(self, fascia, titolo, testo, per_voce=(), combo=None,
                 scarto=0, parent=None):
        super().__init__(parent)
        self._fascia, self._titolo, self._testo = fascia, titolo, testo
        self._per_voce = tuple(per_voce)
        self._combo, self._scarto = combo, scarto

    def eventFilter(self, oggetto, evento):
        tipo = evento.type()
        if tipo in (QEvent.Enter, QEvent.FocusIn):
            self._fascia._proprietario = self
            self._fascia.mostra(self._titolo, self._testo)
        elif tipo in (QEvent.Leave, QEvent.FocusOut):
            #Solo se il proprietario e' ancora questo campo: un Leave del
            #mouse arrivato dopo che il focus e' gia' passato altrove (via
            #tastiera) non deve cancellare l'aiuto del campo nuovo.
            if self._fascia._proprietario is self:
                self._fascia._proprietario = None
                self._fascia.riposo()
        return False        # mai consumato: il controllo deve funzionare

    def su_voce(self, indice):
        posizione = indice - self._scarto
        if not (0 <= posizione < len(self._per_voce)):
            return
        etichetta = self._combo.itemText(indice) if self._combo is not None else ""
        self._fascia._proprietario = self
        self._fascia.mostra(testi.help_choice_title(self._titolo, etichetta),
                             self._per_voce[posizione])


def osserva(widget, fascia, titolo, testo, per_voce=(), scarto=0, anche=()):
    """Aggancia un controllo alla fascia. Torna il filtro, gia' installato.

    Il filtro diventa figlio del widget: senza un riferimento vivo Qt lo
    raccoglie e l'aggancio smette di funzionare **senza dirlo**, che e' il
    modo peggiore in cui questo codice possa rompersi.

    `anche` sono le altre superfici della stessa riga -- il nome
    dell'opzione a sinistra, il contenitore che tiene controllo e
    pastiglia. Ricevono lo **stesso** filtro, non uno per ciascuna, e la
    differenza conta: `_proprietario` e' un oggetto solo, quindi il Leave di
    una superficie e l'Enter della sua vicina non si contendono la fascia --
    con due filtri distinti l'uscita dal nome cancellerebbe l'aiuto che
    l'ingresso nel campo ha appena messo. Passare col mouse sul nome
    dell'opzione e' il gesto naturale di chi cerca di capire cosa sia:
    prima rispondeva solo la casella dove si scrive il valore.
    """
    combo = widget if isinstance(widget, QComboBox) else None
    filtro = _Filtro(fascia, titolo, testo, per_voce, combo, scarto, parent=widget)
    widget.installEventFilter(filtro)
    for altro in anche:
        if altro is not None:
            altro.installEventFilter(filtro)
    if combo is not None:
        combo.highlighted.connect(filtro.su_voce)
    return filtro
