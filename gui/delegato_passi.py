"""Il delegato che disegna ogni voce della lista dei passi.

Il testo dell'elemento (`item.text()`) resta il nome del passo: molto altro
codice e molti test lo leggono, e non devono accorgersi di niente. Questo
delegato disegna sopra quel modello due righe in piu' -- il nome e sotto il
sommario del passo (`StepDef.summary`) -- e un badge di stato a destra,
letti da due ruoli propri del modello (`RUOLO_SOMMARIO`, `RUOLO_STATO`), mai
da `item.text()`.

`paint` e' un metodo virtuale chiamato da Qt: un'eccezione la' dentro non
risale a nessun chiamante, PyQt5 la trasforma in un `qFatal` e il processo
se ne va con dentro ogni finestra aperta -- lo stesso rischio gia' descritto
in `gui/loss_plot.py` e in `gui/numeri.py`. I dati che arrivano nei due
ruoli non sono garantiti: nascono dal catalogo o da uno stato calcolato
altrove, e un sommario assente, lunghissimo o di tipo sbagliato non deve
mai fermare il disegno. La scelta e' validare *prima* di disegnare, non
racchiudere il disegno in una rete -- la stessa scelta fatta in
`gui/preview_grid.py` e in `gui/loss_plot.py`: ogni misura usata per
costruire un rettangolo e' gia' non negativa quando arriva a `drawText`/
`drawRoundedRect`, quindi non c'e' combinazione di dati storti che sollevi.
"""
from PyQt5.QtCore import QRect, QSize, Qt
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QPalette
from PyQt5.QtWidgets import QStyle, QStyledItemDelegate

from gui import testi
from gui.theme import STATO_COLORE, TEXT, punti, scala_di

RUOLO_SOMMARIO = Qt.UserRole + 1
RUOLO_STATO = Qt.UserRole + 2

_MARGINE = 6
_INTERLINEA = 2
_PADDING_BADGE = 6

# Gli stessi colori della pillola di stage (gui.theme.STATO_COLORE): un passo
# "done"/"ready"/"blocked" porta lo stesso colore dello stage a cui appartiene.
_COLORE_STATO = {chiave: QColor(valore) for chiave, valore in STATO_COLORE.items()}


def _testo_sicuro(valore):
    """`valore` se e' una stringa, altrimenti stringa vuota.

    I due ruoli viaggiano per `QVariant`: un mittente storto (un catalogo
    corrotto, uno stato calcolato male) puo' mandare un intero, una lista o
    None invece della stringa attesa, e il disegno non deve fermarsi per
    questo -- si tratta come "nessun testo", non come un errore.
    """
    return valore if isinstance(valore, str) else ""


def colore_e_etichetta_stato(stato):
    """Il colore del badge e la sua etichetta visibile, o (None, "") se
    `stato` non e' uno dei tre STATE_* riconosciuti -- assente, storto, o
    di un tipo che non e' nemmeno una stringa (un dizionario indicizzato da
    un valore non hashable solleverebbe altrimenti)."""
    stato = _testo_sicuro(stato)
    colore = _COLORE_STATO.get(stato)
    if colore is None:
        return None, ""
    return colore, testi.step_badge_label(stato)


class DelegatoPassi(QStyledItemDelegate):
    """Nome + sommario su due righe, badge di stato a destra.

    Lo sfondo di selezione/hover si disegna a mano (`_disegna_sfondo`), non
    chiamando `QStyledItemDelegate.paint` -- il tentativo naturale, con
    `displayText` svuotato per lasciargli disegnare solo il fondo, e' andato
    in **segmentation fault** durante una `QListWidget` vera a schermo, col
    foglio di stile scuro applicato (mai nei test sintetici su un `QImage`,
    solo con lo stile reale attivo: `QStyleSheetStyle` avvolge `CE_ItemViewItem`
    e la combinazione coi due metodi virtuali non regge). Trovato riproducendo
    per davvero, non nei test isolati -- che infatti restano verdi anche per
    la versione che si e' rivelata insicura. `fillRect` con un colore della
    stessa palette non passa da quel percorso.
    """

    def _disegna_sfondo(self, pittore, opzione):
        """Il fondo della riga: pieno se selezionata, sfumato se sotto il
        mouse, altrimenti nulla -- il resto lo ha gia' dipinto la vista."""
        palette = opzione.palette
        if opzione.state & QStyle.State_Selected:
            pittore.fillRect(opzione.rect, palette.color(QPalette.Highlight))
        elif opzione.state & QStyle.State_MouseOver:
            sfumato = QColor(palette.color(QPalette.Highlight))
            sfumato.setAlpha(60)
            pittore.fillRect(opzione.rect, sfumato)

    def _con_ruolo(self, opzione, ruolo):
        """Il font dell'opzione riportato a `ruolo`, alla scala di adesso.

        La scala si ricava dal font che Qt passa nell'opzione -- che il
        foglio di stile ha gia' scalato -- invece di rileggerla da
        `gui.preferenze`: qui non c'e' un secondo valore da tenere
        allineato, quindi non c'e' modo che il nome del passo resti alla
        misura di prima mentre l'etichetta accanto e' gia' cambiata. E'
        esattamente cio' che succedeva: `sizeHintForRow(0)` valeva 52 px sia
        a 1.0 sia a 1.3, e `View › Text size` non arrivava a questa lista.
        """
        font = QFont(opzione.font)
        font.setPointSizeF(punti(ruolo, scala_di(opzione.font)))
        return font

    def _font_nome(self, opzione):
        return self._con_ruolo(opzione, "passo")

    def _font_sommario(self, opzione):
        return self._con_ruolo(opzione, "minore")

    def _font_badge(self, opzione):
        font = self._con_ruolo(opzione, "minore")
        font.setBold(True)
        return font

    def _larghezza_vista(self):
        """La larghezza vera del viewport della lista che questo delegato
        serve, o None se non e' figlio di una vista vera -- i test lo
        costruiscono cosi', a volte senza nessun parent.

        Interrogare `opzione.rect` da solo non basta: Qt lo rialimenta con
        l'ultima risposta di `sizeHint` per la stessa riga, quindi una
        larghezza "naturale" piu' larga del viewport (calcolata per
        contenere il sommario intero) resta un punto fisso che non si
        restringe mai da solo -- e' cosi' che il sommario finiva piu'
        largo del pannello, con una barra di scorrimento orizzontale a
        nasconderne la coda invece di elissarla sul posto.
        """
        vista = self.parent()
        viewport = getattr(vista, "viewport", None)
        if viewport is None:
            return None
        larghezza = viewport().width()
        return larghezza if larghezza > 0 else None

    def sizeHint(self, opzione, indice):
        nome = _testo_sicuro(indice.data(Qt.DisplayRole))
        sommario = _testo_sicuro(indice.data(RUOLO_SOMMARIO))
        metriche_nome = QFontMetrics(self._font_nome(opzione))
        metriche_sommario = QFontMetrics(self._font_sommario(opzione))
        altezza = (metriche_nome.height() + metriche_sommario.height()
                   + _INTERLINEA + 2 * _MARGINE)
        larghezza_vista = self._larghezza_vista()
        if larghezza_vista is not None:
            # Mai piu' largo della vista: e' cosi' che ogni riga resta
            # nella stessa larghezza in cui verra' davvero disegnata, e
            # l'ellissi di `_disegna_riga` (che elide dentro quella stessa
            # larghezza) e' l'unica cosa che puo' tagliare il sommario.
            larghezza = larghezza_vista
        else:
            larghezza = (max(metriche_nome.horizontalAdvance(nome),
                              metriche_sommario.horizontalAdvance(sommario))
                         + 2 * _MARGINE)
        return QSize(larghezza, altezza)

    def paint(self, pittore, opzione, indice):
        # save()/restore() sono igiene dello stato del QPainter fra una riga
        # e la prossima (penna/pennello/font cambiano qui sotto), non una
        # rete per un'eccezione: nessun except in questo metodo, per la
        # stessa scelta strutturale del docstring del modulo.
        pittore.save()
        try:
            self._disegna_riga(pittore, opzione, indice)
        finally:
            pittore.restore()

    def _disegna_riga(self, pittore, opzione, indice):
        self._disegna_sfondo(pittore, opzione)

        area = opzione.rect.adjusted(_MARGINE, _MARGINE, -_MARGINE, -_MARGINE)
        if area.width() <= 0 or area.height() <= 0:
            return

        nome = _testo_sicuro(indice.data(Qt.DisplayRole))
        sommario = _testo_sicuro(indice.data(RUOLO_SOMMARIO))
        colore_badge, etichetta_badge = colore_e_etichetta_stato(indice.data(RUOLO_STATO))

        area_testo = area
        if colore_badge is not None and etichetta_badge:
            area_testo = self._disegna_badge(pittore, area, colore_badge, etichetta_badge, opzione)

        selezionata = bool(opzione.state & QStyle.State_Selected)
        palette = opzione.palette
        colore_nome = palette.color(QPalette.HighlightedText if selezionata else QPalette.Text)
        colore_sommario = colore_nome if selezionata else TEXT.darker(140)

        larghezza_testo = max(0, area_testo.width())
        altezza_nome = QFontMetrics(self._font_nome(opzione)).height()
        rect_nome = QRect(area_testo.left(), area_testo.top(), larghezza_testo, altezza_nome)
        alto_sommario = max(0, area_testo.bottom() - rect_nome.bottom() - _INTERLINEA)
        rect_sommario = QRect(area_testo.left(), rect_nome.bottom() + _INTERLINEA,
                               larghezza_testo, alto_sommario)

        if rect_nome.width() > 0 and rect_nome.height() > 0 and nome:
            font_nome = self._font_nome(opzione)
            metriche = QFontMetrics(font_nome)
            pittore.setFont(font_nome)
            pittore.setPen(colore_nome)
            pittore.drawText(rect_nome, Qt.AlignLeft | Qt.AlignVCenter,
                              metriche.elidedText(nome, Qt.ElideRight, rect_nome.width()))

        if rect_sommario.width() > 0 and rect_sommario.height() > 0 and sommario:
            font_sommario = self._font_sommario(opzione)
            metriche = QFontMetrics(font_sommario)
            pittore.setFont(font_sommario)
            pittore.setPen(colore_sommario)
            pittore.drawText(rect_sommario, Qt.AlignLeft | Qt.AlignVCenter,
                              metriche.elidedText(sommario, Qt.ElideRight, rect_sommario.width()))

    def _disegna_badge(self, pittore, area, colore, etichetta, opzione):
        """Disegna la pastiglia di stato sul bordo destro di `area` e torna
        cio' che resta a sinistra per nome e sommario.

        Se `area` e' piu' stretta di quanto la pastiglia vorrebbe occupare,
        la pastiglia si restringe fino a riempirla -- mai un rettangolo piu'
        largo del contenitore che lo ospita, e mai uno di larghezza negativa.
        """
        font = self._font_badge(opzione)
        metriche = QFontMetrics(font)
        larghezza_naturale = metriche.horizontalAdvance(etichetta) + 2 * _PADDING_BADGE
        larghezza_badge = max(0, min(area.width(), larghezza_naturale))
        altezza_badge = max(0, min(area.height(), metriche.height() + _PADDING_BADGE))

        rect_badge = QRect(0, 0, larghezza_badge, altezza_badge)
        rect_badge.moveCenter(area.center())
        rect_badge.moveRight(area.right())

        if larghezza_badge > 0 and altezza_badge > 0:
            pittore.setPen(Qt.NoPen)
            pittore.setBrush(colore)
            raggio = altezza_badge / 2
            pittore.drawRoundedRect(rect_badge, raggio, raggio)

            pittore.setFont(font)
            pittore.setPen(QColor(Qt.white))
            pittore.drawText(rect_badge, Qt.AlignCenter, etichetta)

        spazio = larghezza_badge + _MARGINE if larghezza_badge > 0 else 0
        return area.adjusted(0, 0, -spazio, 0)
