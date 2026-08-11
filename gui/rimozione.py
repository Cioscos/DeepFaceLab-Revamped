"""Togliere un widget da un layout senza lasciarlo a schermo.

`setParent(None)` sembra bastare e non basta, e il modo in cui non basta e'
silenzioso. Qt marca il widget come nascosto, ma **azzera** anche
`WA_WState_ExplicitShowHide` -- cioe' "nascosto perche' e' capitato, non
perche' qualcuno l'abbia chiesto". Un widget senza genitore e' una finestra
di primo livello, e al primo giro di ciclo Qt mostra quelle che nessuno ha
nascosto per scelta: l'orfano riappare come una finestrella a se' stante,
col suo contenuto dentro, in mezzo allo schermo. `deleteLater()` non copre
il buco: la cancellazione differita non viene smaltita a ogni giro, e nel
frattempo la finestra c'e' gia'.

Misurato prima della correzione, offscreen: cento ricostruzioni di una fila
di cinque riquadri -- il gesto di trascinare il cursore del tempo ne fa una
per evento -- lasciavano **cinquecento** widget di primo livello, tutti
visibili dopo un solo `processEvents()`. Con `hide()` prima del distacco:
zero. Sono le finestrelle che l'utente vedeva comparire trascinando lo
slider, e ogni tanto da sole, con dentro le tessere di stato.

L'ordine dei tre passi non e' negoziabile: `hide()` **prima** di
`setParent(None)`, o l'attributo che regge l'invariante viene azzerato
subito dopo essere stato messo.
"""


def stacca(widget):
    """Nasconde, distacca e programma la distruzione di `widget`.

    `None` e' un caso legittimo, non un errore: `QLayout.takeAt(...).widget()`
    torna `None` per gli elementi che widget non sono (spaziatori,
    sotto-layout), e chi svuota un layout non deve doverlo sapere.
    """
    if widget is None:
        return
    widget.hide()
    widget.setParent(None)
    widget.deleteLater()


def svuota(layout):
    """Toglie ogni elemento da `layout`, senza lasciare finestre in giro."""
    while layout.count():
        stacca(layout.takeAt(0).widget())
