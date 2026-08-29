"""Le tessere di stato in cima alla scheda del training.

Le stesse coppie che compongono la riga di sempre sotto il grafico
(`status_line.valori_di_stato`) diventano qui cinque riquadri -- etichetta
piccola in maiuscoletto, valore grande -- perche' un numero letto da lontano
si vede a colpo d'occhio, una riga di dodici parole in corsivo grigio no.
Nessuna regola vive qui: `valori_di_stato` decide cosa mostrare e come
formattarlo, questo modulo si limita a disegnarlo.

`aggiorna` lavora **sul posto** finche' il numero di tessere non cambia:
un `setText` per valore cambiato, niente widget nuovi. Fino al 2026-08-29
ricostruiva i cinque riquadri a ogni chiamata -- e viene chiamata due volte
al secondo dagli eventi `iter` e a ogni pixel di trascinamento del cursore
dello storico -- ed era lo sfarfallio che l'utente vedeva in cima alla
scheda. L'invariante che la ricostruzione garantiva ("una tessera che
l'evento nuovo non porta piu' sparisce") regge lo stesso: quando cambia il
numero di coppie si ricostruisce tutto, e quando non cambia ogni riquadro
riceve chiave, etichetta e valore della coppia nuova nella sua posizione,
quindi niente resta a schermo col valore di prima.

Due cose tengono ferme le tessere fra un evento e l'altro. L'allungo in
coda al layout: senza, `QHBoxLayout` spartiva la larghezza in parti uguali
e la comparsa dell'ETA spostava **tutte** le tessere (misurato fino a 208
px). E la larghezza minima del valore, che cresce con il testo piu' largo
mai mostrato e non si restringe mai: le cifre che cambiano non muovono
niente. Viene dalle metriche del font a runtime, non da pixel fissi, cosi'
vale con qualunque font Windows o Linux scelgano.
"""
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from gui import testi
from gui.rimozione import svuota


class TessereStato(QWidget):
    """Un riquadro per coppia (etichetta, valore).

    `aggiorna` e `valori` sono l'unica API pubblica: chi la usa passa le
    coppie di `status_line.valori_di_stato` cosi' come sono, senza sapere
    (ne' dover sapere) come vengono disegnate.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addStretch(1)
        self._riquadri = []

    def aggiorna(self, coppie):
        """Porta le tessere alle coppie date: sul posto se sono tante quante
        prima, ricostruite se no (vedi il docstring del modulo)."""
        coppie = list(coppie)
        if len(coppie) != len(self._riquadri):
            self._ricostruisci(coppie)
            return
        for riquadro, (etichetta, valore) in zip(self._riquadri, coppie):
            self._imposta(riquadro, etichetta, valore)

    def valori(self):
        """Le coppie mostrate adesso, lette dai riquadri veri.

        Non c'e' una copia delle coppie tenuta a parte apposta: se
        `aggiorna` un giorno smettesse di disegnare, `valori()` se ne
        accorgerebbe da sola invece di continuare a raccontare cio' che le
        e' stato passato -- e' il layout, non la memoria, la fonte.
        """
        coppie = []
        for indice in range(self._layout.count()):
            riquadro = self._layout.itemAt(indice).widget()
            if riquadro is None:
                continue
            colonna = riquadro.layout()
            valore = colonna.itemAt(1).widget().text()
            coppie.append((riquadro.property("chiave"), valore))
        return coppie

    def _ricostruisci(self, coppie):
        #Lo smontaggio passa da `gui.rimozione`, non da un `setParent(None)`
        #scritto qui: staccare un widget senza nasconderlo prima lo lascia
        #diventare una finestra di primo livello che Qt ri-mostra da sola.
        svuota(self._layout)
        self._riquadri = []
        for etichetta, valore in coppie:
            riquadro = self._tessera()
            self._imposta(riquadro, etichetta, valore)
            self._riquadri.append(riquadro)
            self._layout.addWidget(riquadro)
        self._layout.addStretch(1)

    def _tessera(self):
        riquadro = QWidget()
        riquadro.setProperty("ruolo", "tessera-riquadro")
        colonna = QVBoxLayout(riquadro)
        nome = QLabel()
        nome.setProperty("ruolo", "sezione")
        numero = QLabel()
        numero.setProperty("ruolo", "tessera")
        colonna.addWidget(nome)
        colonna.addWidget(numero)
        return riquadro

    def _imposta(self, riquadro, etichetta, valore):
        colonna = riquadro.layout()
        nome, numero = colonna.itemAt(0).widget(), colonna.itemAt(1).widget()
        if riquadro.property("chiave") != etichetta:
            #La chiave resta sul riquadro -- non sull'etichetta visibile,
            #che e' gia' tradotta da `testi.tile_label` e non si potrebbe
            #tornare indietro senza ambiguita' -- cosi' anche `valori()` la
            #legge dal widget vero, non da una seconda copia.
            riquadro.setProperty("chiave", etichetta)
            nome.setText(testi.tile_label(etichetta))
            #Sul riquadro, non sulle due etichette dentro: cosi' il
            #suggerimento compare ovunque il mouse si fermi sulla tessera.
            riquadro.setToolTip(testi.tile_tip(etichetta))
        if numero.text() != valore:
            numero.setText(valore)
            larghezza = numero.fontMetrics().horizontalAdvance(valore)
            if larghezza > numero.minimumWidth():
                numero.setMinimumWidth(larghezza)
