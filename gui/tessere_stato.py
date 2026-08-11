"""Le tessere di stato in cima alla scheda del training.

Le stesse coppie che compongono la riga di sempre sotto il grafico
(`status_line.valori_di_stato`) diventano qui cinque riquadri -- etichetta
piccola in maiuscoletto, valore grande -- perche' un numero letto da lontano
si vede a colpo d'occhio, una riga di dodici parole in corsivo grigio no.
Nessuna regola vive qui: `valori_di_stato` decide cosa mostrare e come
formattarlo, questo modulo si limita a disegnarlo.

Niente Qt astruso: un `QHBoxLayout` che si ricostruisce per intero a ogni
`aggiorna`. Sono al massimo cinque tessere, non e' un costo -- ed e' l'unico
modo per non lasciare a schermo il valore di una tessera che l'evento nuovo
non porta piu' (l'obiettivo raggiunto fa sparire l'ETA, un ciclo senza VRAM
annunciata fa sparire la sua tessera): "aggiorno solo quelle che ci sono"
lascerebbe l'ultima buona congelata, ed e' proprio il difetto che questo
modulo esiste per non avere.
"""
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from gui import testi
from gui.rimozione import svuota


class TessereStato(QWidget):
    """Un riquadro per coppia (etichetta, valore), ricostruiti a ogni giro.

    `aggiorna` e `valori` sono l'unica API pubblica: chi la usa passa le
    coppie di `status_line.valori_di_stato` cosi' come sono, senza sapere
    (ne' dover sapere) come vengono disegnate.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

    def aggiorna(self, coppie):
        """Ricostruisce le tessere dalle coppie date.

        Ricostruire invece di aggiornare sul posto e' la scelta che tiene
        vero l'invariante: se una coppia di prima non c'e' piu' in quelle
        nuove, la sua tessera sparisce -- non resta a schermo col valore
        vecchio, che sarebbe una bugia silenziosa (un training che ha
        raggiunto l'obiettivo, con l'ETA di un minuto fa ancora li').

        Lo smontaggio passa da `gui.rimozione`, non da un `setParent(None)`
        scritto qui: staccare un widget senza nasconderlo prima lo lascia
        diventare una finestra di primo livello che Qt ri-mostra da sola --
        ed e' proprio questo il punto del programma dove il difetto si
        vedeva di piu', perche' e' quello che ricostruisce piu' spesso.
        """
        svuota(self._layout)
        for etichetta, valore in coppie:
            self._layout.addWidget(self._tessera(etichetta, valore))

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

    def _tessera(self, etichetta, valore):
        riquadro = QWidget()
        riquadro.setProperty("ruolo", "tessera-riquadro")
        #La chiave resta sul riquadro -- non sull'etichetta visibile, che
        #e' gia' tradotta da `testi.tile_label` e non si potrebbe tornare
        #indietro senza ambiguita' -- cosi' anche `valori()` la legge dal
        #widget vero, non da una seconda copia.
        riquadro.setProperty("chiave", etichetta)
        #Sul riquadro, non sulle due etichette dentro: cosi' il suggerimento
        #compare ovunque il mouse si fermi sulla tessera, nome o numero che
        #sia. Un numero grande senza una frase che dica cos'e' e' proprio
        #cio' che questa scheda non deve avere.
        riquadro.setToolTip(testi.tile_tip(etichetta))
        colonna = QVBoxLayout(riquadro)
        nome = QLabel(testi.tile_label(etichetta))
        nome.setProperty("ruolo", "sezione")
        numero = QLabel(valore)
        numero.setProperty("ruolo", "tessera")
        colonna.addWidget(nome)
        colonna.addWidget(numero)
        return riquadro
