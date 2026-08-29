"""Lo storico delle anteprime che DeepFaceLab scrive gia' su disco.

`<model_dir>/<nome modello>_history/<nome anteprima>/<iterazione a 7 cifre>.jpg`,
uno ogni 10 iterazioni quando l'opzione write_preview_history del modello e'
attiva. Non lo scrive la GUI e non lo cambia: la cartella resta identica che
la corsa sia partita da qui o da un terminale, cosi' chi ci monta un
time-lapse trova immagini omogenee.

Il prezzo di quella scelta e' questa costante: ogni immagine porta la
striscia del grafico incollata sopra, alta esattamente 100 px, e va tolta
perche' nel pannello il grafico c'e' gia', vivo e accanto.
"""
from pathlib import Path

from PyQt5.QtGui import QImage

from gui.numeri import iterazione_utilizzabile

#ModelBase.get_loss_history_preview costruisce la striscia con lh_height=100
#e PreviewHistoryWriter la concatena sopra l'anteprima. E' una duplicazione
#voluta: gui/ non importa models/. La guardia in tests_gui/ la sorveglia.
FASCIA_GRAFICO = 100


class StoricoAnteprime(object):
    def __init__(self, model_dir, model_name):
        self.base = Path(model_dir) / ("%s_history" % model_name)
        #Elenchi tenuti finche' la cartella non cambia: `mtime` della
        #cartella, che sia Windows sia Linux aggiornano quando una voce
        #entra o esce. Rileggere a ogni evento `preview` costava 46 ms a
        #20 000 scatti (misurato 2026-08-29), sul thread dell'interfaccia.
        self._cache = {}          # cartella -> (mtime_ns, risultato)

    def disponibile(self):
        return self.base.is_dir()

    def _memorizzata(self, cartella, calcola):
        try:
            mtime = cartella.stat().st_mtime_ns
        except OSError:
            self._cache.pop(cartella, None)
            return []
        voce = self._cache.get(cartella)
        if voce is not None and voce[0] == mtime:
            return list(voce[1])
        risultato = calcola()
        self._cache[cartella] = (mtime, risultato)
        return list(risultato)

    def anteprime(self):
        if not self.disponibile():
            return []
        return self._memorizzata(
            self.base, lambda: sorted(p.name for p in self.base.iterdir() if p.is_dir()))

    def iterazioni(self, nome):
        """Le iterazioni disponibili per un'anteprima, in ordine crescente.

        Rilette a ogni chiamata: il training continua mentre il pannello e'
        aperto, e uno storico congelato all'apertura invecchierebbe subito.

        E' la terza porta da cui entra un'iterazione, dopo il canale eventi e
        la colonna `iter` del CSV, e passa dalla stessa regola delle altre
        due: `int()` da solo accetta il segno e qualunque grandezza, e da
        qui il numero va dritto nel cursore, nella finestra del grafico e
        nell'indice di colonna. Un nome di file e' un dato esterno come gli
        altri -- chiunque puo' posare un `-5.jpg` in quella cartella.
        """
        cartella = self.base / nome
        if not cartella.is_dir():
            return []
        return self._memorizzata(cartella, lambda: self._iterazioni_da_disco(cartella))

    def _iterazioni_da_disco(self, cartella):
        numeri = []
        for p in cartella.iterdir():
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            try:
                iterazione = int(p.stem)
            except ValueError:
                continue     # _last.jpg: un duplicato dell'ultimo scatto
            if iterazione_utilizzabile(iterazione):
                numeri.append(iterazione)
        return sorted(numeri)

    def immagine(self, nome, iterazione):
        """L'anteprima a quell'iterazione, senza la fascia del grafico."""
        for suffisso in (".jpg", ".jpeg", ".png"):
            percorso = self.base / nome / ("%07d%s" % (iterazione, suffisso))
            if percorso.exists():
                break
        else:
            return None
        img = QImage(str(percorso))
        if img.isNull():
            return None      # troncata o corrotta: la salta chi ci sta sopra
        if img.height() <= FASCIA_GRAFICO:
            return img
        return img.copy(0, FASCIA_GRAFICO, img.width(), img.height() - FASCIA_GRAFICO)
