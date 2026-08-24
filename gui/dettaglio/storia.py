"""La pila di undo sui landmark di UN volto.

Lo stato e' 68 coppie di float: tenerne qualche decina non costa niente, e
non c'e' nessuna ragione per inventare un sistema di diff.

**Per volto, non per finestra.** Passando a un fratello si `ricomincia`:
un Ctrl+Z che disfa una modifica fatta su un altro file e' la stessa
classe di errore dell'undo della cancellazione che apparteneva alla pagina
invece che alla cartella dei volti.
"""
import copy


class Storia:
    def __init__(self, iniziale, tetto=50):
        if tetto <= 0:
            raise ValueError("il tetto della storia deve essere positivo")
        self._tetto = tetto
        self.ricomincia(iniziale)

    def ricomincia(self, iniziale):
        self._stati = [copy.deepcopy(iniziale)]
        self._i = 0
        self._salvata = 0

    def corrente(self):
        """Una COPIA: chi la riceve la modifica per costruire il passo
        successivo, e senza copia modificherebbe la pila da sotto."""
        return copy.deepcopy(self._stati[self._i])

    def applica(self, punti):
        """Aggiunge uno stato, tagliando il ramo da rifare."""
        del self._stati[self._i + 1:]
        self._stati.append(copy.deepcopy(punti))
        self._i = len(self._stati) - 1
        self._pota()

    def _pota(self):
        """Il tetto conta i passi DISFABILI, non gli stati: con tetto 3 si
        torna indietro tre volte. Serve uno stato in piu' del tetto."""
        eccesso = len(self._stati) - (self._tetto + 1)
        if eccesso <= 0:
            return
        del self._stati[:eccesso]
        self._i -= eccesso
        # Il riferimento allo stato salvato puo' cadere fuori dal tetto:
        # da li' in poi non si puo' piu' sapere se si e' tornati a lui, e
        # si dice "modificata". E' il verso che al massimo fa chiedere una
        # conferma di troppo, invece di perdere del lavoro in silenzio.
        self._salvata = None if self._salvata is None else self._salvata - eccesso
        if self._salvata is not None and self._salvata < 0:
            self._salvata = None

    def puo_disfare(self):
        return self._i > 0

    def puo_rifare(self):
        return self._i < len(self._stati) - 1

    def disfa(self):
        if self.puo_disfare():
            self._i -= 1

    def rifa(self):
        if self.puo_rifare():
            self._i += 1

    def segna_salvata(self):
        self._salvata = self._i

    def stato_salvato(self):
        """Una COPIA dei punti dell'ultimo salvataggio, o None se quello
        stato e' caduto fuori dal tetto. Serve a Revert: `modificata()`
        dice SE si e' cambiato qualcosa, non a cosa tornare."""
        if self._salvata is None:
            return None
        return copy.deepcopy(self._stati[self._salvata])

    def modificata(self):
        return self._salvata is None or self._i != self._salvata
