"""La lettura del rapporto MENTRE l'estrazione lo scrive.

`indice.leggi` rilegge il file intero e va bene all'apertura della pagina,
una volta sola. Qui si legge una volta al secondo per tutta la durata di un
job: rileggere tutto ogni volta e' quadratico nel numero di frame, e la
deve reggere 50 000 frame.

Due cose che il file puo' fare mentre lo si legge, ed entrambe sono gia'
successe altrove in questo repository:

- **finire a meta' riga.** `ExtractReport.Scrittore` fa flush a ogni riga,
  ma fra la scrittura e il newline c'e' comunque un istante: l'offset si
  avanza solo fino all'ultimo `\\n` visto, mai oltre, o la coda della riga
  si perde e con lei la voce intera.
- **rimpicciolirsi.** Una seconda estrazione dopo aver svuotato la cache
  riparte da un file piu' corto: con l'offset vecchio non si leggerebbe
  mai piu' niente, in silenzio.
"""
import json
from pathlib import Path

from gui.estrazione import indice


class LettoreIncrementale(object):
    def __init__(self, cache_dir):
        self._percorso = Path(cache_dir) / indice.NOME_RAPPORTO
        self._offset = 0

    def nuove(self):
        """Le voci comparse dall'ultima chiamata. Non solleva mai: gira in
        uno slot di QTimer, e un'eccezione da uno slot chiama qFatal."""
        try:
            dimensione = self._percorso.stat().st_size
        except OSError:
            return []
        if dimensione < self._offset:
            self._offset = 0
        if dimensione == self._offset:
            return []
        try:
            with open(str(self._percorso), "rb") as f:
                f.seek(self._offset)
                grezzo = f.read()
        except OSError:
            return []
        # Fino all'ultimo newline, e non oltre: cio' che segue e' una riga
        # che il figlio sta ancora scrivendo.
        taglio = grezzo.rfind(b"\n")
        if taglio < 0:
            return []
        self._offset += taglio + 1
        fuori = []
        for riga in grezzo[:taglio].split(b"\n"):
            riga = riga.strip()
            if not riga:
                continue
            try:
                v = json.loads(riga.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(v, dict) and isinstance(v.get("nome"), str):
                fuori.append(v)
        return fuori
