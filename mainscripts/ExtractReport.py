"""Il rapporto per frame prodotto dall'estrazione.

Una riga per frame, scritta MENTRE l'estrazione gira e non alla fine: uno
Stop dalla GUI e' l'uccisione del process tree, e cio' che resta in memoria
si perde. File append-only, ultima riga vince -- la stessa forma
dell'indice del faceset, e per le stesse ragioni.

La chiave e' (nome, dimensione, mtime) perche' un sort che rinomina i file
non deve invalidare il rapporto.
"""
import json
import os
from pathlib import Path

FORMATO = 1
NOME_RAPPORTO = "frames.ndjson"

STATO_AUTOMATICO = "automatico"
STATO_CONFERMATO = "confermato"
STATO_SALTATO = "saltato"

# La coppia che il rapporto scrive per i rami tracciati a mano. Una parola
# sola in un posto solo: la scrivono Extractor.main (il ramo `--detector
# manual`, la finestra cv2) e ExtractManual (_op_salva, la sessione nativa
# della GUI), e sono due file diversi.
MOTORE_MANUALE = "manual"


def chiave_di(path):
    st = os.stat(str(path))
    return Path(path).name, st.st_size, st.st_mtime_ns


def voce(path, volti, luminanza, stato, motore=None):
    nome, dimensione, mtime = chiave_di(path)
    return {"formato": FORMATO, "nome": nome, "dimensione": dimensione,
            "mtime": mtime, "n_volti": len(volti), "volti": list(volti),
            "luminanza": None if luminanza is None else float(luminanza),
            "stato": stato,
            # Assente o None = sconosciuto, mai un motore dedotto: le voci
            # scritte prima di questa ondata e quelle ricostruite da
            # `extracttool index` non sanno da dove vengano.
            "motore": None if motore is None else str(motore)}


def leggi(cache_dir):
    """Ultima riga vince. Una riga non decodificabile -- il troncamento che
    uno Stop lascia dietro -- si salta senza sollevare."""
    percorso = Path(cache_dir) / NOME_RAPPORTO
    if not percorso.exists():
        return []
    per_nome = {}
    with open(str(percorso), "r", encoding="utf-8") as f:
        for riga in f:
            riga = riga.strip()
            if not riga:
                continue
            try:
                v = json.loads(riga)
            except ValueError:
                continue
            nome = v.get("nome")
            if isinstance(nome, str):
                per_nome[nome] = v
    return list(per_nome.values())


class Scrittore(object):
    """Apre in append e fa flush a ogni riga: chi legge mentre l'estrazione
    gira deve vedere le righe gia' scritte, non un buffer."""

    def __init__(self, cache_dir):
        self._percorso = Path(cache_dir) / NOME_RAPPORTO
        self._f = None

    def __enter__(self):
        Path(self._percorso).parent.mkdir(parents=True, exist_ok=True)
        self._chiudi_una_riga_troncata(self._percorso)
        self._f = open(str(self._percorso), "a", encoding="utf-8")
        return self

    def scrivi(self, voce_da_scrivere):
        self._f.write(json.dumps(voce_da_scrivere) + "\n")
        self._f.flush()

    def __exit__(self, *_eccezione):
        if self._f is not None:
            self._f.close()
            self._f = None
        return False

    @staticmethod
    def _chiudi_una_riga_troncata(percorso):
        """Un arresto a meta' di una `write` lascia una riga senza `\\n`.

        Senza questa chiusura la prima riga della corsa successiva le si
        incolla dietro e ne diventano illeggibili DUE: quella troncata, che
        era gia' persa, e quella nuova, che non lo era.
        """
        try:
            with open(str(percorso), "rb") as f:
                f.seek(0, os.SEEK_END)
                if f.tell() == 0:
                    return
                f.seek(-1, os.SEEK_END)
                intera = f.read(1) == b"\n"
        except OSError:
            return
        if not intera:
            with open(str(percorso), "a", encoding="utf-8") as f:
                f.write("\n")
