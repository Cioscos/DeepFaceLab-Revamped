"""Il servizio di dettaglio: un volto alla volta, a piena fedelta'.

Persistente perche' l'import costa 6,11 s e servire un volto 5,80 ms: un
processo per doppio click renderebbe la finestra inutilizzabile.

Il protocollo non trasporta byte. I vettori (landmark, poligoni) vanno in
JSON, il raster (la maschera non ridotta) passa da un file annunciato DOPO
essere stato scritto -- lo stesso schema di DFL_PREVIEW_DIR
(PreviewWriter, mainscripts/TrainerLib.py): si scrive su un file
temporaneo, si rinomina in modo atomico (os.replace), e SOLO ALLORA si
annuncia il nome nella risposta. L'immagine del volto non passa affatto:
e' un JPEG ordinario, e chi la vuole la apre da se'.

La maschera dentro il JPEG e' gia' compressa: come in
mainscripts/FacesetIndex.py, la scriviamo verbatim, senza decodificarla
per ricodificarla (costerebbe una decodifica in cambio di niente). Il
formato pero' NON e' garantito PNG: DFLIMG.DFLJPG.set_xseg_mask ripiega
su JPEG oltre una soglia di dimensione (50000 byte, DFLIMG/DFLJPG.py). Il
nome del file scritto qui riflette il formato vero, riconosciuto dai
primi byte, cosi' il client non deve indovinare dall'estensione.

`DFLJPG.load` cattura le proprie eccezioni e, prima di
tornare None, chiama `io.log_err(...)` -- che stampa su `sys.stdout` vero
(`core/interact/interact.py`), lo stesso canale del protocollo JSON riga
per riga. Un file che non e' un JPEG DFL, o un percorso inesistente,
avrebbe scritto un traceback multi-riga PRIMA della risposta di errore,
desincronizzando il parser del client su una pipe vera. `rispondi()` gira
quindi sotto `contextlib.redirect_stdout(sys.stderr)`: qualunque stampa
generata durante l'elaborazione (non solo da `load`, da qualunque cosa
domani) finisce su stderr, mai sul canale del protocollo. `servi()` scrive
la riga di risposta FUORI da quel blocco, quindi il protocollo stesso non
si sposta.
"""
import contextlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from DFLIMG import DFLJPG

FIRMA_PNG = b"\x89PNG\r\n\x1a\n"


def _estensione_maschera(byte_grezzi):
    return "png" if byte_grezzi[:8] == FIRMA_PNG else "jpg"


def _polys_serializzabili(polys):
    """SegIEPolys.dump() torna 'pts' come array numpy dentro ogni
    poligono: json.dumps ci si strozza sopra. I volti di prova di questo
    servizio non hanno poligoni, quindi la suite non lo eserciterebbe da
    sola -- ma un volto reale che li ha non deve poter fermare il ciclo."""
    for poligono in polys.get("polys", []):
        pts = poligono.get("pts")
        if hasattr(pts, "tolist"):
            poligono["pts"] = pts.tolist()
    return polys


def _scrivi_maschera_e_annuncia(byte_grezzi, ident, workdir):
    """Scrive la maschera su un file temporaneo, la rinomina in modo
    atomico e ritorna il nome finale -- solo a scrittura conclusa, mai
    prima, cosi' un file annunciato esiste sempre."""
    nome = "%s_mask.%s" % (ident, _estensione_maschera(byte_grezzi))
    finale = Path(workdir) / nome
    provvisorio = Path(workdir) / (nome + ".tmp")
    provvisorio.write_bytes(byte_grezzi)
    os.replace(str(provvisorio), str(finale))
    return nome


def rispondi(comando, workdir):
    """Un comando -> una risposta. Non solleva mai: qualunque motivo di
    fallimento (file assente, non un JPEG DFL, comando malformato,
    operazione sconosciuta) diventa op='error', mai un'eccezione che
    fermerebbe chi chiama. Gira sotto redirect_stdout(stderr): niente di
    quello che succede qui dentro -- neanche una stampa di libreria che
    non ci aspettiamo -- puo' finire sul canale del protocollo."""
    with contextlib.redirect_stdout(sys.stderr):
        if not isinstance(comando, dict):
            return {"op": "error", "id": None, "motivo": "comando non valido"}
        ident = comando.get("id")
        if comando.get("op") != "open":
            return {"op": "error", "id": ident, "motivo": "operazione sconosciuta"}
        percorso = comando.get("path")
        try:
            dfl = DFLJPG.load(str(percorso))
            if dfl is None:
                raise ValueError("non e' un JPEG DFL")
            forma = list(dfl.get_shape())
            landmarks = np.asarray(dfl.get_landmarks()).tolist()
            polys = None
            if dfl.has_seg_ie_polys():
                polys = _polys_serializzabili(dfl.get_seg_ie_polys().dump())
            risposta = {"op": "opened", "id": ident, "shape": forma,
                        "face_type": dfl.get_face_type(),
                        "landmarks": landmarks, "polys": polys, "mask": None}
            if dfl.has_xseg_mask():
                byte_grezzi = bytes(np.asarray(dfl.get_xseg_mask_compressed()).tobytes())
                risposta["mask"] = _scrivi_maschera_e_annuncia(byte_grezzi, ident, workdir)
            return risposta
        except Exception as e:
            return {"op": "error", "id": ident, "motivo": str(e)}


def servi(entrata, uscita, workdir):
    """Legge un comando JSON a riga da 'entrata', scrive una risposta JSON
    a riga su 'uscita', con un flush dopo ognuna -- il client legge una
    riga alla volta e aspetta: una risposta ferma nel buffer lo blocca per
    l'intero timeout, a ogni doppio click. Una riga non decodificabile
    come JSON riceve comunque una risposta di errore (id=None, non
    essendoci un comando da cui leggerlo): un client in attesa non deve
    restare appeso perche' la riga che ha mandato non si capiva."""
    for riga in entrata:
        riga = riga.strip()
        if not riga:
            continue
        try:
            comando = json.loads(riga)
        except ValueError as e:
            risposta = {"op": "error", "id": None,
                        "motivo": "riga non decodificabile come JSON: %s" % e}
        else:
            risposta = rispondi(comando, workdir)
        uscita.write(json.dumps(risposta) + "\n")
        uscita.flush()


def main(workdir):
    servi(sys.stdin, sys.stdout, Path(workdir))
