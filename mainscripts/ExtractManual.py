"""Il servizio di estrazione manuale: un frame alla volta.

Stessa forma di mainscripts/FacesetDetail.py, e per le stesse ragioni. Due
cautele che quel file ha pagato e che qui valgono identiche:

- `rispondi` gira sotto contextlib.redirect_stdout(sys.stderr), perche' una
  qualunque stampa di libreria sul canale del protocollo desincronizzerebbe
  il parser del client;
- il raster viene annunciato DOPO os.replace, mai prima, cosi' un file
  annunciato esiste sempre.

Nessuno stato attraversa una chiamata all'altra: le tre operazioni (frame,
landmark, salva) sono ciascuna autosufficiente nel proprio comando, e il
servizio non carica alcun modello -- `landmarks_da_vettore` e' geometria
pura. La tela Qt possiede l'interazione: qui non c'e' nessun loop di
eventi, solo un comando e una risposta.
"""
import contextlib
import json
import os
import sys
from pathlib import Path

import numpy as np


def _scrivi_e_annuncia(byte_grezzi, nome, workdir):
    """Scrive su un file temporaneo, rinomina in modo atomico e torna il
    nome finale -- solo a scrittura conclusa, mai prima, cosi' un file
    annunciato esiste sempre."""
    finale = Path(workdir) / nome
    provvisorio = Path(workdir) / (nome + ".tmp")
    provvisorio.write_bytes(byte_grezzi)
    os.replace(str(provvisorio), str(finale))
    return nome


def _op_frame(comando, workdir):
    import cv2
    from core.cv2ex import cv2_imread
    percorso = comando.get("path")
    immagine = cv2_imread(str(percorso))
    if immagine is None:
        raise ValueError("frame non leggibile: %s" % percorso)
    ok, codificato = cv2.imencode(".jpg", immagine,
                                  [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise ValueError("codifica del raster fallita")
    nome = _scrivi_e_annuncia(codificato.tobytes(),
                              "%s_frame.jpg" % comando.get("id"), workdir)
    h, w = immagine.shape[:2]
    return {"op": "frame", "id": comando.get("id"), "raster": nome,
            "shape": [int(h), int(w)]}


def _op_landmark(comando):
    from mainscripts import ExtractorLib
    rect, lmrks = ExtractorLib.landmarks_da_vettore(comando.get("centro"),
                                                    comando.get("punta"))
    if lmrks is None:
        raise ValueError("vettore di lunghezza nulla: nessun landmark")
    return {"op": "landmark", "id": comando.get("id"),
            "rect": [int(v) for v in rect],
            "landmarks": np.asarray(lmrks).tolist()}


def _op_salva(comando):
    from core.cv2ex import cv2_imread
    from facelib import FaceType
    from mainscripts import ExtractorLib

    percorso = Path(comando.get("path"))
    immagine = cv2_imread(str(percorso))
    if immagine is None:
        raise ValueError("frame non leggibile: %s" % percorso)
    uscita = Path(comando.get("output_dir"))
    uscita.mkdir(parents=True, exist_ok=True)
    nome = "%s_%d.jpg" % (percorso.stem, int(comando.get("face_idx", 0)))
    scritto = ExtractorLib.salva_volto(
        immagine=immagine,
        rect=comando.get("rect"),
        image_landmarks=np.asarray(comando.get("landmarks"), dtype=np.float32),
        face_type=FaceType.fromString(comando.get("face_type")),
        image_size=int(comando.get("image_size")),
        jpeg_quality=int(comando.get("jpeg_quality")),
        output_filepath=uscita / nome,
        source_filename=percorso.name,
        manuale=True)
    if scritto is None:
        # Il rettangolo l'ha tracciato l'utente: salva_volto non lo scarta
        # per l'area (manuale=True lo esclude), ma puo' comunque tornare
        # None -- quel caso non deve mai diventare un AttributeError su
        # Path(None).name.
        raise ValueError("volto scartato: nessun file scritto")
    return {"op": "salvato", "id": comando.get("id"), "file": Path(scritto).name}


def rispondi(comando, workdir):
    """Un comando -> una risposta. Non solleva mai."""
    with contextlib.redirect_stdout(sys.stderr):
        if not isinstance(comando, dict):
            return {"op": "error", "id": None, "motivo": "comando non valido"}
        ident = comando.get("id")
        op = comando.get("op")
        try:
            if op == "frame":
                return _op_frame(comando, workdir)
            if op == "landmark":
                return _op_landmark(comando)
            if op == "salva":
                return _op_salva(comando)
            raise ValueError("operazione sconosciuta: %s" % op)
        except Exception as e:
            return {"op": "error", "id": ident, "motivo": str(e)}


def servi(entrata, uscita, workdir):
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
