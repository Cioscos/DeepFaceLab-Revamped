import cv2
import numpy as np
from pathlib import Path
from core.interact import interact as io
from core import imagelib, pathex
import traceback

def cv2_imread(filename, flags=cv2.IMREAD_UNCHANGED, loader_func=None, verbose=True):
    """
    allows to open non-english characters path
    """
    try:
        if loader_func is not None:
            bytes = bytearray(loader_func(filename))
        else:
            with open(filename, "rb") as stream:
                bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        return cv2.imdecode(numpyarray, flags)
    except:
        if verbose:
            io.log_err(f"Exception occured in cv2_imread : {traceback.format_exc()}")
        return None

def cv2_imwrite(filename, img, *args):
    """Torna True se il file e' sul disco per intero.

    Il `.tmp` di mezzo non e' prudenza astratta: su un disco pieno la
    `open` riesce e la `write` no, quindi scrivere sul posto sostituisce
    il file che c'era con un troncone da 0 byte -- che chi legge dopo non
    distingue da un lavoro fatto, e che `cv2.imdecode` rifiuta con
    `!buf.empty()`. Gli errori restano ingoiati come sempre, ma ora il
    chiamante puo' accorgersene dal valore di ritorno.
    """
    ret, buf = cv2.imencode( Path(filename).suffix, img, *args)
    if ret != True:
        return False
    try:
        pathex.scrivi_al_sicuro(filename, lambda stream: stream.write(buf))
        return True
    except:
        return False

def cv2_resize(x, *args, **kwargs):
    h,w,c = x.shape
    x = cv2.resize(x, *args, **kwargs)
    
    x = imagelib.normalize_channels(x, c)
    return x
    