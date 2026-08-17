"""Il client del protocollo di estrazione manuale.

Non importa mainscripts: parla il protocollo. Il trasporto e' iniettabile
perche' i test non devono avviare un processo con torch dentro.

Ogni numero che entra da qui passa da gui/numeri.py con ENTRAMBI i
predicati: la finitezza non basta, 1e300 e' finito e uccide comunque
l'int() di un paintEvent.
"""
from gui import numeri

TIMEOUT_MS = 10000


def _punti_utilizzabili(punti):
    if not isinstance(punti, list) or not punti:
        return None
    fuori = []
    for punto in punti:
        if not isinstance(punto, (list, tuple)) or len(punto) != 2:
            return None
        x, y = punto
        if not (numeri.numero_finito(x) and numeri.numero_finito(y)):
            return None
        if not (numeri.intero_qt_utilizzabile(x) and numeri.intero_qt_utilizzabile(y)):
            return None
        fuori.append((float(x), float(y)))
    return fuori


class Servizio(object):
    def __init__(self, trasporto):
        self._trasporto = trasporto
        self._prossimo_id = 0

    def _invia(self, comando):
        self._prossimo_id += 1
        comando["id"] = self._prossimo_id
        risposta = self._trasporto.invia(comando)
        if not isinstance(risposta, dict) or risposta.get("op") == "error":
            return None
        return risposta

    def frame(self, path):
        risposta = self._invia({"op": "frame", "path": str(path)})
        if risposta is None:
            return None, None
        return risposta.get("raster"), risposta.get("shape")

    def landmark(self, centro, punta):
        risposta = self._invia({"op": "landmark",
                                "centro": [float(centro[0]), float(centro[1])],
                                "punta": [float(punta[0]), float(punta[1])]})
        if risposta is None:
            return None, None
        return risposta.get("rect"), _punti_utilizzabili(risposta.get("landmarks"))

    def salva(self, **campi):
        """Chiede al servizio di salvare il volto corrente.

        `face_idx` e' opzionale nel protocollo ma vale 0 se non lo si
        passa: due chiamate dallo stesso frame senza `face_idx` esplicito
        si sovrascrivono in silenzio, perche' il servizio nomina il file
        col nome del frame E l'indice del volto (`face_idx`), non con un
        contatore che tiene lui. Chi salva piu' volti dallo stesso frame
        deve passare `face_idx` a mano.
        """
        campi["op"] = "salva"
        risposta = self._invia(campi)
        return None if risposta is None else risposta.get("file")

    def ferma(self):
        self._trasporto.chiudi()
