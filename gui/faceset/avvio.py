"""Come si lancia il servizio di dettaglio.

Separato dal client perche' il client deve poter essere provato senza
avviare niente, e perche' il percorso di main.py lo sa la pagina, non chi
parla il protocollo.

Stato di modulo, non di istanza: l'interprete e la radice del repository
sono fatti di processo, gli stessi per ogni finestra di dettaglio che
questa sessione aprira'. Una seconda chiamata a `configura` sostituisce
la precedente in modo pulito (nessun residuo della prima); una chiamata a
`comando_servizio` prima di qualunque `configura` solleva un errore
chiaro invece di comporre un comando insensato con `None` dentro.
"""
from pathlib import Path

_python = None
_dfl_root = None


def configura(python_exe, dfl_root):
    global _python, _dfl_root
    _python = python_exe
    _dfl_root = Path(dfl_root)


def comando_servizio(workdir):
    if _python is None or _dfl_root is None:
        raise RuntimeError("il servizio di dettaglio non e' stato configurato")
    return _python, [str(_dfl_root / "main.py"), "facesettool", "detail",
                     "--workdir", str(workdir)]
