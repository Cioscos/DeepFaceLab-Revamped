"""Come si lancia il servizio di estrazione manuale.

Stato di modulo, non di istanza: l'interprete e la radice del repository
sono fatti di processo. Una chiamata a `comando_servizio` prima di
qualunque `configura` solleva un errore chiaro invece di comporre un
comando con None dentro.
"""
from pathlib import Path

_python = None
_dfl_root = None


def configura(python_exe, dfl_root):
    global _python, _dfl_root
    _python = python_exe
    _dfl_root = Path(dfl_root) if dfl_root is not None else None


def comando_servizio(workdir):
    if _python is None or _dfl_root is None:
        raise RuntimeError("il servizio di estrazione non e' stato configurato")
    return _python, [str(_dfl_root / "main.py"), "extracttool", "manual",
                     "--workdir", str(workdir)]
