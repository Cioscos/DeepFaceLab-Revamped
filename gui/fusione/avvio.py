"""Come si lancia il servizio di fusione. Stato di modulo come
gui/estrazione/avvio.py: python e dfl_root sono fatti di processo."""
from pathlib import Path

_python = None
_dfl_root = None


def configura(python_exe, dfl_root):
    global _python, _dfl_root
    _python = python_exe
    _dfl_root = Path(dfl_root) if dfl_root is not None else None


def dfl_root():
    return _dfl_root


def comando_servizio(workdir, parametri):
    if _python is None or _dfl_root is None:
        raise RuntimeError("il servizio di fusione non e' stato configurato")
    argomenti = [str(_dfl_root / "main.py"), "mergetool", "session",
                 "--input-dir", str(parametri["input_dir"]),
                 "--output-dir", str(parametri["output_dir"]),
                 "--output-mask-dir", str(parametri["output_mask_dir"]),
                 "--aligned-dir", str(parametri["aligned_dir"]),
                 "--model-dir", str(parametri["model_dir"]),
                 "--model", str(parametri["model"]),
                 "--force-model-name", str(parametri["force_model_name"]),
                 "--workers", str(int(parametri["workers"])),
                 "--workdir", str(workdir)]
    if parametri.get("force_gpu_idxs"):
        argomenti += ["--force-gpu-idxs", str(parametri["force_gpu_idxs"])]
    return _python, argomenti
