"""I pesi esterni: manifest con sha256, download alla prima necessita', avviso
di licenza, caricamento strict. Mai un peso non verificato sul disco."""
import hashlib
import pathlib
import shutil
import tomllib
import urllib.request

import torch
from core.interact import interact as io

from facelib.supervisori import adaface, lpips, vit

REPO = pathlib.Path(__file__).resolve().parents[2]            # _internal/DeepFaceLab
MODELLI_ESTERNI = REPO.parents[1] / "_internal" / "modelli-esterni"

MANIFEST = pathlib.Path(__file__).with_name("manifest.toml")


class PesoNonVerificato(RuntimeError):
    pass


class PesoMancante(RuntimeError):
    def __init__(self, nome, causa=None):
        super().__init__(f"peso esterno «{nome}» non trovato e non scaricabile"
                         + (f" ({causa})" if causa else "")
                         + ": controlla la rete, o lancia `python3 -m tools.banco prepara --pesi`")


def leggi_manifest(percorso=None):
    with open(percorso or MANIFEST, "rb") as f:
        return tomllib.load(f)


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blocco in iter(lambda: f.read(1 << 20), b""):
            h.update(blocco)
    return h.hexdigest()


def scarica(nome, radice=MODELLI_ESTERNI, manifest=None):
    voce = (manifest or leggi_manifest())[nome]
    dest = pathlib.Path(radice) / nome / voce["file"]
    if dest.exists() and _sha256(dest) == voce["sha256"]:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    provvisorio = dest.with_suffix(dest.suffix + ".parziale")
    io.log_info(f"scarico {nome} da {voce['url']}")
    richiesta = urllib.request.Request(voce["url"], headers={"User-Agent": "DeepFaceLab-banco/1.0"})
    try:
        with urllib.request.urlopen(richiesta, timeout=60) as r, open(provvisorio, "wb") as f:
            shutil.copyfileobj(r, f)
    except BaseException:
        # un trasferimento interrotto a meta' non lascia un .parziale sul disco
        provvisorio.unlink(missing_ok=True)
        raise
    sha = _sha256(provvisorio)
    if sha != voce["sha256"]:
        provvisorio.unlink()
        raise PesoNonVerificato(f"{nome}: sha256 diversa da quella del manifest")
    provvisorio.replace(dest)
    io.log_info(f"{nome}: {dest.stat().st_size} byte, sha256 {sha}")
    return dest


def scrivi_licenze(radice, manifest=None):
    manifest = manifest or leggi_manifest()
    righe = ["# Licenze dei pesi esterni del banco", "",
             "Generato dal manifest dei pesi esterni; ogni sezione nomina il file che copre.", ""]
    for nome, v in manifest.items():
        righe += [f"## {nome}", "", f"* File: `{v['file']}`", f"* Origine: <{v['url']}>",
                  f"* Licenza: {v['licenza']}", "", "```", v["avviso"].strip(), "```", ""]
    out = pathlib.Path(radice) / "LICENZE.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(righe), encoding="utf-8")
    return out


def _leggi_state_dict(percorso):
    percorso = pathlib.Path(percorso)
    sd = torch.load(str(percorso), map_location="cpu", weights_only=False)
    return sd.get("state_dict", sd) if isinstance(sd, dict) else sd


def carica_state_dict(rete, sd, rinomina=None, prefisso_da_togliere="", solo_prefisso=None):
    out = {}
    for k, v in sd.items():
        if solo_prefisso and not k.startswith(solo_prefisso):
            continue
        if prefisso_da_togliere and k.startswith(prefisso_da_togliere):
            k = k[len(prefisso_da_togliere):]
        out[rinomina(k) if rinomina else k] = v
    rete.load_state_dict(out)          # strict: una chiave mancante o in piu' e' un errore
    return rete.eval()


def carica_torch(rete, percorso, rinomina=None, prefisso_da_togliere="", solo_prefisso=None):
    return carica_state_dict(rete, _leggi_state_dict(percorso), rinomina=rinomina,
                              prefisso_da_togliere=prefisso_da_togliere, solo_prefisso=solo_prefisso)


def _file(nome, radice):
    """Il peso sul disco. Se manca lo scarica dal manifest (sha256 verificata)
    e scrive l'avviso di licenza accanto: la prima potenza accesa in un
    addestramento deve bastare, un utente della GUI non conosce il banco.
    PesoMancante solo se anche lo scaricamento fallisce; una sha che non
    torna resta PesoNonVerificato, non si degrada a «mancante»."""
    voce = leggi_manifest()[nome]
    p = pathlib.Path(radice) / nome / voce["file"]
    if p.exists():
        return p
    io.log_info(f"peso esterno «{nome}» assente: lo scarico ora, una volta sola")
    try:
        scarica(nome, radice)
    except OSError as e:               # URLError e' un OSError: rete, DNS, disco
        raise PesoMancante(nome, e) from e
    scrivi_licenze(radice)
    return p


def carica_dinov2(radice=MODELLI_ESTERNI):
    return carica_torch(vit.dinov2_vits14(), _file("dinov2-vits14", radice))


def carica_lpips(radice=MODELLI_ESTERNI):
    rete = lpips.LPIPS()
    feat = {k[len("features."):]: v for k, v in _leggi_state_dict(_file("alexnet-torchvision", radice)).items()
            if k.startswith("features.")}
    rete.features.load_state_dict(feat)                       # strict: le 10 chiavi conv 0/3/6/8/10
    lin = {k.replace("lin", "").replace(".model.1.weight", ".weight"): v
           for k, v in _leggi_state_dict(_file("lpips-alex", radice)).items()}
    rete.lin.load_state_dict(lin)                             # lin0.model.1.weight -> 0.weight
    return rete.eval()


def carica_adaface_rete(radice=MODELLI_ESTERNI):
    return carica_torch(adaface.IR101(), _file("adaface-ir101-webface12m", radice),
                        prefisso_da_togliere="model.", solo_prefisso="model.")


def carica_adaface(radice=MODELLI_ESTERNI):
    return adaface.encoder(carica_adaface_rete(radice))
