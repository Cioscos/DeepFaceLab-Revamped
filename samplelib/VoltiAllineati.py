"""Elencare e leggere i volti allineati di una cartella senza sapere se il
faceset e' impacchettato: dopo `util --pack-faceset` sul disco non c'e' nessun
.jpg e i byte stanno dentro faceset.pak, agli offset che il pacchetto porta.

Chi elenca una cartella di volti a mano -- con glob o iterdir -- su un faceset
impacchettato non vede niente, e la diagnosi che ne esce ("nessun volto")
accusa il dataset invece del lettore. Qui il percorso resta la chiave per ogni
chiamante: quello del .jpg se c'e', altrimenti `cartella/nome` come se ci
fosse, e i byte arrivano dal pacchetto tramite il `loader_func` che
`DFLJPG.load` e `cv2_imread` accettano gia'."""
import pathlib

from samplelib.PackedFaceset import PackedFaceset

ESTENSIONI = {".jpg", ".jpeg", ".png"}

# (pak, mtime, dimensione) -> {nome: loader}
_indici = {}


def _indice(cartella):
    """{nome: loader} dei volti dentro faceset.pak, {} se la cartella non ne ha
    uno. In cache sulla firma del pacchetto perche' `PackedFaceset.load`
    rilegge la tabella di tutti i campioni, e i chiamanti chiedono un volto
    alla volta: senza cache l'elenco costa quadratico."""
    pak = pathlib.Path(cartella) / "faceset.pak"
    if not pak.exists():
        return {}
    st = pak.stat()
    chiave = (str(pak.resolve()), st.st_mtime_ns, st.st_size)
    if chiave not in _indici:
        _indici[chiave] = {c.filename: c.read_raw_file for c in PackedFaceset.load(pathlib.Path(cartella))}
    return _indici[chiave]


def impacchettato(cartella):
    return (pathlib.Path(cartella) / "faceset.pak").exists()


def nomi(cartella):
    """I nomi dei volti allineati, in ordine: quelli sciolti sul disco e quelli
    dentro il pacchetto. Una cartella puo' avere entrambi -- `pack` cancella
    gli originali solo se glielo si dice."""
    cartella = pathlib.Path(cartella)
    sciolti = [p.name for p in cartella.iterdir()
               if p.suffix.lower() in ESTENSIONI and not p.name.startswith(".")] if cartella.exists() else []
    return sorted(set(sciolti) | set(_indice(cartella)))


def percorsi(cartella):
    """Un percorso per volto. Per un volto impacchettato il file non esiste:
    e' il nome sotto la cartella, che `loader` sa poi ritradurre in byte."""
    cartella = pathlib.Path(cartella)
    return [cartella / n for n in nomi(cartella)]


def loader(percorso):
    """Il `loader_func` con cui leggere questo volto, None se il file sta
    davvero sul disco col suo nome (e allora il nome basta)."""
    percorso = pathlib.Path(percorso)
    if percorso.exists():
        return None
    return _indice(percorso.parent).get(percorso.name)
