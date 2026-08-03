"""Il manifest degli asset e il download verificato che li porta a terra.

I pesi delle reti (`facelib/*.npy`), ffmpeg, i modelli pre-addestrati e
EbSynth non stanno nel repo -- sono gitignorati o troppo grandi per git -- e
vanno scaricati da GitHub Releases. Questo modulo e' l'unico punto che tocca
la rete per farlo, e lo fa con due proprieta' che il resto dell'installer da
per scontate:

- **verifica**: niente arriva sul disco dell'utente senza che il suo
  SHA-256 combaci con quello dichiarato nel manifest;
- **ripresa**: un'interruzione a meta' di un file da 1.8 GB e' l'evento
  atteso, non l'eccezione, e non deve costare di ricominciare da zero.

`download_verified` scarica sempre su `<dest>.part` e rinomina solo a
verifica riuscita: e' cosi' che un rilancio dopo un crash trova o un file
finito e corretto (`dest`), o un file parziale da riprendere (`<dest>.part`),
mai un file a meta' spacciato per completo.
"""
from __future__ import annotations

import hashlib
import shutil
import tarfile
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from setup.paths import InstallPaths

# sha256_of legge a blocchi di 1 MiB: abbastanza grande da non essere il
# collo di bottiglia su un file da centinaia di MB, abbastanza piccolo da non
# tenere mai un intero .npy o un archivio da 700 MB in RAM in un colpo solo.
HASH_CHUNK_BYTES = 1024 * 1024

# I chunk del download sono piu' piccoli dei blocchi dello sha256: e' quello
# che rende visibile, byte per byte, un'interruzione a meta' invece che
# scoprirla solo alla fine di una singola grande read() bloccante.
DOWNLOAD_CHUNK_BYTES = 64 * 1024

# GitHub pone un limite di 2 GB per file di release, ma il margine serve: la
# regola e' generale, non un caso speciale di pretrain_faces.
# Chiunque cambi questo valore deve anche rigenerare setup/manifest.toml,
# altrimenti gli asset gia' spezzati restano spezzati alla vecchia soglia.
MAX_PART_BYTES = 700 * 1024 * 1024

# Il manifest spedito con l'installer, accanto a questo modulo. setup/__main__.py
# (step_ensure_assets) lo usa come default cosi' che il percorso si costruisca
# in un solo posto anche per questo file, non solo per InstallPaths.
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "manifest.toml"


@dataclass(frozen=True)
class AssetPart:
    """Un archivio scaricabile singolarmente: una parte di un asset."""
    url: str
    sha256: str
    size: int


#: I due soli valori validi del campo `kind` di un asset. Un
#: booleano avrebbe fatto lo stesso lavoro logico, ma questi due nomi si
#: leggono da soli nel TOML mentre un `blocks = true/false` richiederebbe
#: comunque un commento per dire cosa succede nell'altro caso.
_VALID_ASSET_KINDS = ("archive", "block")


@dataclass(frozen=True)
class Asset:
    """Una voce del manifest: uno o piu' archivi che finiscono tutti nella
    stessa destinazione, obbligatoria o opzionale.

    `kind` dice a `ensure_asset` come trattare `parts`:

    - `"archive"` (il default, e il caso di tutti gli asset finche' nessuno
      supera il tetto da solo): ogni parte e' un `.tar.gz` indipendente,
      scaricabile, verificabile ed estraibile per conto proprio.
    - `"block"`: le parti sono blocchi di byte di un unico `.tar.gz` --
      `pretrain_faces/faceset.pak`, il formato `PackedFaceset`, e' un blob
      singolo senza sottoinsiemi da isolare in un archivio a se'. Nessuna
      parte e' estraibile da sola: `ensure_asset` le riconcatena in ordine
      (l'ordine di `parts`, non un `sorted()` sui nomi) e verifica
      `whole_sha256` -- lo sha256 dell'intero riassemblato, distinto dallo
      sha256 di ciascun blocco -- prima di estrarre.

    Deliberatamente **non dedotto** dal nome file o dall'estensione delle
    parti (es. `.000`, `.001`): un'inferenza implicita e' cio' che si rompe
    in silenzio quando qualcuno rinomina un file.
    """
    name: str
    dest: str                    # relativo a InstallPaths.root
    required: bool
    parts: tuple[AssetPart, ...]
    kind: str = "archive"
    whole_sha256: str | None = None   # richiesto se kind == "block", None altrimenti


def load_manifest(path: Path) -> list[Asset]:
    """Legge setup/manifest.toml (o un manifest equivalente) in una lista di Asset.

    Quasi nessuna validazione di rete o di dimensione qui: il tetto dei 700
    MB e l'obbligatorieta' dei quattro asset principali riguardano il file
    spedito, non questa funzione -- un manifest costruito a mano puo'
    legittimamente violarli. Il campo `kind` fa eccezione ed e' validato qui:
    un `kind` che non e' ne' "archive" ne' "block" e' un manifest scritto a
    mano o corrotto, non un caso che `ensure_asset` debba scoprire a meta'
    download. `kind` e' obbligatorio quanto `name`/`dest` -- un
    default silenzioso su "archive" trasformerebbe un manifest scritto a
    mano senza `kind` in un fallimento tardivo e fuorviante dentro
    `tarfile.open` (un asset "block" letto come "archive"), invece che in
    un errore immediato che nomina il campo mancante.
    """
    with Path(path).open("rb") as fh:
        raw = tomllib.load(fh)
    assets = []
    for entry in raw.get("asset", ()):
        if "kind" not in entry:
            raise ValueError(
                f"manifest: asset '{entry.get('name')}' non dichiara "
                "'kind' -- deve essere esplicito, uno tra "
                f"{_VALID_ASSET_KINDS}"
            )
        kind = entry["kind"]
        if kind not in _VALID_ASSET_KINDS:
            raise ValueError(
                f"manifest: asset '{entry.get('name')}' ha kind={kind!r} non "
                f"valido -- atteso uno tra {_VALID_ASSET_KINDS}"
            )
        whole_sha256 = entry.get("whole_sha256")
        if kind == "block" and not whole_sha256:
            raise ValueError(
                f"manifest: asset '{entry.get('name')}' e' a blocchi "
                "(kind='block') ma non dichiara whole_sha256 -- senza, "
                "l'installer non puo' verificare l'archivio riconcatenato "
                "prima di estrarlo"
            )
        assets.append(Asset(
            name=entry["name"],
            dest=entry["dest"],
            required=entry.get("required", False),
            kind=kind,
            whole_sha256=whole_sha256,
            parts=tuple(
                AssetPart(url=p["url"], sha256=p["sha256"], size=p["size"])
                for p in entry.get("parts", ())
            ),
        ))
    return assets


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_open_error_message(asset_name: str | None, url: str, exc: urllib.error.URLError) -> str:
    """Messaggio con la forma «Perche': ... Cosa fare: ...» che l'installer usa
    ovunque: il 404 grezzo di urlopen non nomina ne' l'asset ne' l'URL ne' una
    causa
    plausibile, ed e' l'errore che l'utente incontra per primo, alla prima
    corsa reale dell'installer con gli URL veri."""
    who = f"l'asset '{asset_name}'" if asset_name else "un asset del manifest"
    if isinstance(exc, urllib.error.HTTPError):
        perche = f"il server ha risposto HTTP {exc.code} {exc.reason}"
        if exc.code == 404:
            cosa_fare = (
                "verifica che la release GitHub sia stata pubblicata e che il "
                "tag nel manifest sia ancora quello giusto: un URL di "
                "manifest.toml punta a un asset di una release specifica, e "
                "una release non ancora pubblicata o un tag cambiato danno "
                "esattamente questo errore. Un `curl -sI -L` anonimo sullo "
                "stesso URL dice subito se l'asset e' raggiungibile."
            )
        else:
            cosa_fare = (
                "verifica l'URL nel manifest e riprova; se il problema "
                "persiste potrebbe essere un proxy aziendale che altera la "
                "richiesta."
            )
    else:
        perche = f"impossibile raggiungere l'host ({exc.reason})"
        cosa_fare = (
            "verifica che questa macchina sia online e che nessun proxy "
            "aziendale stia bloccando la richiesta, poi rilancia l'installer."
        )
    return (
        f"impossibile scaricare {who} da {url}. Perche': {perche}. "
        f"Cosa fare: {cosa_fare}"
    )


def download_verified(
    part: AssetPart, dest: Path, log, opener=urllib.request.urlopen, asset_name: str | None = None
) -> None:
    """Scarica `part` su `dest`, verificato e ripreso.

    1. se `dest` esiste gia' con lo sha256 giusto, non tocca la rete;
    2. altrimenti scarica su `<dest>.part`, riprendendo con `Range` se quel
       file esiste gia' da un tentativo precedente;
    3. se l'apertura della connessione fallisce (404, host irraggiungibile,
       ...) solleva un errore che nomina l'asset, l'URL e una causa
       plausibile invece del solo testo grezzo di urlopen (finding 3);
    4. se il server ignora l'header `Range` di ripresa (risposta senza
       status 206) invece di rispondere con il segmento richiesto, lo dice
       esplicitamente: senza questo controllo il file duplicato che ne
       risulta arriverebbe comunque al confronto sha256 sotto e verrebbe
       diagnosticato come "corrotto o manomesso", causa sbagliata per un
       proxy/CDN che non supporta i download parziali (finding 4);
    5. se il trasferimento si ferma prima di raggiungere `part.size`, solleva
       senza toccare `<dest>.part`: e' li' che il prossimo tentativo riprende;
    6. a trasferimento completo, verifica lo sha256 per intero (non un
       prefisso): se sbagliato cancella sia `<dest>.part` sia `dest` e
       solleva `ValueError` con l'atteso e il calcolato, cosi' un rilancio
       non lo trova e non lo salta come valido;
    7. se giusto, rinomina `<dest>.part` in `dest`.
    """
    dest = Path(dest)
    if dest.exists() and sha256_of(dest) == part.sha256:
        return  # gia' corretto: niente rete

    part_file = dest.parent / (dest.name + ".part")
    part_file.parent.mkdir(parents=True, exist_ok=True)
    resume_from = part_file.stat().st_size if part_file.exists() else 0

    request = urllib.request.Request(part.url)
    if resume_from:
        request.add_header("Range", f"bytes={resume_from}-")
        if log:
            log.info("riprendo %s da %d/%d byte", part.url, resume_from, part.size)
    elif log:
        log.info("scarico %s (%d byte)", part.url, part.size)

    try:
        response = opener(request)
    except urllib.error.URLError as exc:
        raise RuntimeError(_download_open_error_message(asset_name, part.url, exc)) from exc

    ignored_range = False
    if resume_from:
        status = getattr(response, "status", None)
        if status is None:
            status = response.getcode()
        ignored_range = status != 206

    mode = "ab" if resume_from else "wb"
    with response, part_file.open(mode) as fh:
        while True:
            chunk = response.read(DOWNLOAD_CHUNK_BYTES)
            if not chunk:
                break
            fh.write(chunk)

    if ignored_range:
        # Il server ha risposto senza status 206: ha rimandato il file per
        # intero da byte 0 invece del segmento richiesto con Range, e quanto
        # appena scritto e' i byte gia' presenti nel .part con lo stream
        # intero riaccodato dietro -- mai un file valido. Stessa pulizia del
        # ramo sha256 sotto (cancellare .part e dest), cosi' il prossimo
        # tentativo riparte pulito da zero, senza Range, e non incontra piu'
        # il problema.
        part_file.unlink(missing_ok=True)
        if dest.exists():
            dest.unlink()
        raise ConnectionError(
            f"il server per {part.url} ha ignorato la richiesta di ripresa "
            "(header Range, atteso status 206 e non l'ha dato): ha rimandato "
            "il file per intero invece del segmento richiesto. Perche': "
            "probabilmente un proxy o una CDN che non supporta i download "
            "parziali, non un file corrotto. Cosa fare: rilancia "
            "l'installer, che riprendera' da capo senza Range."
        )

    size = part_file.stat().st_size
    if size < part.size:
        # Trasferimento interrotto, non corrotto: <dest>.part resta sul
        # disco cosi' come sta, per la ripresa al prossimo tentativo.
        raise ConnectionError(
            f"download interrotto per {part.url}: {size} di {part.size} byte "
            "ricevuti (si riprendera' al prossimo tentativo)"
        )

    digest = sha256_of(part_file)
    if digest != part.sha256:
        part_file.unlink(missing_ok=True)
        if dest.exists():
            dest.unlink()
        raise ValueError(
            f"sha256 non corrisponde per {part.url}: atteso {part.sha256}, "
            f"calcolato {digest}"
        )
    part_file.replace(dest)


def _part_filename(part: AssetPart, index: int) -> str:
    name = Path(urlsplit(part.url).path).name
    return name or f"part-{index:02d}.bin"


def asset_is_complete(asset: Asset, paths: InstallPaths) -> bool:
    """Vero se ogni parte di `asset` risulta gia' estratta, senza toccare la
    rete ne' ricalcolare uno sha256.

    Guarda solo il marcatore `<file>.extracted` che `ensure_asset` scrive
    accanto a ogni archivio scaricato, nella stessa cache
    (`<paths.internal>/_e/downloads/<asset.name>/`): e' economico apposta
    (solo `Path.exists()`), pensato per essere interrogato spesso -- dal
    riepilogo di `step_verify` e dalla domanda su `pretrain_faces`, entrambi
    in `setup/__main__.py` -- senza ricalcolare lo sha256 di
    un archivio che puo' pesare centinaia di MB: quel calcolo lo fa gia',
    una volta, `ensure_asset` stesso quando decide se davvero (ri)scaricare.

    Non guarda `paths.root / asset.dest`: per un asset come `facelib`,
    quella destinazione e' dentro l'albero del repo clonato e contiene gia'
    file `.py` tracciati da git a prescindere dal fatto che i pesi `.npy`
    siano mai stati scaricati -- "la cartella non e' vuota" direbbe sempre
    "presente", anche a zero byte di asset scaricati. Vale allo stesso modo
    per tutte le voci del manifest, non solo per facelib: nessuna qui sotto
    e' un caso speciale.

    Per `kind == "block"` non esiste un marcatore per parte --
    nessun blocco si estrae da solo, quindi nessun blocco ha un `.extracted`
    proprio: c'e' un solo marcatore per l'intero asset, scritto da
    `ensure_asset` dopo che l'archivio riassemblato e' stato estratto (vedi
    li' per il nome esatto).
    """
    if not asset.parts:
        return False
    cache_dir = paths.internal / "_e" / "downloads" / asset.name
    if asset.kind == "block":
        return (cache_dir / f"{asset.name}.extracted").exists()
    return all(
        (cache_dir / (_part_filename(part, index) + ".extracted")).exists()
        for index, part in enumerate(asset.parts)
    )


def _ensure_block_asset(
    asset: Asset, dest_dir: Path, cache_dir: Path, log, opener
) -> bool:
    """La meta' di `ensure_asset` per `kind == "block"`: scarica
    ogni blocco verificato, li riconcatena **in ordine di manifest** (mai un
    `sorted()` sui nomi -- un publisher che carica i blocchi con nomi che
    ordinano diversamente dall'ordine reale li riunirebbe sbagliati), verifica
    lo sha256 dell'intero, e solo allora estrae.

    Se lo sha256 dell'intero non torna: non si sa quale blocco sia il
    colpevole, quindi si cancella tutto -- blocchi compresi. Lasciarne uno
    marcio sul disco farebbe ripetere lo stesso fallimento a ogni rilancio
    successivo, senza che l'utente sospetti che quel file esista.

    I blocchi vengono cancellati appena l'archivio riassemblato e' verificato
    corretto, prima ancora di estrarlo: con `pretrain_faces` (~1.8 GB)
    tenerli entrambi vivi fino a dopo l'estrazione raddoppierebbe lo spazio
    occupato piu' a lungo del necessario.
    """
    marker = cache_dir / f"{asset.name}.extracted"
    if marker.exists():
        return False  # gia' completo: niente rete, niente ri-scaricare blocchi gia' ripuliti

    block_paths = []
    for index, part in enumerate(asset.parts):
        block_path = cache_dir / _part_filename(part, index)
        block_paths.append(block_path)
        already_downloaded = block_path.exists() and sha256_of(block_path) == part.sha256
        if not already_downloaded:
            download_verified(part, block_path, log, opener=opener, asset_name=asset.name)

    whole_path = cache_dir / f"{asset.name}.reassembled.tar.gz"
    if log:
        log.info("riconcateno %d blocchi di %s", len(block_paths), asset.name)
    with whole_path.open("wb") as out:
        for block_path in block_paths:  # ordine di asset.parts, non sorted()
            with block_path.open("rb") as fh:
                shutil.copyfileobj(fh, out)

    digest = sha256_of(whole_path)
    if digest != asset.whole_sha256:
        whole_path.unlink(missing_ok=True)
        for block_path in block_paths:
            block_path.unlink(missing_ok=True)
        raise ValueError(
            f"sha256 non corrisponde per l'archivio riconcatenato di "
            f"'{asset.name}': atteso {asset.whole_sha256}, calcolato {digest}. "
            "Non e' possibile sapere quale blocco sia corrotto: tutti i "
            "blocchi scaricati sono stati cancellati, il prossimo tentativo "
            "riparte da zero."
        )

    # Verificato: i blocchi non servono piu', liberare lo spazio prima di
    # estrarre (non dopo -- vedi la docstring).
    for block_path in block_paths:
        block_path.unlink(missing_ok=True)

    if log:
        log.info("estraggo %s in %s", whole_path.name, dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    # filter="data": stessa difesa di sotto, vedi il commento nel ramo "archive".
    with tarfile.open(whole_path) as tf:
        tf.extractall(dest_dir, filter="data")
    whole_path.unlink()
    marker.write_text("")
    return True  # changed e' sempre vero da qui: si e' almeno estratto


def ensure_asset(
    asset: Asset, paths: InstallPaths, log, opener=urllib.request.urlopen
) -> bool:
    """Porta `asset` allo stato completo: scarica le parti mancanti, estrae
    l'archivio (o gli archivi) nella destinazione dichiarata dal manifest.

    Restituisce True se ha fatto qualcosa (scaricato o estratto), False se
    l'asset era gia' completo -- e' cosi' che step_ensure_assets
    (setup/__main__.py) sa se loggare "gia' presente" o il riepilogo di cosa e'
    stato scaricato.

    Per `kind == "block"` delega a `_ensure_block_asset`: le parti
    non sono archivi indipendenti, vanno riconcatenate prima di poter essere
    aperte con `tarfile`. Il ramo sotto resta quello di sempre per
    `kind == "archive"`.
    """
    dest_dir = paths.root / asset.dest
    cache_dir = paths.internal / "_e" / "downloads" / asset.name
    cache_dir.mkdir(parents=True, exist_ok=True)

    if asset.kind == "block":
        return _ensure_block_asset(asset, dest_dir, cache_dir, log, opener)

    changed = False
    for index, part in enumerate(asset.parts):
        archive_path = cache_dir / _part_filename(part, index)
        marker = archive_path.parent / (archive_path.name + ".extracted")

        already_downloaded = archive_path.exists() and sha256_of(archive_path) == part.sha256
        if not already_downloaded:
            download_verified(part, archive_path, log, opener=opener, asset_name=asset.name)
            marker.unlink(missing_ok=True)
            changed = True

        if not marker.exists():
            if log:
                log.info("estraggo %s in %s", archive_path.name, dest_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            # filter="data" (Python 3.12+, PEP 706) rifiuta percorsi assoluti
            # e ".." nell'archivio: senza, un archivio compromesso -- gli URL
            # del manifest sono l'unica cosa che, il giorno in cui una parte
            # viene sostituita, sta fra l'utente e una scrittura arbitraria
            # sul suo disco -- potrebbe scrivere fuori da dest_dir.
            with tarfile.open(archive_path) as tf:
                tf.extractall(dest_dir, filter="data")
            marker.write_text("")
            changed = True

    return changed
