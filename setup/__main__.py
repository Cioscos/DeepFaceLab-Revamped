"""CLI dell'installer: gli otto passi e la loro orchestrazione.

Ogni passo riceve InstallPaths, gli argomenti della CLI e il logger, e non sa
nulla degli altri: e' questo modulo a deciderne l'ordine e a fermarsi al primo
che fallisce. Ciascuno vive nel proprio modulo: step_preflight in
setup/preflight.py, step_sync_repo/step_create_venv/step_install_requirements
in setup/repo.py e setup/runtime.py, step_check_linux_gui_prereqs in
setup/prerequisiti_linux.py, step_ensure_assets in setup/assets.py,
step_layout in setup/layout.py. Fa eccezione step_verify, implementato qui
direttamente: vedi il suo docstring. Chi aggiunge un passo tocca solo il corpo
dei singoli step_*, non parse_args ne' main.

Otto funzioni per sei compiti: i compiti sono sei, ma "runtime" copre tre
funzioni separate (create_venv, install_requirements, gui_prereqs_linux).
SPEC_TASKS sotto elenca i sei nomi e _STEPS deve coprirli tutti: e' cosi'
che un passo sparito -- come "verify" e' sparito nella prima stesura di
questo file -- si nota invece di passare inosservato.

--dry-run elenca i passi e ritorna senza chiamare setup_logging: e' il modo
per provare la CLI nei test senza scaricare 2.4 GB e senza scrivere nemmeno
il file di log.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# install.sh/install.bat invocano questo file come percorso nudo
# ("uv run ... $SETUP/__main__.py"), non come "python -m setup": sys.path[0]
# diventa quindi setup/ e non la radice del repo che lo contiene, e
# "import setup.assets" fallisce con ModuleNotFoundError: No module named
# 'setup' -- verificato per davvero lanciando install.sh su un clone pulito,
# prima di aggiungere questa riga.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from setup.assets import Asset, asset_is_complete, ensure_asset, load_manifest, manifest_for  # noqa: E402
from setup.commands import SHORTCUTS  # noqa: E402
from setup.layout import create_workspace, install_setenv, place_scripts, write_shortcuts  # noqa: E402
from setup.log import setup_logging  # noqa: E402
from setup.paths import InstallPaths, resolve  # noqa: E402
from setup.preflight import (  # noqa: E402
    MIN_DRIVER_LINUX,
    MIN_DRIVER_WINDOWS,
    NEEDED_BYTES,
    check_disk_space,
    check_platform,
    choose_wheel_set,
    detect_gpu,
)
from setup.prerequisiti_linux import (  # noqa: E402
    comando_installazione,
    diagnosi,
    famiglia_distribuzione,
    librerie_mancanti,
    percorso_plugin_xcb,
)
from setup.repo import sync_repo  # noqa: E402
from setup.runtime import create_venv, install_requirements  # noqa: E402


def step_preflight(paths: InstallPaths, args: argparse.Namespace, log) -> None:
    """Rileva la GPU e sceglie la wheel torch (setup/preflight.py).

    setup/preflight.py non stampa nulla: `choose_wheel_set` restituisce solo
    "cuda" o "cpu". Qui si compone il messaggio per l'utente e si
    scrive la scelta su `args.wheel_set`: e' cosi' che `step_install_
    requirements` sa quale file installare senza rifare la
    rilevazione da capo -- `args` e' lo stesso oggetto passato a ogni passo
    da `main()`.
    """
    check_platform()  # il piu' economico dei tre controlli: va per primo
    check_disk_space(paths, NEEDED_BYTES)

    gpu = detect_gpu()
    wheel = choose_wheel_set(gpu, args.cpu)
    args.wheel_set = wheel

    if not gpu.present:
        log.info(
            "Nessuna GPU NVIDIA rilevata (o nvidia-smi assente): si installera' "
            "la wheel CPU. L'addestramento sara' molto piu' lento di quello su "
            "GPU."
        )
    elif wheel == "cpu" and args.cpu:
        log.info(
            "--cpu richiesto: si installera' la wheel CPU anche se e' presente "
            "%s (driver %s).", gpu.name, gpu.driver_version,
        )
    elif wheel == "cpu":
        floor = MIN_DRIVER_WINDOWS if sys.platform == "win32" else MIN_DRIVER_LINUX
        log.error(
            "GPU %s rilevata, ma il driver installato (%s) e' piu' vecchio del "
            "minimo richiesto (%s) per la build CUDA di torch: si procede con "
            "la wheel CPU. Aggiorna il driver da "
            "https://www.nvidia.com/Download/index.aspx e rilancia l'installer "
            "per usare la GPU.",
            gpu.name, gpu.driver_version, floor,
        )
    else:
        log.info(
            "GPU %s rilevata, driver %s: si installera' la wheel CUDA.",
            gpu.name, gpu.driver_version,
        )


def step_sync_repo(paths: InstallPaths, args: argparse.Namespace, log) -> None:
    """Clona o aggiorna _internal/DeepFaceLab (setup/repo.py)."""
    sync_repo(paths, log)


def step_create_venv(paths: InstallPaths, args: argparse.Namespace, log) -> None:
    """Crea il venv con uv (setup/runtime.py)."""
    create_venv(paths, log)


def step_install_requirements(paths: InstallPaths, args: argparse.Namespace, log) -> None:
    """Installa base.txt + cuda.txt/cpu.txt nel venv (setup/runtime.py).

    args.wheel_set e' scritto da step_preflight: i passi non si
    parlano fra loro se non tramite lo stesso oggetto args passato da
    main() a ognuno, cosi' questo passo non deve rifare la rilevazione GPU.
    """
    install_requirements(paths, args.wheel_set, log)


def _venv_site_packages(paths: InstallPaths, runner) -> Path | None:
    """Il site-packages del venv appena costruito, chiesto al suo stesso
    interprete (sysconfig.get_paths()['purelib']): e' l'unico modo corretto
    di trovarlo, perche' il nome della cartella (python3.11/site-packages,
    Lib/site-packages...) varia per piattaforma."""
    esito = runner(
        [str(paths.python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        capture_output=True, text=True,
    )
    if esito.returncode != 0:
        return None
    riga = esito.stdout.strip()
    return Path(riga) if riga else None


def step_check_linux_gui_prereqs(
    paths: InstallPaths, args: argparse.Namespace, log,
    runner=subprocess.run, which=shutil.which,
) -> None:
    """Avvisa se mancano le librerie di sistema che il plugin Qt "xcb" carica
    fuori dal wheel pyqt5 (setup/prerequisiti_linux.py).

    Solo Linux: su Windows PyQt5 porta gia' tutto cio' che le sue interfacce
    servono. Va dopo step_install_requirements, perche' prima non esiste
    ancora un venv con pyqt5 dentro da controllare, e non e' fatale: un
    'installazione dedicata solo al training su una macchina senza schermo
    resta valida anche se nessuna interfaccia grafica si apre mai. Con
    'apt' e in una sessione interattiva si propone di eseguire subito il
    comando di installazione; altrimenti si stampa e si prosegue.
    """
    if not sys.platform.startswith("linux"):
        return

    site_packages = _venv_site_packages(paths, runner)
    if site_packages is None:
        return

    messaggio = diagnosi(site_packages, runner=runner, which=which)
    if messaggio is None:
        return
    log.warning(messaggio)

    famiglia = famiglia_distribuzione(runner, which)
    mancanti = librerie_mancanti(percorso_plugin_xcb(site_packages), runner)
    comando = comando_installazione(mancanti, famiglia)
    if comando is None:
        return
    if not sys.stdin.isatty():
        log.info("esegui a mano quando vuoi: %s", comando)
        return

    # L'assistenza interattiva e' un extra, non il controllo in se': un
    # Ctrl+D al prompt (EOFError) o un 'sudo' assente dal PATH
    # (FileNotFoundError da runner) non devono far dichiarare fallito un
    # passo che, fino a qui, ha gia' fatto il suo lavoro (l'avviso e' gia'
    # stato registrato sopra). Qualunque eccezione qui dentro si logga e si
    # ricade sul comando stampato, esattamente come nel ramo non
    # interattivo.
    try:
        risposta = input(f"Eseguire ora '{comando}'? [s/N] ")
        if risposta.strip().lower() not in ("s", "si", "sì", "y", "yes"):
            log.info("comando non eseguito: puoi lanciarlo a mano in seguito: %s", comando)
            return

        runner(comando.split(), check=False)
        dopo = diagnosi(site_packages, runner=runner, which=which)
        if dopo is None:
            log.info("librerie del plugin Qt xcb installate correttamente.")
        else:
            log.warning(dopo)
    except Exception as exc:
        log.info(
            "assistenza interattiva non riuscita (%s): esegui a mano quando vuoi: %s",
            exc, comando,
        )


def _should_install_pretrain(asset: Asset, paths: InstallPaths, args: argparse.Namespace, log) -> bool:
    """Decide se scaricare pretrain_faces: l'unica domanda che l'installer
    pone, posta una sola volta. Quattro controlli, in quest'ordine,
    e il primo che si applica decide:

    1. --with-pretrain/--no-pretrain gia' scelto (args.with_pretrain non e'
       None): si rispetta senza chiedere.
    2. l'asset e' gia' completo (`asset_is_complete`, setup/assets.py): il
       lavoro e' gia' fatto, chiedere non cambierebbe nulla sul disco. Nessun
       ricalcolo di sha256 qui -- solo il marcatore di estrazione: quel
       calcolo lo fa gia' `ensure_asset` stesso se poi si procede davvero
       (1.8 GB di hashing due volte a ogni rilancio).
    3. --yes: risponde "si'" a ogni domanda interattiva senza porla.
    4. altrimenti, la domanda vera -- l'unica riga di questa funzione che
       tocca stdin.
    """
    if args.with_pretrain is not None:
        return args.with_pretrain
    if asset_is_complete(asset, paths):
        log.info("pretrain_faces gia' presente e verificato: nessuna domanda necessaria")
        return True
    if args.yes:
        log.info("--yes: scarico pretrain_faces senza chiedere")
        return True
    answer = input(
        "Scaricare pretrain_faces adesso? E' un modello pre-addestrato "
        "(1.8 GB) che accelera l'avvio di un nuovo training; puoi "
        "aggiungerlo in un secondo momento rilanciando l'installer con "
        "--with-pretrain. [s/N] "
    )
    return answer.strip().lower() in ("s", "si", "sì", "y", "yes")


def step_ensure_assets(paths: InstallPaths, args: argparse.Namespace, log) -> None:
    """Scarica e verifica gli asset del manifest (setup/assets.py).

    Gli asset obbligatori (facelib, ffmpeg, model_generic_xseg) si
    scaricano sempre. pretrain_faces e'
    l'unico opzionale soggetto a --with-pretrain/--no-pretrain:
    `_should_install_pretrain` decide se scaricarlo, con la
    domanda interattiva per il caso non ancora deciso da nessun flag. EbSynth
    e' opzionale e solo Windows: si scarica da solo, senza
    bisogno di un flag.

    Con setup/manifest.toml ancora ai segnaposto del CHECKPOINT 1 (nessuna
    release caricata), ogni ensure_asset qui sotto fallisce all'apertura
    dell'URL: un errore di rete esplicito e immediato, non un file scaricato
    e verificato contro uno sha256 farlocco.

    L'elenco viene da `manifest_for(paths)`, non da `DEFAULT_MANIFEST`: il
    manifest che comanda e' quello del clone appena sincronizzato -- `repo`
    e' il passo prima di `assets` in `_STEPS`, quindi qui e' gia' aggiornato
    -- e non quello della copia esterna, che nessuno aggiorna mai. Vedi
    `setup/assets.py::manifest_for` per il difetto che questo chiude.
    """
    for asset in load_manifest(manifest_for(paths)):
        if not asset.required:
            if asset.name == "pretrain_faces":
                if not _should_install_pretrain(asset, paths, args, log):
                    log.info("salto pretrain_faces: non richiesto (--with-pretrain per includerlo)")
                    continue
            elif asset.name == "EbSynth" and sys.platform != "win32":
                log.info("salto EbSynth: non disponibile su questa piattaforma")
                continue

        if ensure_asset(asset, paths, log):
            log.info("asset '%s' pronto in %s", asset.name, paths.root / asset.dest)
        else:
            log.info("asset '%s' gia' presente e verificato", asset.name)


def step_layout(paths: InstallPaths, args: argparse.Namespace, log) -> None:
    """Posa gli script utente, le scorciatoie e workspace/ (setup/layout.py).

    install_setenv: senza, ogni script generato da place_scripts e
    ogni scorciatoia da write_shortcuts chiama un _internal/setenv.bat|sh che
    non esiste, e fallisce alla prima riga eseguibile.
    """
    install_setenv(paths, log)
    place_scripts(paths, log)
    write_shortcuts(paths, log)
    create_workspace(paths, log)


def _asset_status_lines(paths: InstallPaths) -> list[str]:
    """Una riga per asset del manifest: presente o assente sul disco.

    "Presente" e' deciso da `asset_is_complete` (setup/assets.py, stesso
    marcatore di estrazione che `_should_install_pretrain` usa per la
    domanda su pretrain_faces), non da "la cartella di destinazione non e'
    vuota": per `facelib` quella cartella e' dentro l'albero del repo
    clonato e contiene gia' sette file `.py` tracciati da git a prescindere
    dai pesi `.npy` -- un controllo sulla cartella direbbe sempre
    "presente", proprio sull'unico asset obbligatorio la cui assenza
    impedisce all'estrazione di funzionare del tutto. Nessun ricalcolo di
    sha256 qui: e' un riepilogo, non un secondo
    controllo di integrita'.
    """
    lines = []
    for asset in load_manifest(manifest_for(paths)):
        dest_dir = paths.root / asset.dest
        present = asset_is_complete(asset, paths)
        lines.append(f"  - {asset.name}: {'presente' if present else 'assente'} ({dest_dir})")
    return lines


def step_verify(paths: InstallPaths, args: argparse.Namespace, log, runner=subprocess.run) -> None:
    """Verifica l'installazione e riepiloga a schermo.

    E' l'unico passo il cui compito e' dire all'utente se ha funzionato.
    Tre cose, in quest'ordine, nessuna opzionale: 1 e 2 sono verifiche che
    possono fallire e devono farlo rumorosamente; 3 e' il riepilogo che le
    riassume, e non ha senso comporlo se 1 o 2 non sono passate.

    1. `<venv>/python -c "import torch; print(torch.__version__,
       torch.cuda.is_available())"` -- torch si importa, e vede (o no) la
       GPU;
    2. `<venv>/python <repo>/main.py --help` -- la CLI dell'applicazione
       parte con l'ambiente appena costruito;
    3. il riepilogo a schermo: dove e' installato, quale wheel (CUDA/CPU) e
       perche', quali asset ci sono e quali no, il comando per cominciare.

    Se args.wheel_set e' "cuda" (preflight aveva trovato una GPU con un
    driver abbastanza recente e la CUDA wheel e' stata installata di
    conseguenza) ma torch.cuda.is_available() torna False, e' la
    discrepanza piu' importante che l'installer possa incontrare -- la wheel
    installata non parla col driver presente -- e va detta forte
    (log.error), non infilata fra le righe del riepilogo. Se invece la CPU
    wheel e' stata scelta deliberatamente (nessuna GPU, driver troppo
    vecchio, o --cpu), False e' il risultato atteso: step_preflight lo ha
    gia' spiegato prima, e qui non e' un errore.
    """
    check = runner(
        [str(paths.python), "-c",
         "import torch; print(torch.__version__, torch.cuda.is_available())"],
        capture_output=True, text=True,
    )
    if check.returncode != 0:
        raise RuntimeError(
            f"'{paths.python} -c \"import torch\"' e' fallito (codice "
            f"{check.returncode}): {check.stderr.strip()}\nPerche': il venv "
            f"in {paths.venv} non ha torch installato o non parte. Cosa "
            "fare: controlla che i passi 'runtime' (venv + requirements) "
            "qui sopra siano andati a buon fine, poi rilancia l'installer."
        )
    torch_version, _, cuda_flag = check.stdout.strip().rpartition(" ")
    cuda_available = cuda_flag == "True"

    if args.wheel_set == "cuda" and not cuda_available:
        log.error(
            "torch %s e' installato ma torch.cuda.is_available() e' False, "
            "nonostante preflight avesse scelto la wheel CUDA: la wheel "
            "installata non parla col driver NVIDIA presente su questa "
            "macchina. Verifica il driver con 'nvidia-smi' e, se necessario, "
            "aggiornalo da https://www.nvidia.com/Download/index.aspx; nel "
            "frattempo puoi rilanciare l'installer con --cpu per usare la "
            "CPU.",
            torch_version,
        )
    else:
        log.info("torch %s, CUDA disponibile: %s", torch_version, cuda_available)

    main_help = runner(
        [str(paths.python), str(paths.repo / "main.py"), "--help"],
        capture_output=True, text=True,
    )
    if main_help.returncode != 0:
        raise RuntimeError(
            f"'{paths.python} {paths.repo / 'main.py'} --help' e' fallito "
            f"(codice {main_help.returncode}): {main_help.stderr.strip()}\n"
            "Perche': l'ambiente Python appena costruito non riesce a "
            "importare l'applicazione. Cosa fare: controlla che i passi "
            "'repo' e 'requirements' qui sopra siano andati a buon fine, poi "
            "rilancia l'installer."
        )

    shortcut = SHORTCUTS[0] + (".bat" if sys.platform == "win32" else ".sh")
    log.info("")
    log.info("=== Installazione completata in %s ===", paths.root)
    log.info("wheel torch: %s (torch %s, CUDA disponibile: %s)", args.wheel_set, torch_version, cuda_available)
    if args.wheel_set == "cuda":
        log.info("  motivo: GPU NVIDIA rilevata con driver sufficiente per la build CUDA.")
    elif args.cpu:
        log.info("  motivo: --cpu richiesto esplicitamente.")
    else:
        log.info("  motivo: nessuna GPU NVIDIA rilevata, o driver troppo vecchio per la build CUDA "
                  "(vedi i messaggi del passo 'preflight' qui sopra).")
    log.info("asset:")
    for line in _asset_status_lines(paths):
        log.info(line)
    log.info("")
    log.info("Per cominciare: %s", paths.root / shortcut)


# I sei compiti dell'installer. _STEPS sotto ha OTTO voci perche' "runtime"
# e' diviso in tre funzioni, ma deve coprire esattamente questi sei nomi,
# ciascuno almeno una volta.
SPEC_TASKS = ("preflight", "repo", "runtime", "assets", "layout", "verify")

# Ordine di esecuzione: ogni voce e' (nome per il log e per --dry-run,
# compito a cui corrisponde, funzione).
_STEPS = (
    ("preflight", "preflight", step_preflight),
    ("repo", "repo", step_sync_repo),
    ("venv", "runtime", step_create_venv),
    ("requirements", "runtime", step_install_requirements),
    ("gui_prereqs_linux", "runtime", step_check_linux_gui_prereqs),
    ("assets", "assets", step_ensure_assets),
    ("layout", "layout", step_layout),
    ("verify", "verify", step_verify),
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dfl-install",
        description="Installa DeepFaceLab: Python, PyTorch, script utente e asset.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path.cwd(),
        help="cartella di installazione (default: la cartella corrente)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="forza la wheel CPU anche se una GPU NVIDIA e' presente",
    )
    pretrain = parser.add_mutually_exclusive_group()
    pretrain.add_argument(
        "--with-pretrain",
        dest="with_pretrain",
        action="store_true",
        help="include pretrain_faces (1.8 GB) senza chiedere",
    )
    pretrain.add_argument(
        "--no-pretrain",
        dest="with_pretrain",
        action="store_false",
        help="salta pretrain_faces senza chiedere",
    )
    # None e' "non deciso": produce la domanda a schermo. Un
    # default booleano la renderebbe irraggiungibile.
    parser.set_defaults(with_pretrain=None)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="non chiede conferma: risponde si' a ogni domanda interattiva",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="elenca i passi e termina, senza scrivere ne' scaricare nulla",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    paths = resolve(args.dest)

    if args.dry_run:
        for name, _spec_task, _step in _STEPS:
            print(f"[dry-run] {name}")
        return 0

    log = setup_logging(paths.log_file)
    for name, _spec_task, step in _STEPS:
        try:
            step(paths, args, log)
        except Exception as exc:
            log.error("passo '%s' fallito: %s", name, exc)
            # Il traceback completo va solo sul file (DEBUG): lo stream e'
            # INFO e un traceback li' sommergerebbe il riepilogo breve che
            # l'utente guarda mentre l'installazione procede. Ma e' esattamente
            # cio' che serve a chi deve capire dopo, su una macchina che non
            # e' la propria (setup/log.py) -- senza, il file di log ha lo
            # stesso identico messaggio breve dello schermo e la sua ragion
            # d'essere (spiegare un fallimento altrove) non e' onorata.
            log.debug("traceback completo del passo '%s':", name, exc_info=True)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
