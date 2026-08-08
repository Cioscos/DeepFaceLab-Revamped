"""Le librerie di sistema che il plugin Qt "xcb" cerca fuori dal wheel pyqt5.

Su Linux il wheel `pyqt5` porta Qt e i suoi plugin, ma il plugin di
piattaforma "xcb" -- quello che ogni finestra di questo pacchetto usa per
aprirsi su X11 -- si appoggia a librerie X11/xcb/xkbcommon che il sistema
operativo deve gia' avere installate: il wheel non le porta con se'. Se
mancano, Qt non riesce a caricare il plugin e nessuna finestra si apre, pur
con un'installazione altrimenti riuscita.

Tutto qui dentro e' pure stdlib (os, shutil, subprocess, pathlib, re) e non
importa mai pyqt5 o torch: e' un controllo *sul sistema*, non
sull'applicazione, e deve poter girare anche in un ambiente che pyqt5 non
lo ha ancora installato per niente. `ldd` e i gestori di pacchetto si
invocano sempre tramite un `runner`/`which` iniettabili, cosi' il
rilevamento si esercita senza dipendere da cosa e' installato sulla
macchina che lo esegue.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

# La tabella soname -> pacchetto Debian/Ubuntu, misurata su un'installazione
# reale con `ldd .../PyQt5/Qt5/plugins/platforms/libqxcb.so` e ogni soname
# risolto a pacchetto con `dpkg -S` sul target del symlink. Diciotto voci: e'
# l'elenco completo di cio' che quel plugin risolve fuori dal wheel, non un
# sottoinsieme.
SONAME_PACCHETTO_APT: dict[str, str] = {
    "libX11.so.6": "libx11-6",
    "libX11-xcb.so.1": "libx11-xcb1",
    "libxcb.so.1": "libxcb1",
    "libxcb-icccm.so.4": "libxcb-icccm4",
    "libxcb-image.so.0": "libxcb-image0",
    "libxcb-keysyms.so.1": "libxcb-keysyms1",
    "libxcb-randr.so.0": "libxcb-randr0",
    "libxcb-render.so.0": "libxcb-render0",
    "libxcb-render-util.so.0": "libxcb-render-util0",
    "libxcb-shape.so.0": "libxcb-shape0",
    "libxcb-shm.so.0": "libxcb-shm0",
    "libxcb-sync.so.1": "libxcb-sync1",
    "libxcb-util.so.1": "libxcb-util1",
    "libxcb-xfixes.so.0": "libxcb-xfixes0",
    "libxcb-xinerama.so.0": "libxcb-xinerama0",
    "libxcb-xkb.so.1": "libxcb-xkb1",
    "libxkbcommon.so.0": "libxkbcommon0",
    "libxkbcommon-x11.so.0": "libxkbcommon-x11-0",
}


def percorso_plugin_xcb(site_packages: Path) -> Path | None:
    """Dove il wheel pyqt5 mette il plugin xcb dentro un site-packages dato.

    None se il file non esiste: significa che pyqt5 non e' (ancora)
    installato li', un caso diverso da "le librerie di sistema mancano" e
    che questo modulo non ha nulla da dire su di esso.
    """
    plugin = Path(site_packages) / "PyQt5" / "Qt5" / "plugins" / "platforms" / "libqxcb.so"
    return plugin if plugin.exists() else None


def librerie_mancanti(plugin: Path, runner=subprocess.run) -> list[str]:
    """I soname che `ldd <plugin>` segna come "not found", ordinati.

    Lista vuota anche quando `ldd` non e' nel PATH: qui non si puo'
    distinguere quel caso da "nessuna libreria manca", perche' entrambi
    producono la stessa lista vuota. E' compito di chi chiama (`diagnosi`
    sotto) controllare la presenza di `ldd` per conto proprio *prima* di
    leggere questo risultato come "va tutto bene".
    """
    if shutil.which("ldd") is None:
        return []
    esito = runner(["ldd", str(plugin)], capture_output=True, text=True)
    mancanti = set()
    for riga in esito.stdout.splitlines():
        if "not found" in riga:
            soname = riga.strip().split()[0]
            if soname:
                mancanti.add(soname)
    return sorted(mancanti)


def famiglia_distribuzione(runner=subprocess.run, which=shutil.which) -> str:
    """"apt" / "dnf" / "pacman" / "zypper" / "sconosciuta".

    Rilevata dalla presenza del gestore pacchetti sul PATH, non dal
    contenuto di /etc/os-release: e' cio' che decide se un comando di
    installazione e' eseguibile su questa macchina, indipendentemente da
    quale distribuzione dichiara di essere. `runner` non serve a questa
    rilevazione (nessun sottoprocesso viene lanciato) ma resta nella firma
    per coerenza con le altre funzioni del modulo, che invece lo usano
    davvero.
    """
    del runner
    for comando, famiglia in (
        ("apt-get", "apt"),
        ("dnf", "dnf"),
        ("pacman", "pacman"),
        ("zypper", "zypper"),
    ):
        if which(comando) is not None:
            return famiglia
    return "sconosciuta"


def comando_installazione(soname_mancanti: list[str], famiglia: str) -> str | None:
    """Il comando per installare le librerie mancanti, o None.

    Solo per "apt": e' l'unica famiglia per cui la tabella misurata sopra
    esiste. Per le altre si torna None piuttosto che indovinare un nome di
    pacchetto mai verificato -- `diagnosi` sa comporre un messaggio utile
    anche senza un comando esatto.
    """
    if famiglia != "apt":
        return None
    pacchetti = sorted({
        SONAME_PACCHETTO_APT[soname]
        for soname in soname_mancanti
        if soname in SONAME_PACCHETTO_APT
    })
    if not pacchetti:
        return None
    return "sudo apt install -y " + " ".join(pacchetti)


def _messaggio_ricerca_pacchetto(famiglia: str) -> str:
    """Il ripiego di `diagnosi` quando non c'e' un comando pronto da
    suggerire: come cercare a mano il pacchetto giusto, senza indovinarne il
    nome. Lo strumento di ricerca cambia per famiglia -- suggerire `dnf
    provides`/`pacman -F` a chi ha apt sarebbe un comando che sulla sua
    macchina semplicemente non esiste, tanto quanto un nome di pacchetto
    inventato.
    """
    if famiglia == "apt":
        return (
            "nessuno dei pacchetti noti copre tutte le librerie mancanti: "
            "cerca il pacchetto che fornisce ciascuna libreria elencata sopra "
            "con `apt-file search <soname>` (richiede 'apt-file update' una "
            "tantum) o `dpkg -S <soname>` se il file risulta gia' "
            "installato da qualche altro pacchetto (sostituendo <soname> con "
            "ciascun nome dell'elenco)."
        )
    return (
        "nessun comando di installazione automatico e' disponibile per "
        "questa distribuzione: cerca il pacchetto che fornisce ciascuna "
        "delle librerie elencate sopra con lo strumento della tua "
        "distribuzione, ad esempio `dnf provides '*/<soname>'` o "
        "`pacman -F <soname>` (sostituendo <soname> con ciascun nome "
        "dell'elenco)."
    )


def diagnosi(site_packages: Path, runner=subprocess.run, which=shutil.which) -> str | None:
    """Il messaggio per l'utente, o None se non c'e' nulla da segnalare.

    None copre due casi distinti sul piano tecnico ma identici su quello
    dell'utente: pyqt5 non e' installato li' (niente da controllare) e
    pyqt5 e' installato con tutte le librerie di sistema che gli servono
    (il controllo e' stato fatto ed e' andato bene). Se invece `ldd` manca
    dal PATH il controllo non puo' essere eseguito affatto, e questo va
    detto esplicitamente invece di restituire None come se fosse andato
    tutto bene.
    """
    plugin = percorso_plugin_xcb(site_packages)
    if plugin is None:
        return None

    if which("ldd") is None:
        return (
            "controllo delle librerie di sistema del plugin Qt \"xcb\" non "
            "eseguito: 'ldd' non e' nel PATH. Se le interfacce grafiche "
            "(main.py gui, main.py xseg editor) falliscono con \"could not "
            "load the Qt platform plugin 'xcb'\", installa 'ldd' (di solito "
            "nel pacchetto 'binutils') e ripeti il controllo."
        )

    mancanti = librerie_mancanti(plugin, runner)
    if not mancanti:
        return None

    righe = [
        "le interfacce grafiche di questo pacchetto (main.py gui, main.py "
        "xseg editor) potrebbero non aprirsi: il plugin Qt \"xcb\" ha "
        "bisogno di librerie di sistema che il wheel pyqt5 non porta con "
        "se'.",
        "librerie mancanti: " + ", ".join(mancanti),
    ]

    famiglia = famiglia_distribuzione(runner, which)
    comando = comando_installazione(mancanti, famiglia)
    if comando is not None:
        righe.append("comando suggerito: " + comando)
        senza_pacchetto = [s for s in mancanti if s not in SONAME_PACCHETTO_APT]
        if senza_pacchetto:
            righe.append(
                "nessun pacchetto noto per: " + ", ".join(senza_pacchetto)
            )
    else:
        righe.append(_messaggio_ricerca_pacchetto(famiglia))
    return "\n".join(righe)
