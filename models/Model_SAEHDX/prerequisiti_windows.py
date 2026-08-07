"""
I prerequisiti di torch.compile su Windows, resi un problema del programma.

Su Windows inductor ha bisogno di due cose che non arrivano da sole: il
backend Triton (il pacchetto triton-windows, che il wheel di torch non porta)
e il compilatore MSVC (cl.exe) nel PATH del processo. Questo modulo li
rileva, ripara il PATH quando il toolchain c'e' ma non e' caricato, e
costruisce le ricette per quando manca tutto. Solo stdlib, tutto iniettabile:
la logica si esercita ovunque, Windows serve solo per l'ultimo miglio.
"""
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

COMANDO_BUILD_TOOLS = ('winget install Microsoft.VisualStudio.2022.BuildTools '
                       '--override "--add Microsoft.VisualStudio.Workload.VCTools '
                       '--includeRecommended --passive"')
LINK_BUILD_TOOLS = "https://visualstudio.microsoft.com/visual-cpp-build-tools"


def su_windows():
    """In una funzione, non inline: i chiamanti la fingono nei collaudi."""
    return sys.platform == "win32"


def trova_vswhere(ambiente=os.environ):
    """vswhere.exe sta a un percorso fisso, documentato da Microsoft."""
    base = ambiente.get("ProgramFiles(x86)")
    if not base:
        return None
    percorso = (Path(base) / "Microsoft Visual Studio" / "Installer"
                / "vswhere.exe")
    return percorso if percorso.exists() else None


def percorso_toolchain(vswhere, esegui=subprocess.run):
    """La radice dell'installazione che ha il toolchain C++, o None."""
    esito = esegui([str(vswhere), "-latest", "-products", "*",
                    "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property", "installationPath"],
                   capture_output=True, text=True)
    if esito.returncode != 0:
        return None
    riga = esito.stdout.strip().splitlines()
    return riga[0].strip() if riga and riga[0].strip() else None


def ambiente_vcvars(vcvars, esegui=subprocess.run):
    """L'ambiente che vcvars64.bat costruisce, come dizionario.

    Il comando e' una stringa unica, non una lista: dev'essere cmd a
    interpretarla, e il quoting che subprocess produce per le liste non
    e' quello che cmd capisce quando il percorso contiene spazi. La
    decodifica tollera i byte fuori pagina: la console non e' sempre
    nella codifica che il processo si aspetta.
    """
    esito = esegui(f'cmd /c call "{vcvars}" && set',
                   capture_output=True, text=True, errors="replace")
    coppie = {}
    for riga in esito.stdout.splitlines():
        if "=" in riga:
            chiave, _, valore = riga.partition("=")
            coppie[chiave] = valore
    return coppie


def importa_ambiente(nuove, ambiente=os.environ):
    """Porta nel processo le variabili nuove o cambiate. Torna quante."""
    cambiate = 0
    for chiave, valore in nuove.items():
        if ambiente.get(chiave) != valore:
            ambiente[chiave] = valore
            cambiate += 1
    return cambiate


def ripara_path_msvc(which=shutil.which, esegui=subprocess.run,
                     ambiente=os.environ):
    """True se alla fine cl.exe e' utilizzabile, False altrimenti.

    Non installa niente: ripara solo il caso di chi il toolchain ce l'ha
    (Visual Studio o le Build Tools) ma lancia da un prompt qualunque, dove
    cl.exe non e' nel PATH. E' l'equivalente automatico di aprire un «x64
    Native Tools Command Prompt».
    """
    if which("cl"):
        return True
    vswhere = trova_vswhere(ambiente)
    if vswhere is None:
        return False
    toolchain = percorso_toolchain(vswhere, esegui)
    if toolchain is None:
        return False
    vcvars = Path(toolchain) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not vcvars.exists():
        return False
    importa_ambiente(ambiente_vcvars(vcvars, esegui), ambiente)
    return which("cl") is not None


def percorso_uv(eseguibile=sys.executable):
    """L'uv.exe dell'installazione, risalendo dal venv corrente.

    Il layout dell'installer e' _internal/.venv/Scripts/python.exe con
    _internal/uv/uv.exe accanto: due livelli sopra Scripts. Dove il layout
    e' un altro (checkout di sviluppo, venv qualunque) non c'e' niente da
    trovare, e si torna None.
    """
    try:
        radice = Path(eseguibile).parents[2]
    except IndexError:
        return None
    percorso = radice / "uv" / "uv.exe"
    return percorso if percorso.exists() else None


def ricetta_build_tools(which=shutil.which):
    """Il comando esatto se winget c'e', il link con le istruzioni se no."""
    if which("winget"):
        return COMANDO_BUILD_TOOLS
    return (f"scarica le Build Tools da {LINK_BUILD_TOOLS} e installa il "
            f"workload «Desktop development with C++»")


def offri_installazione(io_, which=shutil.which, esegui=subprocess.run,
                        trova_modulo=None, eseguibile=sys.executable,
                        ambiente=os.environ):
    """Le offerte al momento dell'accensione dell'opzione, su Windows.

    Rifiutare non cambia nulla: la guardia a valle rinuncera' con la stessa
    ricetta. Il default di ogni domanda e' No, cosi' uno stdin chiuso (avvio
    silenzioso, corsa non presidiata) non installa mai niente da solo.
    """
    if trova_modulo is None:
        trova_modulo = importlib.util.find_spec

    if trova_modulo("triton") is None:
        uv = percorso_uv(eseguibile)
        if uv is not None and io_.input_bool(
                "Install triton-windows now?", False,
                help_message="torch.compile needs the Triton backend. This "
                             "installs the triton-windows package into this "
                             "installation's own environment. No administrator "
                             "rights, a few seconds."):
            esegui([str(uv), "pip", "install", "--python", str(eseguibile),
                    "triton-windows"], check=False)
        else:
            io_.log_info("torch_compile: manca il backend Triton. Comando: "
                         "uv pip install --python "
                         f"{eseguibile} triton-windows")

    if which("cl") is None:
        vswhere = trova_vswhere(ambiente)
        toolchain = (percorso_toolchain(vswhere, esegui)
                     if vswhere is not None else None)
        if toolchain is not None:
            return  # il PATH verra' riparato in automatico, niente da installare
        if which("winget") is not None and io_.input_bool(
                "Install the MSVC Build Tools now? (UAC prompt, several GB)",
                False,
                help_message="torch.compile on Windows also needs the MSVC "
                             "compiler. This runs winget: Windows will show a "
                             "UAC prompt and download a few GB. When it "
                             "finishes, the next run finds the compiler on "
                             "its own."):
            io_.log_info("torch_compile: installazione delle Build Tools "
                         "avviata, puo' durare parecchi minuti.")
            esegui(["winget", "install",
                    "Microsoft.VisualStudio.2022.BuildTools", "--override",
                    "--add Microsoft.VisualStudio.Workload.VCTools "
                    "--includeRecommended --passive"], check=False)
        else:
            io_.log_info("torch_compile: manca il compilatore MSVC. "
                         + ricetta_build_tools(which))
