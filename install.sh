#!/usr/bin/env bash
# Bootstrap dell'installer DeepFaceLab: procura uv, individua o clona il
# repo, cede il controllo a setup/__main__.py. Nessun
# prerequisito oltre a curl/tar, gia' presenti su ogni distribuzione comune.
set -eu

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Le cinque UV_* confinano uv dentro il pacchetto: nessuna scrittura nel
# profilo utente, nessuna modifica al PATH di sistema.
# UV_MANAGED_PYTHON non e' un doppione di UV_PYTHON_INSTALL_DIR: la seconda
# dice soltanto DOVE mettere un Python scaricato, non obbliga a scaricarne
# uno. Senza la prima, su una macchina che un 3.11 ce l'ha gia' (pyenv, i
# pacchetti della distribuzione) uv riusa quello: _internal/python resta
# inesistente e l'installazione punta a un interprete fuori dal pacchetto,
# che nessuno qui aggiorna e che l'utente puo' disinstallare senza sapere
# cosa rompe. Misurato su Windows: senza, pyvenv.cfg finiva su
# C:\Users\<utente>\.pyenv\pyenv-win\versions\3.11.9; con, uv scarica
# CPython 3.11.15 in _internal/python e pyvenv.cfg punta li'.
export UV_INSTALL_DIR="$ROOT/_internal/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/_internal/python"
export UV_CACHE_DIR="$ROOT/_internal/_e/uv-cache"
export UV_NO_MODIFY_PATH=1
export UV_MANAGED_PYTHON=1

# setup/paths.py::resolve calcola uv_bin sotto --dest per il caso comune
# (--dest assente o uguale a questa cartella), ma il binario di uv qui sopra
# viene sempre scaricato accanto a QUESTO script, non a --dest: i due
# divergono se l'utente passa un --dest esplicito verso un'altra cartella
# (caso supportato -- verificato che si rompe per davvero senza questa
# riga: "[Errno 2] No such file or directory:
# '<dest>/_internal/uv/uv'"). DFL_UV_BIN dice a setup/__main__.py dove uv sta
# davvero, senza che Python debba indovinarlo da --dest.
export DFL_UV_BIN="$UV_INSTALL_DIR/uv"

REPO_URL="https://github.com/Cioscos/DeepFaceLab-Revamped.git"

# Passo 1: OS e architettura supportati, PRIMA di scaricare qualunque cosa.
# uv, il CPython standalone e le wheel di PyTorch scaricate piu' avanti sono
# tutte x86_64: su un'altra architettura (ARM64/aarch64 e' il caso reale --
# Raspberry Pi 64 bit, server ARM, Apple Silicon sotto Linux) il passo 3 qui
# sotto scaricherebbe comunque il tarball uv-x86_64-unknown-linux-gnu.tar.gz,
# e la riga "exec ... uv run" in fondo a questo script fallirebbe con "Exec
# format error" da bash stesso -- prima ancora che una riga di Python giri,
# quindi prima che setup/preflight.py::check_platform (che fa lo stesso
# controllo, SUPPORTED_MACHINES) abbia la minima possibilita' di dirlo con
# una causa nominata. Il controllo va ripetuto qui in bash per lo stesso
# motivo per cui il passo 2 sotto ripete il controllo di setup/__main__.py
# su "setup/__main__.py" mancante: la causa vera va detta nel punto in cui
# il fallimento avverrebbe altrimenti, non tre passi dopo.
MACHINE="$(uname -m)"
case "$MACHINE" in
    x86_64|amd64) ;;
    *)
        echo "[install] architettura non supportata: '$MACHINE' (serve x86_64/amd64). uv, il Python standalone e le wheel PyTorch che questo script scarica sono compilati solo per x86_64: su '$MACHINE' l'installazione fallirebbe piu' avanti con un errore muto ('Exec format error') invece che qui, con la causa nominata. Serve una macchina x86_64."
        exit 1
        ;;
esac

# Passo 2: dove sta setup/__main__.py. Due posizioni, in quest'ordine:
# accanto a install.sh (dentro un clone del repo) e dentro
# _internal/DeepFaceLab (installazione gia' presente, si sta rilanciando per
# aggiornare). Se nessuna delle due, e' il primo avvio: si clona.
SETUP=""
if [ -f "$ROOT/setup/__main__.py" ]; then
    SETUP="$ROOT/setup"
elif [ -f "$ROOT/_internal/DeepFaceLab/setup/__main__.py" ]; then
    SETUP="$ROOT/_internal/DeepFaceLab/setup"
fi

if [ -z "$SETUP" ]; then
    echo "[install] setup non trovato accanto a install.sh ne' in _internal/DeepFaceLab: clono $REPO_URL"
    if ! command -v git >/dev/null 2>&1; then
        echo "[install] git non trovato nel PATH: installalo con il gestore pacchetti della tua distribuzione (es. 'sudo apt install git' su Debian/Ubuntu, 'sudo dnf install git' su Fedora) e rilancia install.sh"
        exit 1
    fi
    mkdir -p "$ROOT/_internal"
    git clone --depth 1 "$REPO_URL" "$ROOT/_internal/DeepFaceLab"
    SETUP="$ROOT/_internal/DeepFaceLab/setup"
    # setup/repo.py::sync_repo fa lo stesso controllo dopo ogni clone/pull,
    # ma solo una volta che Python gira -- qui non gira ancora, quindi il
    # controllo va ripetuto in bash. Senza, un clone sul branch sbagliato (o
    # corrotto) fallirebbe dentro "uv run" con un errore che non nomina la
    # causa vera.
    if [ ! -f "$SETUP/__main__.py" ]; then
        echo "[install] il clone in $ROOT/_internal/DeepFaceLab non contiene setup/__main__.py: il branch di default di questo repository potrebbe non includere ancora l'installer, oppure il clone e' corrotto o parziale. Verifica quale branch contiene setup/, requirements/ e scripts/commands.toml, clonalo a mano con 'git clone -b <branch> $REPO_URL', e rilancia install.sh da dentro quel clone."
        exit 1
    fi
fi

# Passo 3: procurarsi uv, se non c'e' gia' da un giro precedente. La
# versione e' fissata nell'URL, non in una variabile: un cambio a monte non
# deve rompere le installazioni esistenti. Aggiornata
# deliberatamente.
if [ ! -x "$UV_INSTALL_DIR/uv" ]; then
    echo "[install] scarico uv 0.12.1..."
    mkdir -p "$UV_INSTALL_DIR"
    UV_TARBALL="$UV_INSTALL_DIR/_download.tar.gz"
    curl -LsSf -o "$UV_TARBALL" "https://github.com/astral-sh/uv/releases/download/0.12.1/uv-x86_64-unknown-linux-gnu.tar.gz"
    tar -xzf "$UV_TARBALL" -C "$UV_INSTALL_DIR" --strip-components=1
    rm -f "$UV_TARBALL"
    chmod +x "$UV_INSTALL_DIR/uv" "$UV_INSTALL_DIR/uvx"
fi

# Passo 4: cedere il controllo a setup/__main__.py, che da qui in poi e'
# tutto Python e identico sui due sistemi. Nessun "pause" qui: a
# differenza di install.bat, un terminale Linux non si chiude da solo alla
# fine di uno script, e set -eu ha gia' fatto uscire con codice != 0 su
# qualunque comando fallito sopra.
#
# --dest "$ROOT" prima di "$@": setup/__main__.py::parse_args ha --dest con
# default Path.cwd(), non la cartella di questo script, quindi senza questo
# argomento esplicito "cd altrove; bash /percorso/install.sh" installerebbe
# nella cwd del chiamante invece che accanto a install.sh (un `cd "$ROOT"`
# qui sopra risolveva lo stesso problema ma
# rompeva un --dest UNC/relativo passato dall'utente e divergeva dalla
# convenzione del pacchetto, dove nessun .bat/.sh cambia mai directory).
# Messo PRIMA di "$@": argparse fa vincere l'ultima occorrenza di
# un'opzione a valore singolo, quindi un --dest esplicito dell'utente (che
# arriva dopo, via "$@") sovrascrive questo senza bisogno di logica in piu'
# -- verificato con setup.__main__.parse_args(["--dest", "A", "--dest",
# "B"]).dest == Path("B").
exec "$UV_INSTALL_DIR/uv" run --python 3.11 --no-project "$SETUP/__main__.py" --dest "$ROOT" "$@"
