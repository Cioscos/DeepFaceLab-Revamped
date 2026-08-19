#!/usr/bin/env bash
# Bootstrap dell'installer DeepFaceLab: procura uv, trova il codice gia'
# presente in _internal/DeepFaceLab o lo scarica ed estrae da li', cede il
# controllo a setup/__main__.py. Nessun prerequisito oltre a
# curl/tar, gia' presenti su ogni distribuzione comune.
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

# L'archivio del ramo pubblicato: due megabyte e mezzo, gli stessi curl/tar
# che servono gia' per uv, e nessun sistema di controllo di versione da
# installare prima. Lo stesso indirizzo sta in setup/codice.py, che comanda
# da qui in avanti: qui serve solo per la primissima volta, quando setup/
# non esiste ancora e non c'e' nessun Python da eseguire.
URL_CODICE="https://codeload.github.com/Cioscos/DeepFaceLab-Revamped/tar.gz/refs/heads/main"

# Passo 2: il codice. Un posto solo dove cercarlo, _internal/DeepFaceLab, e
# se non c'e' lo si scarica. Cercarlo anche accanto a questo script -- come
# faceva prima -- significava installare con una copia di setup/ che nessuno
# aggiorna mai, mentre il codice eseguito e' sempre quello di _internal.
SETUP="$ROOT/_internal/DeepFaceLab/setup"
if [ ! -f "$SETUP/__main__.py" ]; then
    echo "[install] prima installazione: scarico il codice da $URL_CODICE"
    mkdir -p "$ROOT/_internal/DeepFaceLab" "$ROOT/_internal/_e"
    ARCHIVIO="$ROOT/_internal/_e/codice-primo-avvio.tar.gz"
    curl -LsSf -o "$ARCHIVIO" "$URL_CODICE"
    tar -xzf "$ARCHIVIO" -C "$ROOT/_internal/DeepFaceLab" --strip-components=1
    rm -f "$ARCHIVIO"
    if [ ! -f "$SETUP/__main__.py" ]; then
        echo "[install] l'archivio scaricato da $URL_CODICE non contiene setup/__main__.py: e' incompleto, oppure il trasferimento si e' interrotto. Riprova; se il problema resta, scarica quell'indirizzo a mano e verifica cosa contiene."
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
