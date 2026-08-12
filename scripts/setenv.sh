#!/usr/bin/env bash
# Contratto d'ambiente di ogni script utente su Linux. Equivalente di
# setenv.bat, con le differenze che i due sistemi impongono davvero:
#
#  - niente CUDA/CUDNN nel PATH: le wheel di torch li portano con se';
#  - HOME NON viene rediretto. Su Windows la redirezione protegge il profilo
#    utente; su Linux rompe piu' di quanto protegga (ssh, git, sudo). Si
#    confinano invece TMPDIR e XDG_CACHE_HOME, che e' cio' che serve davvero.
INTERNAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export INTERNAL
export LOCALENV_DIR="$INTERNAL/_e"
export TMPDIR="$LOCALENV_DIR/t"
export XDG_CACHE_HOME="$LOCALENV_DIR/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"

unset PYTHONHOME PYTHONPATH
export PYTHON_EXECUTABLE="$INTERNAL/.venv/bin/python"
export PYTHONW_EXECUTABLE="$PYTHON_EXECUTABLE"
export FFMPEG_PATH="$INTERNAL/ffmpeg"
export PATH="$INTERNAL/.venv/bin:$FFMPEG_PATH:$PATH"
# ========== PROGETTO ==========
# WORKSPACE si risolve su tre livelli: DFL_PROJECT, il puntatore scritto
# dall'interfaccia grafica, la radice. Il terzo caso e' il comportamento che
# questo file ha sempre avuto: un'installazione senza progetti non si accorge
# di niente, output compreso.
DFL_PROJECTS_ROOT="$(cd "$INTERNAL/.." && pwd)/workspace"
export DFL_PROJECTS_ROOT
_dfl_progetto=""
if [ -n "${DFL_PROJECT:-}" ]; then
    if [ -d "$DFL_PROJECTS_ROOT/$DFL_PROJECT" ]; then
        _dfl_progetto="$DFL_PROJECT"
    else
        # Nessuna ricaduta: chi ha impostato la variabile ha detto su cosa
        # vuole lavorare, e farlo lavorare su altro e' peggio che fermarsi.
        echo "DFL_PROJECT names a project that does not exist: $DFL_PROJECT" >&2
        return 1 2>/dev/null || exit 1
    fi
elif [ -f "$DFL_PROJECTS_ROOT/.progetto-attivo" ]; then
    read -r _dfl_candidato < "$DFL_PROJECTS_ROOT/.progetto-attivo" || _dfl_candidato=""
    case "$_dfl_candidato" in
        ""|*/*|*\\*|.*) _dfl_candidato="" ;;
    esac
    if [ -n "$_dfl_candidato" ] && [ -d "$DFL_PROJECTS_ROOT/$_dfl_candidato" ]; then
        _dfl_progetto="$_dfl_candidato"
    fi
    unset _dfl_candidato
fi
if [ -n "$_dfl_progetto" ]; then
    export WORKSPACE="$DFL_PROJECTS_ROOT/$_dfl_progetto"
    echo "Project: $_dfl_progetto"
else
    export WORKSPACE="$DFL_PROJECTS_ROOT"
fi
unset _dfl_progetto
export DFL_ROOT="$INTERNAL/DeepFaceLab"
