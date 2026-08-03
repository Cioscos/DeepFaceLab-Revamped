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
export WORKSPACE="$INTERNAL/../workspace"
export DFL_ROOT="$INTERNAL/DeepFaceLab"
