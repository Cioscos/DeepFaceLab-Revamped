# Installing DeepFaceLab Revamped

`install.bat` (Windows) and `install.sh` (Linux) replace the 7-Zip archive the
original build shipped as. They provision everything: a standalone Python, a
virtual environment, torch with the CUDA runtime, the network weights, ffmpeg,
and the numbered scripts for your operating system.

## Prerequisites

- **An up-to-date NVIDIA driver**, if you want to train on the GPU. With an
  older driver (or no NVIDIA GPU at all) the installer still completes, but
  installs the CPU build of torch, which is much slower to train with. See
  [If your driver is too old](#if-your-nvidia-driver-is-too-old) below.
- **At least 15 GB free** on the target disk. This is checked before anything
  is downloaded.
- **`git`** on the PATH, on both systems: the installer uses it to clone and
  update the code.
- Windows: **`curl.exe` / `tar.exe`**, included since Windows 10 1803 (if they
  are missing, the bootstrap falls back to `powershell -Command
  Invoke-WebRequest`). Linux: **`curl` / `tar`**, present on every common
  distribution.

You do not need to pre-install Python, CUDA or cuDNN. The installer downloads a
standalone Python 3.11, creates a virtual environment, and installs torch with
the CUDA runtime bundled in the wheel.

## Installing

```bat
:: Windows (Command Prompt or PowerShell)
git clone https://github.com/Cioscos/DeepFaceLab-Revamped.git DeepFaceLab
cd DeepFaceLab
install.bat
```

```bash
# Linux / WSL
git clone https://github.com/Cioscos/DeepFaceLab-Revamped.git DeepFaceLab
cd DeepFaceLab
./install.sh
```

The installer asks exactly one question, unless a flag already answers it:
whether to download `pretrain_faces` (1.8 GB, which speeds up the start of a
new training run) now or later. Everything else it does on its own, and it
finishes with a summary: where it installed, which torch build it chose (CUDA
or CPU) and why, which assets are present, and the command to start with.

## Updating

Re-running `install.bat` / `install.sh` **is** the update — there is no second
script. Go to the folder you installed into (the one containing `install.bat`)
and run it again. Every step is idempotent: it updates the code with `git pull
--ff-only` (and if you have local modifications in the clone, it tells you and
leaves them alone), regenerates the scripts in `scripts/`, skips assets that
are already downloaded and verified, and **never touches** `workspace/`.

## Flags

| Flag | Effect |
|---|---|
| `--dest <folder>` | install somewhere other than the current folder (usually unnecessary: run the script from inside the installation folder) |
| `--cpu` | force the CPU build of torch even when an NVIDIA GPU is present |
| `--with-pretrain` | include `pretrain_faces` (1.8 GB) without asking |
| `--no-pretrain` | skip `pretrain_faces` without asking |
| `--yes` | answer "yes" to every interactive question without asking it |
| `--dry-run` | list the steps and exit, without writing or downloading anything |

## Where the files go

```
<installation folder>/
├─ install.bat / install.sh        run it again to update
├─ DeepFaceLab GUI.*              start here: the GUI runs the steps for you
├─ scripts/                       every step, one file each, still usable on its own
├─ _internal/
│  ├─ uv/                          the uv binary
│  ├─ python/                      standalone CPython 3.11
│  ├─ .venv/                       torch and dependencies
│  ├─ _e/                          TMP, caches, install.log
│  ├─ ffmpeg/, model_generic_xseg/, pretrain_faces/, EbSynth/
│  └─ DeepFaceLab/                 the code, cloned or updated on every run
└─ workspace/
   ├─ data_src/aligned/
   ├─ data_dst/aligned/
   └─ model/
```

`workspace/` is created empty and the installer never touches it again, on the
first run or on any later one. Your extracted faces and your trained models
live there.

## Bringing a `workspace/` over from an old 7-Zip installation

The on-disk weight format did not change during the port: a `model/` folder
produced by the old build opens as-is here.

1. Install into a **new** folder, as above — do not overwrite the old one.
2. When the installation finishes, copy the contents of the old
   `<7z installation>/workspace/` into `<new installation>/workspace/`,
   overwriting the empty `data_src/aligned`, `data_dst/aligned` and `model`
   folders the installer created.
3. Carry on from the scripts in the new installation.

## If your NVIDIA driver is too old

The installer detects the GPU with `nvidia-smi` and chooses between the CUDA
and CPU builds of torch on its own. If an NVIDIA GPU is present but the driver
is below what the pinned CUDA build needs (`580.88` on Windows, `580.65.06` on
Linux — the same NVIDIA R580 release on both), it says so on screen and
continues anyway with the CPU build: **it does not stop**. Two ways forward:

- update the driver from <https://www.nvidia.com/Download/index.aspx> and run
  the installer again — it will switch to the CUDA build automatically;
- or stay on the CPU build for now (training will be much slower) and update
  whenever you like.

If `nvidia-smi` is missing entirely, or finds no GPU, the same applies: CPU
build, no interruption.

## When something goes wrong

- The full log of every run is at `_internal/_e/install.log`, always, including
  the runs that succeed.
- Every step that fails says so on screen and in the log: what failed, why, and
  what to do about it — and it stops there rather than continuing.
- An installation interrupted halfway through — even halfway through a 1.8 GB
  download — resumes where it left off. Just run the same script again.
