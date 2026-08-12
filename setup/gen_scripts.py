"""Command -> contenuto di un .bat o di un .sh.

Un solo generatore per due sistemi: e' il punto in cui commands.toml smette di
essere dati e diventa i file che l'utente lancia. Il lato Linux e' modellato
su quello Windows, riga per riga.
"""
from __future__ import annotations

from pathlib import Path

from setup.commands import (
    KIND_CLEAR, KIND_EBSYNTH, KIND_MAIN, KIND_VIEWER,
    PAUSE_ALWAYS, PAUSE_ON_ERROR,
    Command, Invocation,
)

# Comandi il cui .bat odierno porta una singola riga di continuazione
# rientrata di 7 spazi anziche' 4 -- unico caso fra i 55, verificato byte per
# byte contro l'ancora. Non e' derivabile da nessun campo di Command: e'
# semplicemente cosi' che il file esiste da prima di questa fase.
_SEVEN_SPACE_INDENT = {"3) cut video (drop video on me)"}

# Due spazi anziche' uno fra il verbo e il `^` di continuazione. Verificato
# contro l'ancora: "faceset resize" (stesso verbo facesettool, stessa forma)
# ne ha uno solo, quindi non e' una regola del verbo -- e' un refuso del solo
# file "4.2) data_src util faceset enhance.bat".
_DOUBLE_SPACE_BEFORE_CARET = {"4.2) data_src util faceset enhance"}

# Una riga fatta di soli 4 spazi compare nell'ancora subito prima del `pause`
# finale, per questi tre comandi soltanto. In "5) ... ALIGNED_DEBUG" si
# aggiunge alla riga vuota normale; nei due "7) merge *" la SOSTITUISCE (niente
# riga vuota vera). Verificato con xxd che non e' un artefatto di
# trascrizione: il file alla radice del pacchetto ha lo stesso identico byte.
# Nessun campo di Command lo rappresenta, e non correla con la struttura degli
# argomenti (i due merge hanno tutti argomenti appaiati, l'ALIGNED_DEBUG ne ha
# due non appaiati in coda): sono refusi indipendenti dei file originali.
_STRAY_BLANK_ADDS_TO_TRAILER = {
    "5) data_dst faceset MANUAL RE-EXTRACT DELETED ALIGNED_DEBUG",
}
_STRAY_BLANK_REPLACES_TRAILER = {
    "7) merge AMP",
    "7) merge SAEHD",
}

# Riga vuota doppia (anziche' singola) fra 'call' e la prima invocazione.
# Verificato contando i .bat con due invocazioni ("8) merged to *", quattro
# file identici nella struttura degli argomenti): tre hanno la riga doppia,
# "8) merged to mp4" no. Non e' quindi una regola sul numero di invocazioni --
# e' un refuso indipendente di tre file su quattro.
_DOUBLE_BLANK_BEFORE_FIRST_INVOCATION = {
    "8) merged to avi",
    "8) merged to mov lossless",
    "8) merged to mp4 lossless",
}

# Un solo comando fra i 55 finisce con un CRLF di troppo (verificato con xxd
# sia sull'ancora sia sull'originale alla radice del pacchetto: identico,
# quindi e' una proprieta' vera del file, non un artefatto di copia): su
# questo unico file vince il riferimento, non la regola generale.
_TRAILING_CRLF = {"10.misc) start EBSynth"}


def _win_path(placeholder_path: str) -> str:
    """'{WORKSPACE}/data_src/aligned' -> '%WORKSPACE%\\data_src\\aligned'."""
    out = placeholder_path
    for name in ("WORKSPACE", "INTERNAL", "DFL_ROOT"):
        out = out.replace("{" + name + "}", f"%{name}%")
    return out.replace("/", "\\")


def _pair_up(args: tuple[str, ...]) -> list[list[str]]:
    """Raggruppa ogni opzione col suo valore, una riga per gruppo.

    I .bat odierni scrivono `--model-dir "%WORKSPACE%\\model" ^` su una riga e
    i flag senza valore (`--lossless`) su una riga per conto loro.
    """
    groups: list[list[str]] = []
    i = 0
    while i < len(args):
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            groups.append([args[i], args[i + 1]])
            i += 2
        else:
            groups.append([args[i]])
            i += 1
    return groups


def _quote_win(arg: str) -> str:
    """I .bat odierni virgolettano i percorsi e non i letterali."""
    rendered = _win_path(arg)
    needs_quotes = "%" in rendered or " " in rendered
    return f'"{rendered}"' if needs_quotes else rendered


def _win_invocation(cmd: Command, inv: Invocation, passthrough: bool) -> list[str]:
    """Una chiamata a main.py, spezzata su piu' righe con le continuazioni ^."""
    sep = "  " if cmd.name in _DOUBLE_SPACE_BEFORE_CARET else " "
    head = '"%PYTHON_EXECUTABLE%" "%DFL_ROOT%\\main.py" ' + " ".join(inv.verb)
    indent = "       " if cmd.name in _SEVEN_SPACE_INDENT else "    "

    pairs = _pair_up(inv.args)
    if not pairs:
        return [head]

    body = [indent + " ".join(_quote_win(a) for a in chunk) for chunk in pairs]
    if passthrough:
        body[-1] += " %1"
    return [head + sep + "^"] + [ln + " ^" for ln in body[:-1]] + [body[-1]]


def _win_clear_workspace() -> list[str]:
    """Le dodici righe di mkdir/rmdir dell'ancora per '1) clear workspace'."""
    lines = ['mkdir "%WORKSPACE%" 2>nul']
    for sub in ("data_src", "data_dst"):
        lines.append(f'rmdir "%WORKSPACE%\\{sub}" /s /q 2>nul')
        lines.append(f'mkdir "%WORKSPACE%\\{sub}" 2>nul')
        lines.append(f'mkdir "%WORKSPACE%\\{sub}\\aligned" 2>nul')
    lines.append('rmdir "%WORKSPACE%\\model" /s /q 2>nul')
    lines.append('mkdir "%WORKSPACE%\\model" 2>nul')
    return lines


def render_bat(cmd: Command) -> bytes:
    lines = ["@echo off"]
    lines += [f"echo {text}" for text in cmd.echo]
    if cmd.pause_before:
        lines.append("pause")

    lines.append('call "%~dp0..\\_internal\\setenv.bat"')
    # setenv.bat puo' uscire con errore (progetto puntato da DFL_PROJECT
    # inesistente): 'call' riporta quell'errorlevel al chiamante ma non
    # ferma da solo lo script chiamante, quindi senza questa riga si
    # proseguirebbe con WORKSPACE/DFL_ROOT non definite o, peggio, STANTIE
    # da un comando precedente riuscito nella stessa sessione cmd.
    lines.append("if %errorlevel% NEQ 0 exit /b %errorlevel%")

    if cmd.kind == KIND_CLEAR:
        # L'ancora non ha riga vuota dopo 'call' per questo comando: le
        # cancellazioni seguono a ruota.
        lines += _win_clear_workspace()
        lines.append("echo DONE")
        lines.append("pause")
        return "\r\n".join(lines).encode("ascii")

    lines.append("")

    if cmd.kind == KIND_MAIN:
        if cmd.name in _DOUBLE_BLANK_BEFORE_FIRST_INVOCATION:
            lines.append("")
        for mk in cmd.mkdirs:
            lines.append(f'mkdir "{_win_path(mk)}" 2>nul')
        if cmd.mkdirs:
            lines.append("")
        for idx, inv in enumerate(cmd.invocations):
            lines += _win_invocation(cmd, inv, cmd.passthrough)
            if idx != len(cmd.invocations) - 1:
                lines.append("")
    elif cmd.kind == KIND_VIEWER:
        lines.append(f'start "" "{_win_path(cmd.target)}"')
    elif cmd.kind == KIND_EBSYNTH:
        lines.append(
            'start "" /D "%INTERNAL%\\EbSynth" /LOW "%INTERNAL%\\EbSynth\\EbSynth.exe" '
            '"%INTERNAL%\\EbSynth\\SampleProject\\sample.ebs"'
        )

    if cmd.name in _STRAY_BLANK_REPLACES_TRAILER:
        lines.append("    ")
    elif cmd.pause == PAUSE_ALWAYS:
        if cmd.name in _STRAY_BLANK_ADDS_TO_TRAILER:
            lines.append("    ")
        lines.append("")
        lines.append("pause")
    elif cmd.pause == PAUSE_ON_ERROR:
        lines.append("")
        lines += ["if %errorlevel% NEQ 0 (", "  pause", ")"]
    # PAUSE_NEVER: nessun trailer.

    if cmd.name in _STRAY_BLANK_REPLACES_TRAILER:
        lines.append("pause")

    if cmd.name in _TRAILING_CRLF:
        lines.append("")

    return "\r\n".join(lines).encode("ascii")


LF = b"\n"

# Messaggio mostrato quando l'utente lancia "10.misc) start EBSynth" su
# Linux: EbSynth non ha una build Linux, quindi lo script deve
# dirlo ed uscire con errore invece di fallire in modo oscuro.
_EBSYNTH_UNAVAILABLE = "EbSynth non e' disponibile su Linux: nessuna build Linux esiste."

# Messaggio del controllo DISPLAY/WAYLAND_DISPLAY prima dell'editor XSeg
# (Qt, headless -> errore di piattaforma che non nomina la causa).
_DISPLAY_CHECK = [
    'if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then',
    '    echo "L\'editor XSeg richiede un display grafico: nessun DISPLAY o WAYLAND_DISPLAY impostato."',
    "    exit 1",
    "fi",
]

_PAUSE_PROMPT = 'read -r -p "Premi Invio per chiudere..." _'


def _sh_path(placeholder_path: str) -> str:
    """'{WORKSPACE}/data_src/aligned' -> '${WORKSPACE}/data_src/aligned'."""
    out = placeholder_path
    for name in ("WORKSPACE", "INTERNAL", "DFL_ROOT"):
        out = out.replace("{" + name + "}", "${" + name + "}")
    return out


def _quote_sh(arg: str) -> str:
    rendered = _sh_path(arg)
    return f'"{rendered}"' if "$" in rendered or " " in rendered else rendered


def _sh_invocation(inv: Invocation, passthrough: bool) -> list[str]:
    """Una chiamata a main.py, spezzata su piu' righe con le continuazioni \\."""
    head = '"$PYTHON_EXECUTABLE" "$DFL_ROOT/main.py" ' + " ".join(inv.verb)
    groups = _pair_up(inv.args)
    if not groups:
        return [head]

    body = ["    " + " ".join(_quote_sh(a) for a in g) for g in groups]
    if passthrough:
        body[-1] += ' "$1"'
    return [head + " \\"] + [ln + " \\" for ln in body[:-1]] + [body[-1]]


def _sh_clear_workspace() -> list[str]:
    """Le stesse cancellazioni/ricreazioni di '1) clear workspace', in bash."""
    lines = ['mkdir -p "$WORKSPACE"']
    for sub in ("data_src", "data_dst"):
        lines.append(f'rm -rf "$WORKSPACE/{sub}"')
        lines.append(f'mkdir -p "$WORKSPACE/{sub}"')
        lines.append(f'mkdir -p "$WORKSPACE/{sub}/aligned"')
    lines.append('rm -rf "$WORKSPACE/model"')
    lines.append('mkdir -p "$WORKSPACE/model"')
    return lines


def render_sh(cmd: Command) -> bytes:
    lines = [
        "#!/usr/bin/env bash",
        "set -eu",
        'source "$(dirname "$0")/../_internal/setenv.sh"',
    ]

    lines += [f'echo "{text}"' for text in cmd.echo]
    if cmd.pause_before:
        lines.append(_PAUSE_PROMPT)

    if cmd.kind == KIND_CLEAR:
        lines += _sh_clear_workspace()
        lines.append('echo "DONE"')
    elif cmd.kind == KIND_MAIN:
        # Il controllo del display precede l'invocazione xseg editor: non un
        # elenco di nomi (fossile senza ancora, a differenza delle sei
        # eccezioni del lato Windows che riproducono un riferimento reale
        # byte per byte), ma una condizione sul dato stesso in
        # commands.toml. Un terzo comando che invocasse "xseg editor" in
        # futuro erediterebbe il controllo senza bisogno di aggiornare
        # questa lista.
        if any(inv.verb == ("xseg", "editor") for inv in cmd.invocations):
            lines += _DISPLAY_CHECK
        for mk in cmd.mkdirs:
            lines.append(f'mkdir -p "{_sh_path(mk)}"')
        if cmd.passthrough:
            lines.append('if [ -z "${1:-}" ]; then')
            lines.append('    read -r -p "Percorso del file di input: " REPLY_PATH')
            lines.append('    set -- "$REPLY_PATH"')
            lines.append("fi")
        for inv in cmd.invocations:
            invocation_lines = _sh_invocation(inv, cmd.passthrough)
            if cmd.pause == PAUSE_ON_ERROR:
                # `set -eu` uscirebbe subito al primo fallimento: negando il
                # comando con `if ! ... ; then` lo si mette in un contesto
                # condizionale, dove -e non si applica, cosi' si puo' mettere
                # in pausa prima di propagare l'errore. Il `then` resta su
                # una riga propria (mai accodato all'ultima riga d'argomenti)
                # perche' tutto cio' che segue "main.py" sulla stessa riga
                # logica sono gli argomenti: un suffisso li' contaminerebbe
                # il confronto con il .bat gemello.
                invocation_lines[0] = "if ! " + invocation_lines[0]
                lines += invocation_lines
                lines.append("then")
                lines.append("    " + _PAUSE_PROMPT)
                lines.append("    exit 1")
                lines.append("fi")
            else:
                lines += invocation_lines
    elif cmd.kind == KIND_VIEWER:
        lines.append(f'xdg-open "{_sh_path(cmd.target)}"')
    elif cmd.kind == KIND_EBSYNTH:
        lines.append(f'echo "{_EBSYNTH_UNAVAILABLE}"')
        lines.append("exit 1")
        # Lo script e' gia' uscito con errore: un trailer di pausa dopo
        # `exit 1` non verrebbe mai raggiunto. Nessun'altra istruzione
        # segue -- vedi il ritorno sotto, che comunque non aggiunge nulla
        # per questo kind perche' cmd.pause == PAUSE_NEVER per l'unica
        # entry di questo kind in commands.toml.

    # Il campo `pause` va onorato per ogni kind che non sia gia' uscito da
    # solo (KIND_EBSYNTH) o che non lo gestisca gia' per-invocazione
    # (KIND_MAIN con PAUSE_ON_ERROR, sopra): altrimenti il dato dichiarato
    # in commands.toml sarebbe un'informazione che il generatore ignora.
    # Prima del fix, KIND_CLEAR tornava subito dopo "echo DONE" senza mai
    # controllare cmd.pause: "1) clear workspace" dichiara pause="always"
    # ma il .sh non metteva mai in pausa dopo la cancellazione.
    handled_inline = cmd.kind == KIND_MAIN and cmd.pause == PAUSE_ON_ERROR
    if cmd.kind != KIND_EBSYNTH and not handled_inline and cmd.pause == PAUSE_ALWAYS:
        lines.append(_PAUSE_PROMPT)

    return "\n".join(lines).encode("utf-8") + LF


def generate(commands: list[Command], out_dir: Path, system: str) -> list[Path]:
    """Scrive i file e restituisce i percorsi. Su linux li rende eseguibili."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for cmd in commands:
        if system == "win":
            path = out_dir / f"{cmd.name}.bat"
            path.write_bytes(render_bat(cmd))
        else:
            path = out_dir / f"{cmd.name}.sh"
            path.write_bytes(render_sh(cmd))
            path.chmod(0o755)
        written.append(path)
    return written
