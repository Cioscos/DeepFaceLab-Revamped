@echo off
rem Bootstrap dell'installer DeepFaceLab: procura uv, individua o clona il
rem repo, cede il controllo a setup/__main__.py. Batch puro:
rem nessun prerequisito oltre a curl.exe/tar.exe (Windows 10 1803+, Windows
rem 11), con ricaduta su powershell se curl manca.
setlocal
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"

rem Le cinque UV_* confinano uv dentro il pacchetto: nessuna scrittura nel
rem profilo utente, nessuna modifica al PATH di sistema.
rem UV_MANAGED_PYTHON non e' un doppione di UV_PYTHON_INSTALL_DIR: la seconda
rem dice soltanto DOVE mettere un Python scaricato, non obbliga a scaricarne
rem uno. Senza la prima, su una macchina che un 3.11 ce l'ha gia' (pyenv, lo
rem Store, python.org) uv riusa quello: _internal\python resta inesistente e
rem l'installazione punta a un interprete fuori dal pacchetto, che nessuno
rem qui aggiorna e che l'utente puo' disinstallare senza sapere cosa rompe.
rem Misurato su Windows: senza, pyvenv.cfg finiva su
rem C:\Users\<utente>\.pyenv\pyenv-win\versions\3.11.9; con, uv scarica
rem CPython 3.11.15 in _internal\python e pyvenv.cfg punta li'.
set "UV_INSTALL_DIR=%ROOT%\_internal\uv"
set "UV_PYTHON_INSTALL_DIR=%ROOT%\_internal\python"
set "UV_CACHE_DIR=%ROOT%\_internal\_e\uv-cache"
set "UV_NO_MODIFY_PATH=1"
set "UV_MANAGED_PYTHON=1"

rem setup\paths.py::resolve calcola uv_bin sotto --dest per il caso comune
rem (--dest assente o uguale a questa cartella), ma il binario di uv qui sopra
rem viene sempre scaricato accanto a QUESTO script, non a --dest: i due
rem divergono se l'utente passa un --dest esplicito verso un'altra cartella
rem (caso supportato -- verificato che si rompe per davvero senza questa
rem riga: "[Errno 2] No such file or directory:
rem '<dest>\_internal\uv\uv'"). DFL_UV_BIN dice a setup\__main__.py dove uv sta
rem davvero, senza che Python debba indovinarlo da --dest.
set "DFL_UV_BIN=%UV_INSTALL_DIR%\uv.exe"

set "REPO_URL=https://github.com/Cioscos/DeepFaceLab-Revamped.git"

rem Passo 1: dove sta setup\__main__.py. Due posizioni, in quest'ordine:
rem accanto a install.bat (dentro un clone del repo) e dentro
rem _internal\DeepFaceLab (installazione gia' presente, si sta rilanciando
rem per aggiornare). Se nessuna delle due, e' il primo avvio: si clona.
set "SETUP="
if exist "%ROOT%\setup\__main__.py" set "SETUP=%ROOT%\setup"
if not defined SETUP if exist "%ROOT%\_internal\DeepFaceLab\setup\__main__.py" set "SETUP=%ROOT%\_internal\DeepFaceLab\setup"
if defined SETUP goto :have_setup

echo [install] setup non trovato accanto a install.bat ne' in _internal\DeepFaceLab: clono %REPO_URL%
where git >nul 2>nul
if errorlevel 1 (
    echo [install] git non trovato nel PATH: installalo da https://git-scm.com/downloads e rilancia install.bat
    goto :fail
)
if not exist "%ROOT%\_internal" mkdir "%ROOT%\_internal"
git clone --depth 1 "%REPO_URL%" "%ROOT%\_internal\DeepFaceLab"
if errorlevel 1 (
    echo [install] git clone fallito.
    goto :fail
)
set "SETUP=%ROOT%\_internal\DeepFaceLab\setup"
rem setup\repo.py::sync_repo fa lo stesso controllo dopo ogni clone/pull, ma
rem solo una volta che Python gira -- qui non gira ancora, quindi va
rem ripetuto qui. Senza, un clone sul branch sbagliato (o corrotto)
rem fallirebbe dentro "uv run" con un errore che non nomina la causa vera.
if not exist "%SETUP%\__main__.py" (
    echo [install] il clone in %ROOT%\_internal\DeepFaceLab non contiene setup\__main__.py: il branch di default di questo repository potrebbe non includere ancora l'installer, oppure il clone e' corrotto o parziale. Verifica quale branch contiene setup\, requirements\ e scripts\commands.toml, clonalo a mano con "git clone -b <branch> %REPO_URL%", e rilancia install.bat da dentro quel clone.
    goto :fail
)

:have_setup
rem Passo 2: procurarsi uv, se non c'e' gia' da un giro precedente. La
rem versione e' fissata nell'URL, non in una variabile: un cambio a monte
rem non deve rompere le installazioni esistenti. Aggiornata
rem deliberatamente.
if exist "%UV_INSTALL_DIR%\uv.exe" goto :have_uv

echo [install] scarico uv 0.12.1...
if not exist "%UV_INSTALL_DIR%" mkdir "%UV_INSTALL_DIR%"
set "UV_ZIP=%UV_INSTALL_DIR%\_download.zip"
where curl >nul 2>nul
if errorlevel 1 goto :uv_via_powershell

curl -L -o "%UV_ZIP%" "https://github.com/astral-sh/uv/releases/download/0.12.1/uv-x86_64-pc-windows-msvc.zip"
if errorlevel 1 (
    echo [install] download di uv con curl fallito.
    goto :fail
)
goto :uv_extract

:uv_via_powershell
echo [install] curl.exe non trovato (Windows anteriore al 1803?): ricado su powershell Invoke-WebRequest.
powershell -Command "Invoke-WebRequest -Uri 'https://github.com/astral-sh/uv/releases/download/0.12.1/uv-x86_64-pc-windows-msvc.zip' -OutFile '%UV_ZIP%'"
if errorlevel 1 (
    echo [install] download di uv con powershell fallito.
    goto :fail
)

:uv_extract
tar -xf "%UV_ZIP%" -C "%UV_INSTALL_DIR%"
if errorlevel 1 (
    echo [install] estrazione di uv fallita.
    goto :fail
)
del "%UV_ZIP%" >nul 2>nul

:have_uv
rem Passo 3: cedere il controllo a setup/__main__.py, che da qui in poi e'
rem tutto Python e identico sui due sistemi.
rem
rem --dest "%ROOT%" prima di %*: setup\__main__.py::parse_args ha --dest
rem con default Path.cwd(), non la cartella di questo file, quindi senza
rem questo argomento esplicito un lancio da una shell con una cwd diversa
rem installerebbe li' invece che accanto a install.bat (un "cd /d %ROOT%"
rem qui sopra risolveva lo stesso problema
rem ma falliva su un percorso UNC, dove cmd.exe non supporta "cd /d", e
rem divergeva dalla convenzione del pacchetto, dove nessun .bat cambia mai
rem directory). Messo PRIMA di %*: argparse fa vincere l'ultima occorrenza
rem di un'opzione a valore singolo, quindi un --dest esplicito dell'utente
rem (che arriva dopo, via %*) sovrascrive questo senza bisogno di logica
rem in piu' -- verificato con setup.__main__.parse_args(["--dest", "A",
rem "--dest", "B"]).dest == Path("B").
"%UV_INSTALL_DIR%\uv.exe" run --python 3.11 --no-project "%SETUP%\__main__.py" --dest "%ROOT%" %*
if errorlevel 1 goto :fail

rem Anche il successo si ferma: install.bat si lancia con un doppio clic,
rem e senza pausa la finestra spariva portandosi via il riepilogo finale
rem di step_verify -- proprio l'elenco che dice quali asset sono a terra.
rem Il ramo :fail qui sotto la pausa ce l'aveva gia': il successo era il
rem solo esito che l'utente non poteva leggere.
endlocal
pause
exit /b 0

:fail
rem Un errore non deve far sparire la finestra prima che l'utente lo legga
rem: niente pausa non e' un'opzione qui.
echo.
echo [install] l'installazione non e' andata a buon fine. Guarda i messaggi sopra e _internal\_e\install.log.
endlocal
pause
exit /b 1
