@echo off
rem Bootstrap dell'installer DeepFaceLab: procura uv, trova il codice gia'
rem presente in _internal\DeepFaceLab o lo scarica ed estrae da li', cede il
rem controllo a setup\__main__.py. Batch puro: nessun
rem prerequisito oltre a curl.exe/tar.exe (Windows 10 1803+, Windows 11),
rem con ricaduta su powershell se curl manca.
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

rem L'archivio del ramo pubblicato: due megabyte e mezzo, gli stessi
rem curl/tar che servono gia' per uv, e nessun sistema di controllo di
rem versione da installare prima. Lo stesso indirizzo sta in
rem setup\codice.py, che comanda da qui in avanti: qui serve solo per la
rem primissima volta, quando setup\ non esiste ancora e non c'e' nessun
rem Python da eseguire.
set "URL_CODICE=https://codeload.github.com/Cioscos/DeepFaceLab-Revamped/tar.gz/refs/heads/main"

rem Passo 1: il codice. Un posto solo dove cercarlo, _internal\DeepFaceLab,
rem e se non c'e' lo si scarica. Cercarlo anche accanto a questo script --
rem come faceva prima -- significava installare con una copia di setup\ che
rem nessuno aggiorna mai, mentre il codice eseguito e' sempre quello di
rem _internal.
set "SETUP=%ROOT%\_internal\DeepFaceLab\setup"
if exist "%SETUP%\__main__.py" goto :have_setup

echo [install] prima installazione: scarico il codice da %URL_CODICE%
if not exist "%ROOT%\_internal\DeepFaceLab" mkdir "%ROOT%\_internal\DeepFaceLab"
if not exist "%ROOT%\_internal\_e" mkdir "%ROOT%\_internal\_e"
set "ARCHIVIO=%ROOT%\_internal\_e\codice-primo-avvio.tar.gz"
where curl >nul 2>nul
if errorlevel 1 goto :codice_via_powershell

curl --fail -L -o "%ARCHIVIO%" "%URL_CODICE%"
if errorlevel 1 (
    echo [install] download del codice con curl fallito.
    goto :fail
)
goto :codice_estrai

:codice_via_powershell
echo [install] curl.exe non trovato (Windows anteriore al 1803?): ricado su powershell Invoke-WebRequest.
powershell -Command "Invoke-WebRequest -Uri '%URL_CODICE%' -OutFile '%ARCHIVIO%'"
if errorlevel 1 (
    echo [install] download del codice con powershell fallito.
    goto :fail
)

:codice_estrai
tar -xf "%ARCHIVIO%" -C "%ROOT%\_internal\DeepFaceLab" --strip-components=1
if errorlevel 1 (
    echo [install] estrazione del codice fallita.
    goto :fail
)
del "%ARCHIVIO%" >nul 2>nul
if not exist "%SETUP%\__main__.py" (
    echo [install] l'archivio scaricato da %URL_CODICE% non contiene setup\__main__.py: e' incompleto, oppure il trasferimento si e' interrotto. Riprova; se il problema resta, scarica quell'indirizzo a mano e verifica cosa contiene.
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

curl --fail -L -o "%UV_ZIP%" "https://github.com/astral-sh/uv/releases/download/0.12.1/uv-x86_64-pc-windows-msvc.zip"
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
