rem ========== BASE ENV ==========
SET INTERNAL=%~dp0
SET INTERNAL=%INTERNAL:~0,-1%
rem overriding windows user/local environment
SET LOCALENV_DIR=%INTERNAL%\_e
SET TMP=%LOCALENV_DIR%\t
SET TEMP=%LOCALENV_DIR%\t
SET HOME=%LOCALENV_DIR%\u
SET HOMEPATH=%LOCALENV_DIR%\u
SET USERPROFILE=%LOCALENV_DIR%\u
SET LOCALAPPDATA=%USERPROFILE%\AppData\Local
SET APPDATA=%USERPROFILE%\AppData\Roaming
rem Le cartelle appena nominate vanno create qui, come fa gia' setenv.sh con
rem TMPDIR e XDG_CACHE_HOME: puntarci senza crearle non redirige niente.
rem Con _e\t inesistente il trampolino di uv dentro .venv\Scripts\python.exe
rem stampa "warning: Failed to set cwd to temp dir" a ogni script, e -- il
rem danno vero, che il warning nasconde -- tempfile.gettempdir() scarta TMP
rem e ripiega su C:\WINDOWS\Temp: i temporanei finiscono fuori dal pacchetto,
rem che e' esattamente cio' che questo blocco esiste per impedire. Misurato
rem su Windows. mkdir crea anche i genitori, quindi %LOCALAPPDATA% tira su
rem _e\u e _e\u\AppData per conto suo.
if not exist "%TMP%" mkdir "%TMP%"
if not exist "%LOCALAPPDATA%" mkdir "%LOCALAPPDATA%"
if not exist "%APPDATA%" mkdir "%APPDATA%"

rem Con USERPROFILE rediretto qui, la shell di Windows risolve da qui anche
rem le sue known folder, e il picker nativo di cartelle apre il ramo Desktop
rem prima ancora di mostrarsi: puntato al nulla si apre con un errore ("...
rem non disponibile. Se la posizione e' in questo PC, assicurarsi che il
rem dispositivo o l'unita' sia connessa..."). Riprodotto sul campo aprendo
rem Workspace > Open..., ma non e' un problema di una finestra sola: lo
rem prende ogni dialog nativo del pacchetto, editor XSeg compreso. Le sei
rem sono quelle che il riquadro di navigazione risolve da solo; crearle
rem costa niente e rende il profilo rediretto un profilo valido, invece di
rem uno a cui manca sempre la prossima cartella.
for %%D in (Desktop Documents Downloads Music Pictures Videos) do if not exist "%USERPROFILE%\%%D" mkdir "%USERPROFILE%\%%D"

rem ========== PYTHON ENV ==========
SET PYTHON_PATH=%INTERNAL%\.venv\Scripts
rem overriding default python env vars in order not to interfere with any system python installation
SET PYTHONHOME=
SET PYTHONPATH=
SET PYTHONEXECUTABLE=%PYTHON_PATH%\python.exe
SET PYTHONWEXECUTABLE=%PYTHON_PATH%\pythonw.exe
SET PYTHON_EXECUTABLE=%PYTHON_PATH%\python.exe
SET PYTHONW_EXECUTABLE=%PYTHON_PATH%\pythonw.exe
SET PYTHON_BIN_PATH=%PYTHON_EXECUTABLE%
SET PYTHON_LIB_PATH=%PYTHON_PATH%\Lib\site-packages
SET QT_QPA_PLATFORM_PLUGIN_PATH=%PYTHON_LIB_PATH%\PyQt5\Qt\plugins
SET PATH=%PYTHON_PATH%;%PYTHON_PATH%\Scripts;%PATH%

rem ========== ADDITIONAL ENV ==========
SET FFMPEG_PATH=%INTERNAL%\ffmpeg
SET PATH=%FFMPEG_PATH%;%PATH%
rem ========== PROGETTO ==========
rem WORKSPACE si risolve su tre livelli: DFL_PROJECT, il puntatore scritto
rem dall'interfaccia grafica, la radice. Il terzo caso e' il comportamento che
rem questo file ha sempre avuto: un'installazione senza progetti non si accorge
rem di niente, output compreso.
SET DFL_PROJECTS_ROOT=%INTERNAL%\..\workspace
SET DFL_PROGETTO=
SET DFL_CANDIDATO=
if defined DFL_PROJECT (
    if exist "%DFL_PROJECTS_ROOT%\%DFL_PROJECT%\" (
        SET DFL_PROGETTO=%DFL_PROJECT%
    ) else (
        rem Nessuna ricaduta: chi ha impostato la variabile ha detto su cosa
        rem vuole lavorare, e farlo lavorare su altro e' peggio che fermarsi.
        echo DFL_PROJECT names a project that does not exist: %DFL_PROJECT% 1>&2
        exit /b 1
    )
) else (
    if exist "%DFL_PROJECTS_ROOT%\.progetto-attivo" (
        set /p DFL_CANDIDATO=<"%DFL_PROJECTS_ROOT%\.progetto-attivo"
    )
)
rem Le stesse esclusioni del lato .sh (`case "" | */* | *\* | .*`), che sono
rem anche la regola di gui/progetti.py::_nome_esce_dalla_radice: vuoto, una
rem barra in qualunque verso, i due punti di una lettera di unita' (senza
rem barra un'unita' non e' altrimenti intercettata: "C:" da solo), un nome
rem che inizia per punto. Lettura e controlli in blocchi separati per la
rem stessa ragione della set /p sopra: dentro il blocco parentesizzato
rem %DFL_CANDIDATO% si espanderebbe prima di essere scritta.
if defined DFL_CANDIDATO if not "%DFL_CANDIDATO%"=="%DFL_CANDIDATO:\=%" SET DFL_CANDIDATO=
if defined DFL_CANDIDATO if not "%DFL_CANDIDATO%"=="%DFL_CANDIDATO:/=%" SET DFL_CANDIDATO=
if defined DFL_CANDIDATO if not "%DFL_CANDIDATO%"=="%DFL_CANDIDATO::=%" SET DFL_CANDIDATO=
if defined DFL_CANDIDATO if "%DFL_CANDIDATO:~0,1%"=="." SET DFL_CANDIDATO=
if defined DFL_CANDIDATO SET DFL_PROGETTO=%DFL_CANDIDATO%
SET DFL_CANDIDATO=
if defined DFL_PROGETTO (
    if not exist "%DFL_PROJECTS_ROOT%\%DFL_PROGETTO%\" SET DFL_PROGETTO=
)
if defined DFL_PROGETTO (
    SET WORKSPACE=%DFL_PROJECTS_ROOT%\%DFL_PROGETTO%
    echo Project: %DFL_PROGETTO%
) else (
    SET WORKSPACE=%DFL_PROJECTS_ROOT%
)
SET DFL_ROOT=%INTERNAL%\DeepFaceLab
rem Errorlevel pulito prima di tornare al chiamante. Il solo cammino che deve
rem propagare un errore (DFL_PROJECT che nomina un progetto inesistente,
rem sopra) esce gia' con `exit /b 1` prima di arrivare qui: ogni altro
rem cammino e' un successo, ma puo' aver lasciato un errorlevel diverso da
rem zero per conto suo -- `set /p` su un puntatore che esiste ed e' vuoto
rem (zero byte: una ricaduta benigna sulla radice) lascia %errorlevel%=1
rem pendente, misurato con cmd.exe vero. Da quando lo script chiamante
rem controlla l'errorlevel subito dopo la call (setup/gen_scripts.py), un
rem residuo cosi' fermerebbe ogni comando dell'utente in silenzio, senza un
rem messaggio. Un `exit /b 0` esplicito qui elimina l'intera classe, non
rem solo questo caso: qualunque comando sopra che lasci un errorlevel
rem sporco viene comunque ripulito da questa riga, che dev'essere l'ultima
rem del file.
exit /b 0
