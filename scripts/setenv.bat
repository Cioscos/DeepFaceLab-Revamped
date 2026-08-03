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
SET WORKSPACE=%INTERNAL%\..\workspace
SET DFL_ROOT=%INTERNAL%\DeepFaceLab
