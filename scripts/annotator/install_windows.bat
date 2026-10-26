@echo off
REM One-time setup of the reef-track annotation tool on Windows. Double-click this file.
REM Safe to run again (e.g. after downloading a newer version of reef-track).
REM   1. installs Miniforge in %USERPROFILE%\miniforge3 if it isn't there
REM   2. creates/updates the environment %USERPROFILE%\envs\reeftrack-annotate (~1.7 GB)
REM   3. puts an "Annotate fish videos" launcher on the Desktop
setlocal
set "REPO=%~dp0..\.."
pushd "%REPO%" && set "REPO=%CD%" && popd
set "MF=%USERPROFILE%\miniforge3"
set "ENV=%USERPROFILE%\envs\reeftrack-annotate"
echo reef-track annotator setup
echo   code:        %REPO%
echo   environment: %ENV%
echo.

if not exist "%MF%\Scripts\conda.exe" (
  echo Installing Miniforge ^(one-off, ~100 MB download^)...
  curl -fL -o "%TEMP%\Miniforge3.exe" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe || goto :error
  start /wait "" "%TEMP%\Miniforge3.exe" /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /S /D=%MF%
)
if not exist "%MF%\Scripts\conda.exe" goto :error

if exist "%ENV%\python.exe" (
  call "%MF%\Scripts\conda.exe" env update --prefix "%ENV%" -f "%REPO%\environment-annotate.yml" --prune || goto :error
) else (
  call "%MF%\Scripts\conda.exe" env create --prefix "%ENV%" -f "%REPO%\environment-annotate.yml" -y || goto :error
)
"%ENV%\python.exe" -m pip install -q -e "%REPO%" --no-deps || goto :error

set "LAUNCHER=%USERPROFILE%\Desktop\Annotate fish videos.bat"
(
  echo @echo off
  echo REM Opens the reef-track annotation tool. Keep this window open while you annotate.
  echo call "%MF%\Scripts\activate.bat" "%ENV%"
  echo cd /d "%REPO%"
  echo reeftrack annotate
  echo if errorlevel 1 pause
) > "%LAUNCHER%"

echo.
echo Done. Double-click "Annotate fish videos" on your Desktop to start.
pause
exit /b 0

:error
echo.
echo Setup failed - please send a screenshot of this window to the reef-track maintainer.
pause
exit /b 1
