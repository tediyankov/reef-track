@echo off
REM One-time setup of reef-track WITH MASK TRACKING (SAM) on Windows. Double-click this file.
REM Without an NVIDIA GPU, tracking runs on the CPU and is slow.
REM Safe to run again (e.g. after downloading a newer version of reef-track).
REM   1. installs Miniforge in %USERPROFILE%\miniforge3 if it isn't there
REM   2. creates/updates the full environment %USERPROFILE%\envs\reeftrack-full (~6 GB; PyTorch + SAM)
REM   3. puts a "Track fish videos" launcher on the Desktop
setlocal
set "REPO=%~dp0..\.."
pushd "%REPO%" && set "REPO=%CD%" && popd
set "MF=%USERPROFILE%\miniforge3"
set "ENV=%USERPROFILE%\envs\reeftrack-full"
echo reef-track setup (annotation + mask tracking)
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
  call "%MF%\Scripts\conda.exe" env update --prefix "%ENV%" -f "%REPO%\environment.yml" --prune || goto :error
) else (
  call "%MF%\Scripts\conda.exe" env create --prefix "%ENV%" -f "%REPO%\environment.yml" -y || goto :error
)
"%ENV%\python.exe" -m pip install -q -e "%REPO%" --no-deps || goto :error

REM check nothing silently failed (conda keeps a half-built environment if the pip step breaks)
"%ENV%\Scripts\reeftrack.exe" check-install || goto :error
REM download the tracking model now, so the first click on a fish doesn't wait for a download
echo Downloading the tracking model (one-off, can take several minutes)...
"%ENV%\Scripts\reeftrack.exe" fetch-models || echo Model download failed; the GUI will retry when you first track a fish.

set "LAUNCHER=%USERPROFILE%\Desktop\Track fish videos.bat"
(
  echo @echo off
  echo REM Opens the reef-track annotation tool with mask tracking. Keep this window open while you work.
  echo call "%MF%\Scripts\activate.bat" "%ENV%"
  echo cd /d "%REPO%"
  echo reeftrack annotate
  echo if errorlevel 1 pause
) > "%LAUNCHER%"

echo.
echo Done. Double-click "Track fish videos" on your Desktop to start.
pause
exit /b 0

:error
echo.
echo Setup failed - please send a screenshot of this window to the reef-track maintainer.
pause
exit /b 1
