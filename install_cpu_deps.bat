@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY=%~dp0python\python.exe"
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "NO_PAUSE="
if /i "%~1"=="--nopause" set "NO_PAUSE=1"

if not exist "%PY%" (
    echo [FAIL] Not found: %PY%
    if not defined NO_PAUSE pause
    exit /b 1
)

echo [1/4] Uninstall torch / torchaudio ...
"%PY%" -m pip uninstall -y torch torchaudio

echo.
echo [2/4] Install CPU torch / torchaudio ...
"%PY%" -m pip install torch torchaudio --index-url %TORCH_INDEX%
if errorlevel 1 (
    echo [FAIL] CPU torch install failed
    if not defined NO_PAUSE pause
    exit /b 1
)

echo.
echo [3/4] Upgrade ctranslate2 ...
"%PY%" -m pip install --upgrade ctranslate2
if errorlevel 1 (
    echo [FAIL] ctranslate2 install failed
    if not defined NO_PAUSE pause
    exit /b 1
)

echo.
echo [4/4] Verify torch:
"%PY%" -c "import sys,torch; print('exe:', sys.executable); print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"

echo.
echo [CHECK] detect_infer_device.py:
"%PY%" "%~dp0scripts\detect_infer_device.py"

echo.
echo Done. Run package.bat next.
if not defined NO_PAUSE pause
