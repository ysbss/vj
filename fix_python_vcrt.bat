@echo off
rem 将系统 MSVC 运行库复制到 python 目录，避免 python 加载 jpackage runtime\bin 里不兼容的 msvcp140.dll
setlocal EnableExtensions
cd /d "%~dp0"

set "PYDIR=%~dp0python"
set "CT2DIR=%PYDIR%\Lib\site-packages\ctranslate2"
if not exist "%PYDIR%\python.exe" (
    echo [错误] 未找到 %PYDIR%\python.exe
    exit /b 1
)

set "SYS=%SystemRoot%\System32"
for %%F in (msvcp140.dll vcruntime140.dll vcruntime140_1.dll) do (
    if exist "%SYS%\%%F" (
        copy /Y "%SYS%\%%F" "%PYDIR%\" >nul
        echo [ok] %%F -> python\
        if exist "%CT2DIR%" (
            copy /Y "%SYS%\%%F" "%CT2DIR%\" >nul
            echo [ok] %%F -> ctranslate2\
        )
    ) else (
        echo [跳过] 系统目录无 %%F
    )
)

echo.
echo 完成。请重新运行 DS.exe 测试歌词对齐。
pause
