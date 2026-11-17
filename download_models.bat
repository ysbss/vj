@echo off

setlocal EnableExtensions

cd /d "%~dp0"



set "PY=python\python.exe"



echo ==================================================

echo  VJ 歌词模型预下载（可选，非打包必需）

echo  不运行也可以：首次使用歌词功能时会自动下载

echo  Whisper : faster-whisper-medium（约 1.5 GB）

echo  翻译    : lemmonyjiang/opus-mt-ja-zh-jav（约 300 MB）

echo ==================================================



if not exist "%PY%" (

    echo [失败] 未找到 %PY%

    pause

    exit /b 1

)



"%PY%" scripts\download_models.py

if errorlevel 1 (

    echo [失败] 模型下载未完成

    pause

    exit /b 1

)



echo.

echo 模型已保存到 models\ 目录（可选，打包时不会自动带上）。

pause

