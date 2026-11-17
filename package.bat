@echo off
chcp 936 >nul 2>&1
setlocal EnableExtensions
cd /d "%~dp0"

set "APP_NAME=DS"
set "MAIN_JAR=ds-1.0-SNAPSHOT.jar"
set "DIST_DIR=dist"
set "APP_DIR=%DIST_DIR%\%APP_NAME%"

echo ==================================================
echo  DS һ�����´��
echo ==================================================

echo.
echo [1/5] Maven ������...
call mvn clean package
if errorlevel 1 (
    echo [ʧ��] mvn clean package δͨ��
    pause
    exit /b 1
)

if not exist "target\%MAIN_JAR%" (
    echo [ʧ��] δ�ҵ�
    echo target\%MAIN_JAR%
    pause
    exit /b 1
)

echo.
echo [2/5] �����ɷ���Ŀ¼, ������
if exist "%APP_DIR%" (
    rmdir /s /q "%APP_DIR%"
    if exist "%APP_DIR%" (
        echo [ʧ��] �޷�ɾ��
        echo %APP_DIR%
        echo ��ر� DS.exe ����ش��ں�����
        pause
        exit /b 1
    )
    echo ��ɾ����Ŀ¼
)

echo.
echo [3/5] jpackage ���� exe, ������̨����
jpackage --name %APP_NAME% --input target --main-jar %MAIN_JAR% --main-class com.ds.MainApp --type app-image --dest %DIST_DIR% --win-console
if errorlevel 1 (
    echo [ʧ��] jpackage ִ��ʧ�ܣ���ȷ�� JDK 21 �� jpackage �� PATH ��
    pause
    exit /b 1
)

set "DS_EXE=%APP_DIR%\%APP_NAME%.exe"
if not exist "%DS_EXE%" (
    echo [ʧ��] δ�ҵ�
    echo %DS_EXE%
    pause
    exit /b 1
)

echo.
echo [4/5] ���� ffmpeg �� python, ����ϴ������ĵȴ�
xcopy /E /I /Y /Q "ffmpeg" "%APP_DIR%\ffmpeg"
if errorlevel 1 (
    echo [ʧ��] ���� ffmpeg ʧ��
    pause
    exit /b 1
)

xcopy /E /I /Y /Q "python" "%APP_DIR%\python"
if errorlevel 1 (
    echo [ʧ��] ���� python ʧ��
    pause
    exit /b 1
)

echo [4c/5] copy MSVC runtime into python dir ...
for %%F in (msvcp140.dll vcruntime140.dll vcruntime140_1.dll) do (
    if exist "%SystemRoot%\System32\%%F" (
        copy /Y "%SystemRoot%\System32\%%F" "%APP_DIR%\python\" >nul 2>&1
        if exist "%APP_DIR%\python\Lib\site-packages\ctranslate2" (
            copy /Y "%SystemRoot%\System32\%%F" "%APP_DIR%\python\Lib\site-packages\ctranslate2\" >nul 2>&1
        )
    )
)
if not exist "%APP_DIR%\python\msvcp140.dll" (
    echo [warn] python\msvcp140.dll missing - run fix_python_vcrt.bat
)

if exist "scripts" (
    xcopy /E /I /Y /Q "scripts" "%APP_DIR%\scripts"
    if errorlevel 1 (
        echo [ʧ��] ���� scripts ʧ��
        pause
        exit /b 1
    )
)

if exist "models\whisper-medium\model.bin" (
    echo [4b/5] ���Ʊ��� Whisper ģ�� whisper-medium ...
    if not exist "%APP_DIR%\models\whisper-medium" mkdir "%APP_DIR%\models\whisper-medium"
    xcopy /E /I /Y /Q "models\whisper-medium\*.bin" "%APP_DIR%\models\whisper-medium\"
    xcopy /E /I /Y /Q "models\whisper-medium\*.json" "%APP_DIR%\models\whisper-medium\" >nul 2>&1
    xcopy /E /I /Y /Q "models\whisper-medium\*.txt" "%APP_DIR%\models\whisper-medium\" >nul 2>&1
) else (
    echo [��ʾ] δ�ҵ� models\whisper-medium\model.bin������ʱ���������� Whisper
)

echo [��ʾ] ����� hf-cache��Whisper ����ʹ�� models\whisper-medium

echo.
echo [5/5] ��������Ŀ¼��˵���ļ�...
if not exist "%APP_DIR%\input_objects" mkdir "%APP_DIR%\input_objects"
if not exist "%APP_DIR%\output_objects" mkdir "%APP_DIR%\output_objects"

@REM if exist "input_objects" (
@REM     xcopy /E /I /Y /Q "input_objects\*" "%APP_DIR%\input_objects\" >nul 2>&1
@REM )

if exist "doubao_ex.txt" copy /Y "doubao_ex.txt" "%APP_DIR%\" >nul

if exist "ʹ��˵��.txt" copy /Y "ʹ��˵��.txt" "%APP_DIR%\" >nul
if exist "NOTICE" copy /Y "NOTICE" "%APP_DIR%\" >nul
if exist "LICENSE" copy /Y "LICENSE" "%APP_DIR%\" >nul

echo.
echo ==================================================
echo  ������
echo.
echo  ��ִ���ļ�
echo  %DS_EXE%
echo.
echo  ý���زĴ����
echo  %APP_DIR%\input_objects\
echo.
echo  ��Ʒ�����
echo  %APP_DIR%\output_objects\
echo.
echo  ��־�ļ�
echo  %APP_DIR%\app.log
echo ==================================================
pause