@echo off
setlocal enabledelayedexpansion

@REM Define paths and names
set LCM_GEN_DES_PATH=arc_bridge
set LCM_PKG_DIR=lcm_msgs
set JAVA_ARCHIVE_NAME=lcm_msgs.jar
set LCM_PKG_PATH=%LCM_GEN_DES_PATH%\%LCM_PKG_DIR%

@REM Fix LCM Windows compatibility issue (https://github.com/lcm-proj/lcm/pull/581)
echo Applying LCM Windows compatibility fix...
for /f "tokens=2 delims= " %%i in ('pip show lcm ^| findstr "Location:"') do set LCM_LOCATION=%%i
set LCM_INIT_FILE=%LCM_LOCATION%\lcm\__init__.py

if exist "%LCM_INIT_FILE%" (
    @REM Check if the fix is already applied
    findstr /C:"subprocess.call([os.path.join" "%LCM_INIT_FILE%" >nul
    if !errorlevel! equ 0 (
        echo Backing up original __init__.py...
        copy "%LCM_INIT_FILE%" "%LCM_INIT_FILE%.backup"
        
        echo Applying Windows compatibility patch...
        python scripts/apply_lcm_patch.py "%LCM_INIT_FILE%"
        if !errorlevel! neq 0 (
            echo Error: Failed to apply LCM patch
            exit /b 1
        )
    ) else (
        echo LCM Windows compatibility fix already applied.
    )
) else (
    echo Error: LCM __init__.py not found at %LCM_INIT_FILE%
    exit /b 1
)

echo Generating LCM types...

@REM Generate Python types
for %%f in ("lcm_types\*.lcm") do (
    lcm-gen -p "%%f" --ppath arc_bridge
)

@REM Generate Java types
for %%f in ("lcm_types\*.lcm") do (
    lcm-gen -j "%%f" --jpath arc_bridge
)

@REM Check if generated files exist first
if not exist "%LCM_PKG_PATH%\*.py" (
    echo Error: No files found in !LCM_PKG_PATH!
    echo LCM types generation failed.
    exit /b 1
)

echo Done!

@REM Try to automatically locate the lcm.jar file
@REM First, get the LCM installation location from pip
for /f "tokens=2 delims= " %%i in ('pip show lcm ^| findstr "Location:"') do set LCM_LOCATION=%%i

@REM Try different possible locations for lcm.jar
if exist "%LCM_LOCATION%\share\java\lcm.jar" (
    set LCM_JAR_PATH=%LCM_LOCATION%\share\java\lcm.jar
    echo Found LCM jar in pip site-packages: !LCM_JAR_PATH!
) else if exist "C:\Program Files\lcm\share\java\lcm.jar" (
    set LCM_JAR_PATH=C:\Program Files\lcm\share\java\lcm.jar
    echo Found LCM jar in Program Files: !LCM_JAR_PATH!
) else if exist "lcm\build\lcm-java\lcm.jar" (
    set LCM_JAR_PATH=lcm\build\lcm-java\lcm.jar
    echo Found LCM jar from source build: !LCM_JAR_PATH!
) else if exist "lcm_types\lcm.jar" (
    set LCM_JAR_PATH=lcm_types\lcm.jar
    echo Found manually copied LCM jar: !LCM_JAR_PATH!
) else (
    echo Error: Unable to locate lcm.jar
    echo Searched locations:
    echo   - !LCM_LOCATION!\share\java\lcm.jar
    echo   - C:\Program Files\lcm\share\java\lcm.jar
    echo   - lcm\build\lcm-java\lcm.jar
    echo   - lcm_types\lcm.jar
    exit /b 1
)

echo Compiling generated Java types...

javac -cp %LCM_JAR_PATH% %LCM_PKG_PATH%\*.java --release 8 -Xlint:-options
if %errorlevel% neq 0 (
    echo Error: Java types compilation failed.
    goto :skip_java
)

cd %LCM_GEN_DES_PATH%

echo Creating JAR archive...
jar cf %LCM_PKG_DIR%\%JAVA_ARCHIVE_NAME% %LCM_PKG_DIR%\*.class
if !errorlevel! neq 0 (
    echo Warning: Failed to create Java archive.
)

cd ..

echo Done!

:skip_java

@REM Find the latest MATLAB preference folder and write to javaclasspath.txt
set MATLAB_PREF_DIR_BASE=%USERPROFILE%\AppData\Roaming\MathWorks\MATLAB

if exist "%MATLAB_PREF_DIR_BASE%" (
    @REM Find the latest MATLAB release folder (R20XX format)
    set MATLAB_LATEST_PREF_DIR=
    for /f "delims=" %%d in ('dir "%MATLAB_PREF_DIR_BASE%\R*" /b /ad /o:-n 2^>nul') do (
        if not defined MATLAB_LATEST_PREF_DIR set MATLAB_LATEST_PREF_DIR=%MATLAB_PREF_DIR_BASE%\%%d
    )
    
    if defined MATLAB_LATEST_PREF_DIR (
        echo Using MATLAB preference folder: !MATLAB_LATEST_PREF_DIR!
        
        @REM Set the javaclasspath.txt file path
        set JAVACLASS_PATH_FILE=!MATLAB_LATEST_PREF_DIR!\javaclasspath.txt
        set LCM_JAVA_ARCHIVE_PATH=%CD%\!LCM_PKG_PATH!\!JAVA_ARCHIVE_NAME!
        
        @REM Write LCM paths to javaclasspath.txt
        echo Adding .jar path to javaclasspath.txt
        echo !LCM_JAR_PATH!> "!JAVACLASS_PATH_FILE!"
        echo !LCM_JAVA_ARCHIVE_PATH!>> "!JAVACLASS_PATH_FILE!"
        
        echo MATLAB configuration completed successfully.
    ) else (
        echo No MATLAB preference folder found. Skipping MATLAB configuration.
    )
) else (
    echo MATLAB preference base directory not found: %MATLAB_PREF_DIR_BASE%
    echo Skipping MATLAB configuration.
)
