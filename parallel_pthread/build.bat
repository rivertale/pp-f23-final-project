@echo off

if not exist build mkdir build
pushd build

where /q cl.exe
if %ERRORLEVEL% neq 0 (
    echo MSVC aborted: "cl" not found - please run under MSVC x64 native tools command prompt
) else (
    cl /nologo /Fe:kmeans.exe ..\main.c /link /INCREMENTAL:NO
)

del /q *.obj
popd