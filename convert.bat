@echo off
setlocal EnableExtensions
REM Local Dream NPU conversion (repo root)
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "TOOL=%ROOT%\tools\npu-convert"

if not exist "%TOOL%\pyproject.toml" (
  echo Error: tools\npu-convert not found 1>&2
  exit /b 1
)

where uv >nul 2>nul
if not errorlevel 1 (
  uv run --directory "%TOOL%" ld-convert %*
  exit /b %ERRORLEVEL%
)

python "%ROOT%\convert.py" %*
exit /b %ERRORLEVEL%
