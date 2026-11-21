@echo off
setlocal
title 微光实验室 v0.2.1
cd /d "%~dp0"

rem 查找演示 Python：环境变量 > 本机演示环境 > PATH 上的 python > py 启动器
set "PY="
if defined MOBILEIE_PYTHON if exist "%MOBILEIE_PYTHON%" set "PY=%MOBILEIE_PYTHON%"
if not defined PY if exist "F:\dev-tools\python-envs\MobileIE-Demo\Scripts\python.exe" set "PY=F:\dev-tools\python-envs\MobileIE-Demo\Scripts\python.exe"
if not defined PY (
  where python >nul 2>nul
  if not errorlevel 1 set "PY=python"
)
if not defined PY (
  where py >nul 2>nul
  if not errorlevel 1 set "PY=py -3"
)

if not defined PY (
  echo [错误] 未找到 Python。
  echo 请安装 Python 3.10+ 并勾选 Tcl/Tk，或设置环境变量 MOBILEIE_PYTHON 指向 python.exe。
  pause
  exit /b 1
)

if not exist "%~dp0desktop.py" (
  echo [错误] 当前目录缺少 desktop.py，请保持解压后的文件相对位置不变。
  pause
  exit /b 1
)

rem 自检模式：带参数 check 时只验证依赖，不打开界面
if /i "%~1"=="check" (
  "%PY%" -c "import numpy,PIL,ai_edge_litert;print('依赖自检通过 ai-edge-litert',ai_edge_litert.__version__)"
  exit /b %errorlevel%
)

"%PY%" "%~dp0desktop.py"
if errorlevel 1 (
  echo.
  echo [提示] 程序异常退出。若提示缺少模块，请执行：
  echo     "%PY%" -m pip install -r "%~dp0requirements.txt"
  pause
)
