@echo off
REM Ver la app BTC en el navegador - doble clic y listo.
REM No abrir app.py directamente: Streamlit se prende con este archivo.
chcp 65001 >nul
cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
  echo No encuentro Python. Instalalo desde https://www.python.org/downloads/
  echo IMPORTANTE: en el instalador tilda "Add python.exe to PATH".
  pause
  exit /b 1
)

echo Prendiendo la app... se abre sola en tu navegador.
echo Si no se abre sola, copia la direccion "Local URL" en Chrome o Edge.
echo Para apagarla, cerra esta ventana.
echo.
python -m streamlit run app.py
pause
