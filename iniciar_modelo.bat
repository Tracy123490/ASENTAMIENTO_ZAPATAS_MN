@echo off
cd /d "%~dp0"
python -m streamlit run "%~dp0modelo_m.py"
if errorlevel 1 (
    echo.
    echo No se pudo iniciar el modelo.
    echo Instala las dependencias con: python -m pip install -r requirements.txt
    pause
)
