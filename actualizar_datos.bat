@echo off
REM Actualizacion semanal del pipeline BTC - 100% sin API key.
REM Doble clic: baja RSS (acumulando), actualiza precios, re-etiqueta y reintenta entrenar.
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1

echo ============================================================
echo  Pipeline BTC - actualizacion semanal (RSS, sin API key)
echo  %DATE% %TIME%
echo ============================================================
echo.

echo [1/4] Descargando noticias RSS (acumulando sin duplicar)...
python download_news_rss.py --append
if errorlevel 1 (
  echo   AVISO: no se pudo descargar RSS. Revisa tu conexion e intenta de nuevo.
) else (
  echo   OK noticias.
)
echo.

echo [2/4] Actualizando precios BTC-USD...
python download_prices.py
if errorlevel 1 (
  echo   AVISO: fallo la descarga de precios.
) else (
  echo   OK precios.
)
echo.

echo [3/4] Etiquetando dataset con retornos 24h...
python build_dataset.py
if errorlevel 1 (
  echo   AVISO: no se pudo construir el dataset. Quizas faltan noticias o precios.
) else (
  echo   OK dataset.
)
echo.

echo [4/4] Intentando entrenar modelo (necesita 200+ filas con BUY/SELL/HOLD)...
python train_model.py
if errorlevel 1 (
  echo.
  echo   INFO: todavia no hay suficiente historia para entrenar.
  echo   Segui corriendo este .bat una vez por semana con --append.
  echo   Cuando haya 200+ titulares con subas y bajas, el modelo se entrena solo.
) else (
  echo   OK modelo entrenado en artifacts\model.joblib.
)
echo.
echo ============================================================
echo  Listo. Ahora podes ver todo con:  streamlit run app.py
echo ============================================================
pause
