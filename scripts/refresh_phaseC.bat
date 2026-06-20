@echo off
REM Weekly Phase C bars-only refresh.
REM
REM Refreshes market_bars for ~400 S&P 500 symbols that are NOT in the LLM
REM universe. These bars power the Lookup page but do NOT affect the nightly
REM digest or paper-trading pipeline.
REM
REM Run time: 30-90 minutes (400 symbols, yfinance, rate-limited).
REM Recommended schedule: Sunday 8:00 AM (well away from the daily 6 PM run).
REM
REM IMPORTANT: Never overlap with daily_paper_trade.bat — both hold the DuckDB
REM write lock. Daily runs at 6 PM on weekdays; this runs Sunday 8 AM = safe.

setlocal
cd /d "C:\Users\Coleman\Projects\AlphaEngine"

echo. >> data\phaseC_refresh.log
echo ===== %DATE% %TIME% ===== >> data\phaseC_refresh.log
call .venv\Scripts\activate.bat

echo --- phase-c bar refresh (since 10 days) --- >> data\phaseC_refresh.log
python scripts\backfill_sp500.py --since 10 --log-level WARNING >> data\phaseC_refresh.log 2>&1
set RC=%ERRORLEVEL%

if %RC%==0 (
    echo Phase C refresh SUCCEEDED >> data\phaseC_refresh.log
) else (
    echo Phase C refresh FAILED (exit %RC%) >> data\phaseC_refresh.log
    REM Fire a balloon toast so failure is visible
    powershell -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; $n=New-Object System.Windows.Forms.NotifyIcon; $n.Icon=[System.Drawing.SystemIcons]::Warning; $n.Visible=$true; $n.ShowBalloonTip(8000,'AlphaEngine','Phase C refresh failed - check phaseC_refresh.log',[System.Windows.Forms.ToolTipIcon]::Error); Start-Sleep 3; $n.Dispose()" >> data\phaseC_refresh.log 2>&1
)

exit /b %RC%
