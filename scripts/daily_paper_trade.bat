@echo off
REM Daily AlphaEngine paper-trading runner.
REM
REM Intended for Windows Task Scheduler.  Skips weekends (markets closed,
REM no fresh prices, no point spending ~$0.15 on a digest of stale data).
REM Logs each run to data/daily_paper_trade.log.
REM
REM Schedule recommendation: weekdays at 5:30pm or 6:00pm PT
REM   (after 4pm ET close + buffer for yfinance to update).

setlocal

REM ---- Skip weekends ----
for /f %%i in ('powershell -NoProfile -Command "(Get-Date).DayOfWeek.value__"') do set DOW=%%i
REM DayOfWeek: Sunday=0, Monday=1, ..., Saturday=6
if "%DOW%"=="0" goto weekend
if "%DOW%"=="6" goto weekend

REM ---- Run ----
cd /d "C:\Users\Coleman\Projects\AlphaEngine"
echo. >> data\daily_paper_trade.log
echo ===== %DATE% %TIME% ===== >> data\daily_paper_trade.log
call .venv\Scripts\activate.bat

REM Daily DuckDB backup BEFORE any writes. Idempotent same-day; keeps last 14.
echo --- db backup --- >> data\daily_paper_trade.log
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\backup_db.ps1 >> data\daily_paper_trade.log 2>&1

REM Refresh price bars + FRED macro for the 115-symbol universe (last 7 days).
REM CRITICAL: must run before digest so entry prices exist for today's signals.
REM Non-blocking: stale bars are caught by freshness indicator; digest still runs.
echo --- bar refresh (universe) --- >> data\daily_paper_trade.log
python scripts\backfill.py --since 7 --log-level WARNING >> data\daily_paper_trade.log 2>&1
if errorlevel 1 echo [warn] universe backfill exited non-zero, proceeding anyway >> data\daily_paper_trade.log

REM NOTE: Phase C bars-only refresh (400 S&P 500 symbols) is NOT run here.
REM It takes 30-90 min and will hold the DuckDB write lock across the nightly
REM Task Scheduler boundary, locking out GDELT and the digest on the next run.
REM Run refresh_phaseC.bat on a separate weekly schedule (Sunday 8am recommended).

REM Generate today's ML signal ranks (momentum composite + XGBoost).
REM Free - pure local compute on the bars refreshed above. Non-blocking:
REM the LLM digest does not depend on these rows existing.
echo --- ml signals --- >> data\daily_paper_trade.log
python scripts\run_ml_signals.py --log-level WARNING >> data\daily_paper_trade.log 2>&1
if errorlevel 1 echo [warn] ml signal generation exited non-zero, proceeding anyway >> data\daily_paper_trade.log

REM Refresh earnings dates for the real book's names (holdings snapshot).
REM Free (yfinance), non-blocking; keeps the Action Center's earnings-trim
REM guard current. --only-holdings keeps it to ~60 symbols (~1-2 min).
echo --- earnings refresh (holdings) --- >> data\daily_paper_trade.log
python scripts\refresh_earnings.py --only-holdings >> data\daily_paper_trade.log 2>&1
if errorlevel 1 echo [warn] earnings refresh exited non-zero, proceeding anyway >> data\daily_paper_trade.log

REM Refresh GDELT geopolitical signals (last 7 days only - older data already in DB).
REM Non-blocking: failure here shouldn't stop the paid digest call below.
echo --- gdelt ingest --- >> data\daily_paper_trade.log
python scripts\ingest_gdelt.py --timespan 7d --polite-sleep 4 >> data\daily_paper_trade.log 2>&1
if errorlevel 1 echo [warn] gdelt ingest exited non-zero, proceeding to digest anyway >> data\daily_paper_trade.log

REM Generate today's digest + open/score paper trades (the $0.15 path).
echo --- digest + paper run --- >> data\daily_paper_trade.log
python scripts\paper_trader.py run-day --generate >> data\daily_paper_trade.log 2>&1
set RC=%ERRORLEVEL%

REM Post-run: write status JSON + fire balloon toast on failure.
echo --- post-run check --- >> data\daily_paper_trade.log
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\post_run_check.ps1 %RC% >> data\daily_paper_trade.log 2>&1

exit /b %RC%

:weekend
cd /d "C:\Users\Coleman\Projects\AlphaEngine"
echo. >> data\daily_paper_trade.log
echo ===== %DATE% %TIME% ===== >> data\daily_paper_trade.log
echo Skipping: %DATE% is a weekend. >> data\daily_paper_trade.log
echo Skipping: %DATE% is a weekend.
REM Still write a status JSON so the dashboard shows the weekend skip.
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\post_run_check.ps1 0 >> data\daily_paper_trade.log 2>&1
exit /b 0
