@echo off
REM Launch the AlphaEngine dashboard for remote access from your phone over
REM Tailscale.  Binds to this machine's Tailscale IP when Tailscale is up, so
REM the dashboard is reachable ONLY over your private tailnet -- not your LAN
REM and not the public internet.  Falls back to all interfaces (0.0.0.0) with a
REM warning if Tailscale isn't running yet.
REM
REM From your phone (with Tailscale installed + logged into the SAME account):
REM   http://<this-machine-name>:8501   (Tailscale MagicDNS)  -- or --
REM   http://<the Tailscale IP printed below>:8501

setlocal
cd /d "C:\Users\Coleman\Projects\AlphaEngine"
call .venv\Scripts\activate.bat

REM Locate the tailscale CLI (PATH first, then the default install dir).
set "TSCLI=tailscale"
where tailscale >nul 2>&1 || set "TSCLI=C:\Program Files\Tailscale\tailscale.exe"

REM Bind to the Tailscale IPv4 if we can get one; else all interfaces.
set "BINDADDR=0.0.0.0"
for /f "delims=" %%i in ('"%TSCLI%" ip -4 2^>nul') do set "BINDADDR=%%i"

if "%BINDADDR%"=="0.0.0.0" (
  echo [warn] No Tailscale IP found - binding to 0.0.0.0 ^(all interfaces^).
  echo        Start Tailscale ^(tray icon -^> Connect^) for private-only access,
  echo        then re-run this script.
) else (
  echo Binding dashboard to Tailscale IP %BINDADDR% - reachable from your
  echo phone over Tailscale, not exposed on your LAN or the internet.
)
echo.

streamlit run dashboard\app.py --server.address %BINDADDR% --server.port 8501 --server.headless true
