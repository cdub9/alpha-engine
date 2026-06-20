# Post-run check for daily_paper_trade.bat
#
# Called by the bat AFTER the digest pipeline finishes. Does two things:
#   1. Always writes data\last_run_status.json (machine-readable summary)
#   2. On failure, fires a Windows balloon notification so the user knows
#      without having to open the dashboard
#
# Failure is detected from EITHER:
#   - Non-zero exit code passed in as $args[0]
#   - "Traceback", "[ERROR]", or "ANTHROPIC_API_KEY not set" in the tail of
#     the log (catches cases where Python died but exit code was masked)
#
# Skipped runs (weekend / holiday) are NOT failures — they exit 0 and
# the log says "Skipping". JSON status='skipped' in that case.

[CmdletBinding()]
param(
    [int]$ExitCode = 0
)

$ErrorActionPreference = 'Continue'
$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition)
$logPath = Join-Path $projectRoot 'data\daily_paper_trade.log'
$statusPath = Join-Path $projectRoot 'data\last_run_status.json'

# Tail the log (last 60 lines is plenty for one run's output)
$tail = ''
if (Test-Path $logPath) {
    try {
        $tail = (Get-Content $logPath -Tail 60 -ErrorAction Stop) -join "`n"
    } catch {
        $tail = "<could not read log: $($_.Exception.Message)>"
    }
}

# Classify
$status = 'ok'
$errorReason = $null

if ($tail -match 'Skipping:\s*\S+\s+is a weekend') {
    $status = 'skipped_weekend'
} elseif ($tail -match 'Skipping digest generation') {
    $status = 'skipped_holiday'
} elseif ($ExitCode -ne 0) {
    $status = 'error'
    $errorReason = "exit code $ExitCode"
} elseif ($tail -match 'Traceback') {
    $status = 'error'
    $errorReason = 'Python traceback in log'
} elseif ($tail -match 'ANTHROPIC_API_KEY not set') {
    $status = 'error'
    $errorReason = 'API key not configured'
} elseif ($tail -match '\[ERROR\]') {
    $status = 'error'
    $errorReason = 'ERROR line in log'
}

# Build status object
$statusObj = [ordered]@{
    timestamp     = (Get-Date).ToString('o')
    status        = $status
    exit_code     = $ExitCode
    error_reason  = $errorReason
    log_tail      = $tail
}

# Write JSON (always — success and failure both)
try {
    $statusObj | ConvertTo-Json -Depth 4 | Set-Content -Path $statusPath -Encoding UTF8
} catch {
    Write-Host "post_run_check: failed to write status JSON: $($_.Exception.Message)"
}

# On failure, fire a Windows balloon notification
if ($status -eq 'error') {
    try {
        Add-Type -AssemblyName System.Windows.Forms -ErrorAction Stop
        Add-Type -AssemblyName System.Drawing -ErrorAction Stop
        $notify = New-Object System.Windows.Forms.NotifyIcon
        $notify.Icon = [System.Drawing.SystemIcons]::Error
        $notify.Visible = $true
        $notify.BalloonTipTitle = 'AlphaEngine — Daily run FAILED'
        $notify.BalloonTipText  = "$errorReason. See data\daily_paper_trade.log."
        $notify.BalloonTipIcon  = [System.Windows.Forms.ToolTipIcon]::Error
        $notify.ShowBalloonTip(20000)
        # Keep the icon alive long enough for the toast to render in newer Windows
        Start-Sleep -Seconds 12
        $notify.Dispose()
    } catch {
        Write-Host "post_run_check: failed to show balloon: $($_.Exception.Message)"
    }
}

Write-Host "post_run_check: status=$status exit_code=$ExitCode"
