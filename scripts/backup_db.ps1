# Daily DuckDB backup.
#
# Called at the start of daily_paper_trade.bat, BEFORE any writes. Copies
# data\alpha_engine.duckdb to data\backups\alpha_engine.YYYYMMDD.duckdb
# (skips if today's backup already exists - idempotent for same-day re-runs).
# Then prunes to the most recent N backups (default 14 days).
#
# Why before writes: if Monday's run corrupts the file, today's backup is
# yesterday's good state - that's the recovery point.
#
# Cost: a few hundred MB at most; well under typical disk.

[CmdletBinding()]
param(
    [int]$KeepDays = 14
)

$ErrorActionPreference = 'Continue'
$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition)
$dbPath = Join-Path $projectRoot 'data\alpha_engine.duckdb'
$backupDir = Join-Path $projectRoot 'data\backups'

if (-not (Test-Path $dbPath)) {
    Write-Host "backup_db: source DB not found at $dbPath - skipping."
    exit 0
}

# Ensure backup directory exists
if (-not (Test-Path $backupDir)) {
    New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
}

$today = Get-Date -Format 'yyyyMMdd'
$dest = Join-Path $backupDir "alpha_engine.$today.duckdb"

if (Test-Path $dest) {
    Write-Host "backup_db: today's backup already exists at $dest - skipping copy."
} else {
    try {
        Copy-Item -Path $dbPath -Destination $dest -ErrorAction Stop
        $size = (Get-Item $dest).Length / 1MB
        Write-Host ("backup_db: copied to {0} ({1:N1} MB)" -f $dest, $size)
    } catch {
        Write-Host "backup_db: copy failed: $($_.Exception.Message)"
        # Non-fatal - don't block the daily run
        exit 0
    }
}

# Prune - keep only the most recent $KeepDays backup files
try {
    $all = Get-ChildItem -Path $backupDir -Filter 'alpha_engine.*.duckdb' |
        Sort-Object LastWriteTime -Descending
    if ($all.Count -gt $KeepDays) {
        $toDelete = $all | Select-Object -Skip $KeepDays
        foreach ($f in $toDelete) {
            Remove-Item $f.FullName -Force
            Write-Host "backup_db: pruned old backup $($f.Name)"
        }
    }
    Write-Host "backup_db: $($all.Count) backups retained (keep last $KeepDays)"
} catch {
    Write-Host "backup_db: prune failed: $($_.Exception.Message)"
}

exit 0
