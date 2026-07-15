[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidatePattern('^[a-zA-Z0-9.-]+$')]
    [string] $SshHost,

    [Parameter(Mandatory)]
    [ValidateRange(1, 65535)]
    [int] $SshPort,

    [Parameter()]
    [string] $SshKeyPath = (Join-Path $HOME '.ssh\codex_vast_ed25519'),

    [Parameter()]
    [ValidateRange(1, 65535)]
    [int] $LocalPort = 8080
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory)]
        [string] $Command,

        [Parameter(Mandatory)]
        [string[]] $Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Command failed with exit code $LASTEXITCODE."
    }
}

$resolvedKeyPath = (Resolve-Path -LiteralPath $SshKeyPath).Path
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$branch = (& git -C $repositoryRoot branch --show-current).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($branch)) {
    throw 'Deploy from a named Git branch.'
}
if ($branch -notmatch '^[a-zA-Z0-9._/-]+$') {
    throw "The current branch name cannot be passed safely to the remote host: $branch"
}

$trackedChanges = & git -C $repositoryRoot status --short --untracked-files=no
if ($LASTEXITCODE -ne 0) {
    throw 'Unable to inspect the Git worktree.'
}
if ($trackedChanges) {
    throw 'Commit tracked changes before deploying so the remote version is reproducible.'
}

$runtimeDirectory = Join-Path $repositoryRoot '.runtime'
New-Item -ItemType Directory -Force -Path $runtimeDirectory | Out-Null
$bundlePath = Join-Path $runtimeDirectory 'voice-light.bundle'
$localEnvironmentPath = Join-Path $runtimeDirectory 'compute.env'
$tunnelPidPath = Join-Path $runtimeDirectory 'compute-tunnel.pid'
$remoteRepositoryPath = '/workspace/Voice-Light'
$target = "root@$SshHost"

if (Test-Path -LiteralPath $bundlePath) {
    Remove-Item -LiteralPath $bundlePath
}
Invoke-NativeCommand -Command 'git' -Arguments @(
    '-C', $repositoryRoot, 'bundle', 'create', $bundlePath, "refs/heads/$branch"
)
Invoke-NativeCommand -Command 'scp.exe' -Arguments @(
    '-i', $resolvedKeyPath, '-P', "$SshPort", $bundlePath,
    (Join-Path $PSScriptRoot 'provision-vast.sh'), "${target}:/tmp/"
)
Invoke-NativeCommand -Command 'ssh.exe' -Arguments @(
    '-i', $resolvedKeyPath, '-p', "$SshPort", '-o', 'BatchMode=yes',
    '-o', 'ConnectTimeout=15', $target, 'bash', '/tmp/provision-vast.sh',
    '/tmp/voice-light.bundle', $remoteRepositoryPath, $branch
)
Invoke-NativeCommand -Command 'scp.exe' -Arguments @(
    '-i', $resolvedKeyPath, '-P', "$SshPort",
    "${target}:${remoteRepositoryPath}/.env.compute", $localEnvironmentPath
)
$computeTokenLine = Get-Content -LiteralPath $localEnvironmentPath |
    Where-Object { $_ -like 'VOICE_LIGHT_COMPUTE_TOKEN=*' } |
    Select-Object -First 1
if ([string]::IsNullOrWhiteSpace($computeTokenLine)) {
    throw 'The downloaded compute environment does not contain VOICE_LIGHT_COMPUTE_TOKEN.'
}
$computeToken = $computeTokenLine.Substring('VOICE_LIGHT_COMPUTE_TOKEN='.Length)
$readinessHeaders = @{ Authorization = "Bearer $computeToken" }

if (Test-Path -LiteralPath $tunnelPidPath) {
    $existingProcessId = [int](Get-Content -LiteralPath $tunnelPidPath -Raw)
    $existingProcess = Get-Process -Id $existingProcessId -ErrorAction SilentlyContinue
    if ($null -ne $existingProcess -and $existingProcess.Name -eq 'ssh') {
        Stop-Process -Id $existingProcessId
        Wait-Process -Id $existingProcessId -ErrorAction SilentlyContinue
    }
}

$sshArguments = @(
    '-N', '-T', '-i', $resolvedKeyPath, '-p', "$SshPort",
    '-L', "127.0.0.1:${LocalPort}:127.0.0.1:8000",
    '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
    '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3',
    '-o', 'ConnectTimeout=15', $target
)
$tunnelProcess = Start-Process -FilePath 'ssh.exe' -ArgumentList $sshArguments -WindowStyle Hidden -PassThru
Set-Content -LiteralPath $tunnelPidPath -Value $tunnelProcess.Id

for ($attempt = 1; $attempt -le 120; $attempt++) {
    if ($tunnelProcess.HasExited) {
        throw "The SSH tunnel exited with code $($tunnelProcess.ExitCode)."
    }
    try {
        $response = Invoke-WebRequest `
            -Uri "http://127.0.0.1:$LocalPort/health/ready" `
            -Headers $readinessHeaders `
            -TimeoutSec 2 `
            -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            Write-Host "Deployment complete. Compute is available at http://127.0.0.1:$LocalPort."
            Write-Host "Local compute settings were saved to $localEnvironmentPath."
            exit 0
        }
    }
    catch {
        Start-Sleep -Seconds 1
    }
}

throw "The compute readiness endpoint did not become ready through local port $LocalPort."
