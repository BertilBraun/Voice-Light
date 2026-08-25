$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Speech

$pilotDirectory = $PSScriptRoot
$sourceDirectory = Join-Path $pilotDirectory 'sources'
New-Item -ItemType Directory -Force -Path $sourceDirectory | Out-Null

$planFilenames = @('plan.json', 'interruption-plan.json')
$renderPlans = @()
foreach ($planFilename in $planFilenames) {
    $planPath = Join-Path $pilotDirectory $planFilename
    $plan = Get-Content -Raw -LiteralPath $planPath | ConvertFrom-Json
    $speakerVoice = @{}
    foreach ($speaker in $plan.speakers) {
        $speakerVoice[$speaker.speaker_id] = $speaker.voice_id
    }

    $planSourceDirectory = Join-Path $sourceDirectory $plan.plan_id
    New-Item -ItemType Directory -Force -Path $planSourceDirectory | Out-Null
    $sources = @()
    foreach ($event in $plan.events) {
        if ($event.kind -eq 'pause') {
            continue
        }
        $audioPath = Join-Path $planSourceDirectory ($event.event_id + '.wav')
        $synthesizer = [System.Speech.Synthesis.SpeechSynthesizer]::new()
        try {
            $synthesizer.SelectVoice($speakerVoice[$event.speaker_id])
            $synthesizer.SetOutputToWaveFile($audioPath)
            $synthesizer.Speak($event.text)
        }
        finally {
            $synthesizer.Dispose()
        }
        $repositoryPrefix = (Get-Location).Path + [System.IO.Path]::DirectorySeparatorChar
        $relativeAudioPath = $audioPath.Replace($repositoryPrefix, '')
        $sources += [ordered]@{
            event_id = $event.event_id
            audio_path = $relativeAudioPath
            audio_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $audioPath).Hash.ToLowerInvariant()
            provenance = [ordered]@{
                backend_id = 'windows-sapi-smoke-only'
                runtime_version = [Environment]::OSVersion.VersionString
                model_id = $speakerVoice[$event.speaker_id]
                model_revision = 'installed-voice-2026-08-25'
                model_license = 'Microsoft-Windows-component-terms-review-required'
                seed = $plan.seed
                generation_seconds = $null
                real_time_factor = $null
            }
        }
    }

    $renderPlans += [ordered]@{
        plan = $plan
        sample_rate_hz = 16000
        sources = $sources
        augmentation = [ordered]@{
            user_gain_db = -1.5
            assistant_gain_db = -0.5
            signal_to_noise_db = 28.0
            reverb_decay_seconds = 0.12
            codec = 'mulaw_8bit'
            microphone_low_hz = 100.0
            microphone_high_hz = 7200.0
            user_to_assistant_crosstalk_db = -35.0
            assistant_to_user_crosstalk_db = -32.0
        }
    }
}

$request = [ordered]@{
    corpus_id = 'sapi_speech_smoke_v1'
    description = 'Local SAPI speech smoke artifact for validating the planned-timeline pipeline.'
    plans = $renderPlans
}

$requestPath = Join-Path $pilotDirectory 'request.json'
$requestJson = $request | ConvertTo-Json -Depth 20
$utf8WithoutBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($requestPath, $requestJson, $utf8WithoutBom)
Write-Output "Wrote $requestPath"
