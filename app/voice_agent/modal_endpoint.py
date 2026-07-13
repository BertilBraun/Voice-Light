from __future__ import annotations

from pathlib import Path

import modal
from fastapi import FastAPI, WebSocket

from app.voice_agent.session import SessionPolicy, VoiceAgentSession, send_session_error

MODEL_CACHE_DIRECTORY = "/model-cache"
COSYVOICE_DIRECTORY = Path(MODEL_CACHE_DIRECTORY) / "CosyVoice2"
PROMPT_AUDIO_PATH = Path("/opt/CosyVoice/asset/zero_shot_prompt.wav")
HUGGING_FACE_CACHE_DIRECTORY = f"{MODEL_CACHE_DIRECTORY}/huggingface"
MODELSCOPE_CACHE_DIRECTORY = f"{MODEL_CACHE_DIRECTORY}/modelscope"
TORCH_CACHE_DIRECTORY = f"{MODEL_CACHE_DIRECTORY}/torch"

model_cache = modal.Volume.from_name("voice-light-agent-model-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10")
    .entrypoint([])
    .apt_install(
        "build-essential",
        "clang",
        "ffmpeg",
        "git",
        "git-lfs",
        "libsndfile1",
        "libsox-dev",
        "sox",
    )
    .run_commands(
        "git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice",
        "pip install --upgrade pip 'setuptools<81' wheel",
        "pip install --no-build-isolation openai-whisper==20231117",
        "pip install -r /opt/CosyVoice/requirements.txt",
        "python -m venv /opt/nemotron-venv",
        "/opt/nemotron-venv/bin/pip install "
        "'numpy>=1.26,<2' 'pydantic>=2.0' torch==2.6.0 "
        "'transformers>=5.13.0,<5.14.0'",
        "/opt/nemotron-venv/bin/pip install librosa==0.10.2",
    )
    .uv_pip_install(
        "fastapi[standard]>=0.115.0",
        "huggingface-hub>=0.30.0,<1.0",
        "pydantic>=2.0.0",
        "silero-vad>=6.2.1",
        "torch==2.3.1",
        "torchaudio==2.3.1",
        "transformers==4.51.3",
    )
    .env(
        {
            "HF_HOME": HUGGING_FACE_CACHE_DIRECTORY,
            "HF_HUB_CACHE": f"{HUGGING_FACE_CACHE_DIRECTORY}/hub",
            "HUGGINGFACE_HUB_CACHE": f"{HUGGING_FACE_CACHE_DIRECTORY}/hub",
            "MODELSCOPE_CACHE": MODELSCOPE_CACHE_DIRECTORY,
            "PYTHONPATH": "/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS",
            "TORCH_HOME": TORCH_CACHE_DIRECTORY,
            "XDG_CACHE_HOME": MODEL_CACHE_DIRECTORY,
        }
    )
    .add_local_python_source("app")
)

app = modal.App("VoiceLightAgent")


@app.cls(
    image=image,
    gpu="L40S",
    min_containers=0,
    max_containers=1,
    scaledown_window=30,
    timeout=43_200,
    volumes={MODEL_CACHE_DIRECTORY: model_cache},
)
@modal.concurrent(max_inputs=1)
class VoiceAgentServer:
    @modal.enter()
    def setup(self) -> None:
        from huggingface_hub import snapshot_download

        from app.voice_agent.nemotron_client import NemotronStreamingTranscriber
        from app.voice_agent.runtime import (
            CosyVoiceSpeechSynthesizer,
            SileroSpeechDetector,
            TransformersLanguageModel,
        )

        snapshot_download(
            "FunAudioLLM/CosyVoice2-0.5B",
            local_dir=COSYVOICE_DIRECTORY,
        )
        self.speech_detector_type = SileroSpeechDetector
        self.transcriber = NemotronStreamingTranscriber()
        self.language_model = TransformersLanguageModel()
        self.speech_synthesizer = CosyVoiceSpeechSynthesizer(
            model_directory=COSYVOICE_DIRECTORY,
            prompt_audio_path=PROMPT_AUDIO_PATH,
        )
        model_cache.commit()

    @modal.asgi_app()
    def web_app(self) -> FastAPI:
        web_app = FastAPI(title="Voice Light Streaming Agent")

        @web_app.websocket("/session")
        async def session_endpoint(websocket: WebSocket) -> None:
            try:
                session = VoiceAgentSession(
                    websocket=websocket,
                    speech_detector=self.speech_detector_type(),
                    transcriber=self.transcriber,
                    language_model=self.language_model,
                    speech_synthesizer=self.speech_synthesizer,
                    policy=SessionPolicy(),
                )
                await session.run()
            except Exception as error:
                await send_session_error(websocket=websocket, error=error)
                await websocket.close(code=1011)

        return web_app
