from __future__ import annotations

from dataclasses import dataclass

from app.local.analyses.end_of_turn.detectors.livekit_text_turn import (
    CandidateTranscriptTurn,
    ChatMessagePayload,
    TranscriptTurn,
    TranscriptWord,
    _candidate_transcript_turns,
    _format_chat_messages,
    _livekit_text_turn_events,
    _normalized_chat_messages,
    _speaker_speech_segments,
    _transcript_turn,
)
from app.local.analyses.end_of_turn.service import SpeechSegment


@dataclass(frozen=True)
class FixedScorer:
    probability: float

    def completion_probability(self, chat_messages: list[ChatMessagePayload]) -> float:
        return self.probability


@dataclass(frozen=True)
class TemplateTokenizer:
    def apply_chat_template(
        self,
        conversation: list[ChatMessagePayload],
        add_generation_prompt: bool,
        add_special_tokens: bool,
        tokenize: bool,
    ) -> str:
        rendered_messages = [
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>"
            for message in conversation
        ]
        return "".join(rendered_messages)


def test_candidate_transcript_turns_use_speaker_gaps_and_recent_chat_context() -> None:
    transcript_turns = [
        TranscriptTurn(
            speaker="Speaker1",
            text="I need to think",
            start_seconds=0.0,
            end_seconds=1.0,
            words=(),
        ),
        TranscriptTurn(
            speaker="Speaker2",
            text="okay",
            start_seconds=1.1,
            end_seconds=1.4,
            words=(),
        ),
        TranscriptTurn(
            speaker="Speaker1",
            text="done now",
            start_seconds=2.0,
            end_seconds=3.0,
            words=(),
        ),
    ]

    candidates = _candidate_transcript_turns(
        transcript_turns=transcript_turns,
        speaker_label="Speaker1",
        audio_duration_seconds=4.0,
        candidate_silence_seconds=0.5,
    )

    assert len(candidates) == 2
    assert candidates[0].evaluate_seconds == 1.5
    assert candidates[0].following_silence_seconds == 1.0
    assert candidates[0].chat_messages == [
        {"role": "user", "content": "I need to think"},
    ]
    assert candidates[1].evaluate_seconds == 3.5
    assert candidates[1].following_silence_seconds == 1.0
    assert candidates[1].chat_messages == [
        {"role": "user", "content": "I need to think"},
        {"role": "assistant", "content": "okay"},
        {"role": "user", "content": "done now"},
    ]


def test_livekit_text_turn_events_gate_candidate_turns_by_probability() -> None:
    candidate_turns = [
        CandidateTranscriptTurn(
            speech_start_seconds=1.0,
            speech_end_seconds=2.0,
            evaluate_seconds=2.5,
            following_silence_seconds=1.5,
            chat_messages=[{"role": "user", "content": "hello"}],
        )
    ]

    accepted_events = _livekit_text_turn_events(
        candidate_turns=candidate_turns,
        scorer=FixedScorer(probability=0.02),
        completion_threshold=0.011,
    )
    rejected_events = _livekit_text_turn_events(
        candidate_turns=candidate_turns,
        scorer=FixedScorer(probability=0.005),
        completion_threshold=0.011,
    )

    assert len(accepted_events) == 1
    assert accepted_events[0].time_seconds == 2.5
    assert rejected_events == []


def test_format_chat_messages_normalizes_and_removes_final_end_token() -> None:
    chat_messages: list[ChatMessagePayload] = [
        {"role": "user", "content": " Hello, THERE! "},
        {"role": "user", "content": "Are--you there?"},
        {"role": "assistant", "content": "Yes."},
    ]

    normalized_messages = _normalized_chat_messages(chat_messages=chat_messages)
    formatted_text = _format_chat_messages(
        tokenizer=TemplateTokenizer(),
        chat_messages=chat_messages,
    )

    assert normalized_messages == [
        {"role": "user", "content": "hello there are--you there"},
        {"role": "assistant", "content": "yes"},
    ]
    assert formatted_text == (
        "<|im_start|>user\nhello there are--you there<|im_end|><|im_start|>assistant\nyes"
    )


def test_transcript_turn_uses_word_timings_clipped_to_analysis_duration() -> None:
    turn = {
        "speaker": "Speaker1",
        "text": "hello after cap",
        "startTime": 0.0,
        "endTime": 4.0,
        "words": [
            {"text": "hello", "startTime": 0.2, "endTime": 0.5},
            {"text": "after", "startTime": 3.0, "endTime": 3.5},
        ],
    }

    transcript_turn = _transcript_turn(
        turn=turn,
        max_duration_seconds=2.0,
        speaker="Speaker1",
        offset_seconds=0.0,
    )

    assert transcript_turn == TranscriptTurn(
        speaker="Speaker1",
        text="hello",
        start_seconds=0.2,
        end_seconds=0.5,
        words=(TranscriptWord(text="hello", start_seconds=0.2, end_seconds=0.5),),
    )


def test_speaker_speech_segments_returns_target_speaker_turns() -> None:
    transcript_turns = [
        TranscriptTurn(
            speaker="Speaker1",
            text="target",
            start_seconds=0.0,
            end_seconds=1.0,
            words=(),
        ),
        TranscriptTurn(
            speaker="Speaker2",
            text="other",
            start_seconds=1.1,
            end_seconds=2.0,
            words=(),
        ),
    ]

    speech_segments = _speaker_speech_segments(
        transcript_turns=transcript_turns,
        speaker_label="Speaker1",
    )

    assert speech_segments == [SpeechSegment(start_seconds=0.0, end_seconds=1.0)]
