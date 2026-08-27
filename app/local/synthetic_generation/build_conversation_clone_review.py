from __future__ import annotations

import argparse
import html
from collections.abc import Sequence
from pathlib import Path

from app.local.synthetic_generation.conversation_prompts import (
    AssistantTurnPrompt,
    EnglishConversationPromptPlan,
    EnglishConversationPromptSet,
    UserPrompt,
)
from app.local.synthetic_generation.conversation_tts import (
    ConversationTtsManifest,
    RenderedConversationUserUnit,
)
from app.local.synthetic_generation.conversation_voice_references import (
    ConversationVoiceReference,
    load_voice_reference_manifest,
)


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = _parser().parse_args(arguments)
    render_clone_review(
        prompts_path=parsed.prompts,
        references_path=parsed.references,
        renders_path=parsed.renders,
        corpus_directory=parsed.corpus,
        output_path=parsed.output,
    )


def render_clone_review(
    prompts_path: Path,
    references_path: Path,
    renders_path: Path,
    corpus_directory: Path,
    output_path: Path,
) -> None:
    prompt_set = EnglishConversationPromptSet.model_validate_json(
        prompts_path.read_text(encoding="utf-8")
    )
    references = load_voice_reference_manifest(references_path)
    render_manifest = ConversationTtsManifest.model_validate_json(
        renders_path.read_text(encoding="utf-8")
    )
    references_by_plan = {reference.plan_id: reference for reference in references.references}
    units_by_plan = {
        plan.plan_id: tuple(
            unit for unit in render_manifest.rendered_units if unit.plan_id == plan.plan_id
        )
        for plan in prompt_set.plans
    }
    expected_plan_ids = {plan.plan_id for plan in prompt_set.plans}
    if set(references_by_plan) != expected_plan_ids:
        raise ValueError("Review references must exactly match the prompt plans.")
    if any(not units_by_plan[plan_id] for plan_id in expected_plan_ids):
        raise ValueError("Every review conversation requires rendered user audio.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sections = "".join(
        _conversation_section(
            plan,
            references_by_plan[plan.plan_id],
            units_by_plan[plan.plan_id],
            renders_path.parent,
            corpus_directory,
            output_path.parent,
        )
        for plan in prompt_set.plans
    )
    document = "".join(
        (
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width,initial-scale=1'>",
            "<title>Qwen reference and CosyVoice conversation review</title><style>",
            "body{font:16px system-ui;max-width:1120px;margin:28px auto;padding:0 18px;",
            "background:#101827;color:#e5e7eb}section,article{background:#1f2937;",
            "border:1px solid #374151;border-radius:12px;padding:16px;margin:14px 0}",
            "h1,h2,h3{color:#f9fafb}.meta{color:#a5b4c7}.tag{display:inline-block;",
            "padding:3px 9px;border-radius:999px;background:#374151;margin:0 5px 5px 0}",
            "audio{width:100%}.assistant{border-left:4px solid #60a5fa}",
            ".user{border-left:4px solid #34d399}</style></head><body>",
            "<h1>Qwen references → CosyVoice conversations</h1>",
            "<p>Each section compares the original Qwen VoiceDesign reference with a complete ",
            "CosyVoice-rendered conversation conditioned on that exact reference. Assistant ",
            "turns are intentionally inaudible. No generated item has been filtered.</p>",
            sections,
            "</body></html>",
        )
    )
    output_path.write_text(document, encoding="utf-8")


def _conversation_section(
    plan: EnglishConversationPromptPlan,
    reference: ConversationVoiceReference,
    rendered_units: tuple[RenderedConversationUserUnit, ...],
    render_directory: Path,
    corpus_directory: Path,
    review_directory: Path,
) -> str:
    units_by_id = {unit.prompt.unit_id: unit for unit in rendered_units}
    if len(units_by_id) != len(rendered_units):
        raise ValueError(f"Conversation {plan.plan_id} repeats a rendered unit ID.")
    source_audio_path = corpus_directory / "conversations" / plan.plan_id / "source.wav"
    if not source_audio_path.exists():
        raise ValueError(f"Compiled conversation audio is missing for {plan.plan_id}.")
    reference_url = _relative_url(reference.audio_path, review_directory)
    source_url = _relative_url(source_audio_path, review_directory)
    ordered_elements = sorted(
        (*plan.assistant_turns, *plan.user_prompts),
        key=lambda element: element.sequence_index,
    )
    timeline = "".join(
        _timeline_card(element, units_by_id, render_directory, review_directory)
        for element in ordered_elements
    )
    voice = plan.base_user_voice
    return "".join(
        (
            f"<section><h2>{html.escape(plan.topic)}</h2>",
            f"<p class='meta'>{html.escape(plan.plan_id)} · {html.escape(plan.domain.value)} · ",
            f"{html.escape(voice.perceived_age.value)} · {html.escape(voice.accent.value)} · ",
            f"{html.escape(voice.pitch.value)} pitch · ",
            f"{html.escape(voice.vocal_weight.value)} voice",
            "</p><article><h3>Qwen VoiceDesign reference</h3>",
            f"<p>{html.escape(reference.reference_text)}</p>",
            f"<p class='meta'>{html.escape(reference.voice_instruction)}</p>",
            f"<audio controls preload='none' src='{html.escape(reference_url)}'></audio></article>",
            "<article><h3>Complete CosyVoice conversation</h3>",
            "<p class='meta'>Silent spans represent assistant speech probability, ",
            "not user holds.</p>",
            f"<audio controls preload='none' src='{html.escape(source_url)}'></audio></article>",
            "<h3>Exact conversation plan and individual clone renders</h3>",
            timeline,
            "</section>",
        )
    )


def _timeline_card(
    element: AssistantTurnPrompt | UserPrompt,
    units_by_id: dict[str, RenderedConversationUserUnit],
    render_directory: Path,
    review_directory: Path,
) -> str:
    match element:
        case AssistantTurnPrompt():
            return "".join(
                (
                    "<article class='assistant'><span class='tag'>assistant reply</span>",
                    "<span class='tag'>inaudible</span>",
                    f"<p>{html.escape(element.text)}</p></article>",
                )
            )
        case _:
            unit = units_by_id[element.unit_id]
            audio_path = render_directory / unit.clip.audio_path
            audio_url = _relative_url(audio_path, review_directory)
            return "".join(
                (
                    "<article class='user'><span class='tag'>",
                    f"{html.escape(element.condition)}</span>",
                    f"<span class='tag'>{html.escape(element.delivery.pace.value)}</span>",
                    f"<span class='tag'>{html.escape(element.delivery.affect.value)}</span>",
                    f"<p>{html.escape(str(element.text))}</p>",
                    f"<audio controls preload='none' src='{html.escape(audio_url)}'>",
                    "</audio></article>",
                )
            )


def _relative_url(path: Path, review_directory: Path) -> str:
    if not path.exists():
        raise ValueError(f"Review audio is missing: {path}")
    return path.resolve().relative_to(review_directory.resolve()).as_posix()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a reference-versus-CosyVoice conversation listening page."
    )
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--renders", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


if __name__ == "__main__":
    main()
