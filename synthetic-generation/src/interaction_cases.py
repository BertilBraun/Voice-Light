from __future__ import annotations

from src.models import InteractionCase, InteractionCaseFile, SpeakerPrompt

DEFAULT_CASES = InteractionCaseFile(
    experiment_id="examples_v1",
    description="Initial interruption and backchannel tests",
    cases=[
        InteractionCase(
            case_id="corrective_interruption",
            title="Corrective interruption",
            description="Speaker B corrects Speaker A and takes the floor.",
            speaker_a=SpeakerPrompt(
                text=(
                    "We could take the train from the central station, because the last one "
                    "leaves at around..."
                ),
                instruction=(
                    "A casually confident young adult speaking naturally to a friend at a medium "
                    "conversational pace. The speaker is explaining a simple travel plan. Near the "
                    "end, become less certain and trail off gently, as though another person has "
                    "begun correcting the statement. Keep the delivery subtle and ordinary."
                ),
            ),
            speaker_b=SpeakerPrompt(
                text="No, they moved it to the north station.",
                instruction=(
                    "A clear conversational young adult voice entering quickly to correct an "
                    "important detail. Speak slightly faster than normal with mild urgency, "
                    "confidence, and friendliness. Begin decisively without hesitation."
                ),
            ),
            alignment_notes="Try starting B 100-300 ms before A ends.",
            tags=["interruption", "correction", "yield"],
            default_b_offset_seconds=-0.2,
        ),
        InteractionCase(
            case_id="enthusiastic_interruption",
            title="Enthusiastic collaborative interruption",
            description="Speaker B recognizes and completes Speaker A's thought with excitement.",
            speaker_a=SpeakerPrompt(
                text=(
                    "I found this tiny restaurant near the river, and apparently they make "
                    "their own..."
                ),
                instruction=(
                    "A warm, animated young adult speaking informally to a friend. Sound "
                    "genuinely pleased about a recent discovery. Use a natural medium pace and "
                    "trail off with positive energy near the end."
                ),
            ),
            speaker_b=SpeakerPrompt(
                text="The fresh pasta! I've been there.",
                instruction=(
                    "A spontaneous, energetic young adult voice suddenly recognizing what the "
                    "other speaker means. Enter quickly with genuine excitement and warmth. "
                    "Emphasize fresh pasta, then relax slightly."
                ),
            ),
            alignment_notes="Try starting B just before A's final word fully resolves.",
            tags=["interruption", "collaborative", "enthusiasm"],
            default_b_offset_seconds=-0.18,
        ),
        InteractionCase(
            case_id="backchannel",
            title="Backchannel without yielding",
            description="Speaker B gives a short acknowledgment while Speaker A keeps the floor.",
            speaker_a=SpeakerPrompt(
                text=(
                    "So I called them again this morning, and after checking the booking for a few "
                    "minutes, they finally found it under my middle name."
                ),
                instruction=(
                    "A natural conversational storyteller speaking continuously at a moderate "
                    "pace. Begin with mild frustration and transition toward relief near the end. "
                    "Include a subtle prosodic boundary after this morning, but continue the same "
                    "turn."
                ),
            ),
            speaker_b=SpeakerPrompt(
                text="Mm-hm.",
                instruction=(
                    "A very short, quiet, attentive acknowledgment from an engaged listener. "
                    "Communicate that the listener is following without attempting to take the "
                    "floor. Keep the delivery soft and brief."
                ),
            ),
            alignment_notes=(
                "Try placing B near the boundary after 'this morning'. Alternative texts: Mhm., "
                "Mm., Uh-huh., Yeah."
            ),
            tags=["backchannel", "listener", "no-yield"],
            default_b_offset_seconds=-3.0,
        ),
    ],
)
