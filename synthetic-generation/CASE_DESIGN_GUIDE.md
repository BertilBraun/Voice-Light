# Synthetic Conversation Case Design Guide

This guide defines how LLM-authored case files should describe synthetic two-speaker
conversation samples. The goal is not just plausible text. The generated tracks must make
sense when overlaid by the placement system.

## General Case Rules

Each case represents one conversational event type:

- `backchannel`
- `interruption`
- `completion`
- `internal_pause`

Use exactly one typed `placement` object per case. Do not mix event types inside one case.

Speaker text should be long enough to give the TTS model local context. Prefer one or two
natural sentences for Speaker A. Keep Speaker B as long as the event requires:

- backchannels are short
- interruptions are usually one short corrective or collaborative clause
- completions may be a full next turn
- internal-pause cases usually only need Speaker A

Use ElevenLabs v3 bracket tags directly in `speaker_a.text` and `speaker_b.text` for
delivery. These tags should annotate emotion, pacing, prominence, and interaction role.
Examples:

- `[calm, conversational]`
- `[quickly, correcting]`
- `[quietly, attentive]`
- `[thinking aloud]`
- `[brief pause]`

Do not put hidden backend mappings or provider-specific transformations in code. The LLM
must write the exact text and tags that should be sent to the TTS backend.

Avoid:

- ellipses for completions or handoffs
- long theatrical stage directions
- multiple unrelated events in one case
- anchors that are not exact or near-exact substrings of Speaker A
- Speaker A continuing normally after an interruption

## Backchannel

A backchannel is a short listener acknowledgment while Speaker A keeps the floor.

Speaker A must make an acknowledgeable statement before the anchor. The anchor should end
after a complete idea, not after a setup fragment.

Good anchor:

```text
I moved the budget review to next week
```

Bad anchor:

```text
So I called them again this morning
```

Speaker B should be low-prominence and brief:

```text
[quietly, attentive] Mm-hm.
```

Placement:

```json
{
  "type": "backchannel",
  "anchor_text": "I moved the budget review to next week",
  "delay_ms": 300
}
```

The default delay should usually be 200-400 ms after the anchor ends. Speaker A continues
after the backchannel.

## Interruption

An interruption is not just overlap. Speaker A must stop because Speaker B entered.

Speaker A should end at the interrupted/decompleted point, but the placement anchor must
include enough prior speech for Speaker B to plausibly understand what is being corrected
or completed. Do not anchor before the disambiguating word or phrase has been spoken.

For a correction of "Thursday morning", anchoring on "by Thursday" is too early. Speaker B
cannot know whether the issue is the day, the time of day, or the whole deadline until
"morning" has been heard. Anchor on the full phrase plus one complete continuation word,
then let Speaker A begin the next word before being cut off.

Do not write a full sentence where Speaker A continues with several more words after the
intended interruption. Use a dash or an explicit short fragment at the end if the backend
handles it naturally.

Good Speaker A:

```text
[conversational, focused] I think we can send the contract draft by Thursday morning, if leg-
```

Good Speaker B:

```text
[quickly, correcting] Friday morning is safer.
```

Bad Speaker A:

```text
I think we can send the contract draft by Thursday morning if legal comes back today and the pricing table stays the same-
```

That is wrong for an interruption because Speaker A keeps talking long after the correction
should have stopped the turn. Also avoid anchoring too early:

```json
{
  "type": "interruption",
  "anchor_text": "by Thursday",
  "mode": "at_anchor_end",
  "lead_ms": 80
}
```

That is too early when the correction depends on the phrase "Thursday morning".

Placement should usually be at the end of the disambiguating phrase plus one complete
continuation word. The overlap should happen around the next cut-off word, not before the
meaning is recoverable.

```json
{
  "type": "interruption",
  "anchor_text": "by Thursday morning if",
  "mode": "at_anchor_end",
  "lead_ms": 0
}
```

The overlap should feel causal: B enters, and A stops almost immediately.

## Completion

A completion or handoff is a normal next speaker turn after Speaker A finishes.

Speaker A should end as a complete turn. Do not use ellipses or trailing-off punctuation,
because those often make the TTS model stretch the final words. Speaker B should sound like
a normal next turn, not an interruption.

Placement:

```json
{
  "type": "completion",
  "pause": "short"
}
```

Pause ranges:

- `short`: 200-500 ms
- `medium`: 500-1000 ms
- `long`: 1000-2000 ms

Use `short` for ordinary handoffs and `long` for questions or thinking time.

## Internal Pause

An internal pause is a pause inside a single speaker's coherent turn. Speaker B is not
placed by this type.

Speaker A should include enough context before and after the pause. The pause should occur
after a meaningful phrase that can be located by ASR.

Example:

```text
[thinking aloud] I checked the incident timeline again, and the restart happened right
after the backup job finished. [brief pause] That makes me think the scheduler fired
correctly.
```

Placement:

```json
{
  "type": "internal_pause",
  "anchor_text": "the restart happened right after the backup job finished",
  "pause": "medium"
}
```

The output metadata records the measured ASR gap after the anchor and the next word start.

## Raw And Trimmed Audio

Generation stores both:

- `raw_audio_url`: the original backend WAV before cleanup
- `audio_url`: the trimmed WAV used for canonical placement

Use raw audio to diagnose TTS behavior. Use trimmed audio for placement experiments only
after the trim behavior has been approved for the batch.
