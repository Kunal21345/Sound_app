# Sound App

Standalone audio-production workspace for the Storybook project. It reads story
content from the sibling `story_app` and keeps models, references, stems, caches,
previews, intermediate renders, and masters inside `Sound_app`.

Only final production MP3s used by the website are published to
`story_app/public/story/audio`. Do not store production work or preview audio in
the website project.

## Gotu: canonical male narrator

Gotu is the one locked male story voice. Its reference hash, XTTS model,
conditioning, prosody, deterministic seed, and mastering settings live in
`config/gotu_voice.json`. Other production scripts must call the central
renderer instead of duplicating or adjusting these values.

```bash
# Read-only health check; creates no sound files.
python scripts/render_gotu_voice.py

# One-pass local render. A repeated line is returned from cache.
python scripts/render_gotu_voice.py "Narration to speak"

# Create one explicitly named deliverable inside Sound_app.
python scripts/render_gotu_voice.py "Narration to speak" \
  --output .audio-work/gotu/renders/narration.mp3
```

The CLI automatically switches to the pinned local Python environment when the
current interpreter does not have the exact packages. It exposes no voice-style
override flags. Audio stays dry so final music and SFX can be mixed separately.

## Setup

Python 3.9 is recommended because the existing XTTS environment was built with
that version.

```bash
cd Sound_app
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

By default, the apps should be siblings:

```text
storybook/
├── Sound_app/
└── story_app/
```

If the website is elsewhere, set `STORY_APP_ROOT` to its absolute path before
running a command.

## Production commands

Run commands from this directory:

```bash
python scripts/generate_story_audio.py
python scripts/enhance_story_audio.py
python scripts/generate_full_musical_story.py
python scripts/prepare_seed_vc_guides.py
python scripts/build_soulx_score_metadata.py
python scripts/mix_story_one_background.py
python scripts/mix_seed_vc_preview.py VOCAL_PATH OUTPUT_PATH
```

### Story 2 optimized workflow

Story 2 is intentionally gated to prevent accidental full renders:

```bash
# 1. Read-only cache audit. This is also the default command.
python scripts/generate_moonlit_waterfall_audio.py

# 2. Mix and publish using cached narration only.
python scripts/generate_moonlit_waterfall_audio.py --full

# 3. Only when the audit reports missing chunks and approval is explicit.
python scripts/generate_moonlit_waterfall_audio.py --full --allow-synthesis
```

Add `--keep-master` only when an uncompressed WAV is genuinely needed. Normal
production runs create an MP3 only. XTTS is imported only when synthesis is
explicitly allowed and a cached chunk is actually missing. Missing Story 2
chunks now route through Gotu; the approved existing chunks are reused without
conversion or re-synthesis.

Generated work, models, source stems, and caches live in `.audio-work/`. Finished
story tracks are written to the sibling website's `public/story/audio/` folder.
Only app-ready production deliverables belong there.

## Project layout

```text
Sound_app/
├── .audio-work/          # models, references, caches, chunks, and masters
├── config/               # locked voice profiles
└── scripts/              # sound-production code

story_app/public/story/audio/ # final production MP3s only
```

## Existing environments

The former `.audio-venv` and `.audio-venv39` directories are preserved here for
reference. Python virtual environments can contain absolute paths, so create the
fresh `.venv` shown above for reliable use after the move.
