# Sound App

Standalone audio-production workspace for the Storybook project. It reads story
content from the sibling `story_app` and keeps models, references, stems, caches,
previews, intermediate renders, and masters inside `Sound_app`.

Only final production MP3s used by the website are published to
`story_app/public/story/audio`. Do not store production work or preview audio in
the website project.

## Gotu: canonical male narrator

Gotu is the one locked male story voice. Its reference hash, OpenVoice V2 model,
conditioning, prosody, deterministic seed, and mastering settings live in
`config/gotu_voice.json`. Other production scripts must call the central
renderer instead of duplicating or adjusting these values.

The current backend is the MIT-licensed OpenVoice V2 stack with MeloTTS as its
English base synthesizer. All checkpoints and language assets are stored under
`.audio-work/openvoice/`, and inference runs locally without accepting a gated
model license or contacting a hosted speech API.

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
source .gotu-venv/bin/activate
python -m pip install -r requirements.txt
```

## Local voice studio

The studio UI runs in Next.js, with the Flask app serving its local speech
renderer. Start both processes from this `Sound_app` directory in separate
terminal tabs:

```bash
# Terminal 1: local rendering API
source .venv/bin/activate
python -m scripts.voice_qa.webapp

# Terminal 2: React frontend
npm install
npm run dev
```

Open <http://localhost:3000>. The Next.js dev server proxies `/api` and `/audio`
to the Flask API on port 5000. Flask automatically switches to the pinned
`.gotu-venv` when launched from another Python environment.

The studio requires an uploaded voice recording (at least 3 seconds, up to 25 MB).
Long recordings are accepted; only the first minute is used for voice matching.
On mobile, use the Voice settings button below the script to open the settings
drawer. Generated audio includes a waveform preview and a Download action that
saves the WAV file.
It sends that file with the script to `/api/ask`, extracts an OpenVoice target
embedding, and converts the generated speech to that voice. Clean speech from
one speaker works best. It does not fall back to Gotu when a reference is absent.

Pace controls MeloTTS speed. Variation controls acoustic and timing noise.
Expression presets adjust speed and acoustic variation; energy affects acoustic
variation and output loudness. Clarity applies a 75 Hz high-pass filter, and
auto-punctuation adds sentence endings to unpunctuated lines. These are delivery
adjustments, not a guarantee of a particular acted emotion. Change settings and
generate again to hear the result.

Reference embeddings and renders are cached in `.audio-work/studio`, keyed by
the uploaded file, script, and settings. Temporary uploads are removed after
generation. Restart Flask after backend changes; restart Next.js after changes
to its proxy configuration.

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
