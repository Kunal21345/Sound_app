# Sound production rules

## Canonical male narrator: Gotu

Whenever a request calls for Gotu, a male story narrator, or the established
male narrator voice, use `scripts/render_gotu_voice.py` or import
`scripts/gotu_voice.py`. Do not create another male voice profile.

- Gotu is locked to `config/gotu_voice.json` and the exact reference hash in
  that file. Do not substitute, trim, enhance, or overwrite the reference.
- Do not ask the user for voice settings. Gotu already fixes the model,
  conditioning, prosody, seed, speed, loudness, and output format.
- Reuse the content-addressed cache. Render only missing requested text; never
  regenerate a cache hit.
- Keep Gotu narration dry and centered. Add music or effects only in a separate
  production mix when the user explicitly requests them.
- Do not create preview variants, raw-edit chains, or full-story alternatives
  unless the user explicitly asks for them.
- All references, caches, intermediates, and renders stay in `Sound_app`.
  Publish only final website-ready tracks to `story_app/public/story/audio`.
- Running the renderer with no text is a read-only audit and must not create
  audio.
