"""Speech-in / expressive-speech-out Q&A pipeline.

Flow: user audio -> local Whisper STT -> Claude (intent + answer + expression)
-> expressive Gotu render -> answer audio.

This package never edits config/gotu_voice.json or the canonical Gotu cache;
expressive renders live in their own cache namespace (see render.py).
"""
