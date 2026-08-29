"""
Alternative TTS solutions for Python 3.14+
Since Coqui TTS requires Python < 3.12, use these alternatives.
"""

# Option 1: pyttsx3 - Offline, uses Windows SAPI voices
def use_pyttsx3(text, output_file):
    """Offline TTS using Windows system voices"""
    import pyttsx3
    engine = pyttsx3.init()

    # Optional: Configure voice properties
    engine.setProperty('rate', 150)    # Speed
    engine.setProperty('volume', 0.9)  # Volume 0-1

    # Get available voices
    voices = engine.getProperty('voices')
    # engine.setProperty('voice', voices[0].id)  # Change voice if needed

    engine.save_to_file(text, output_file)
    engine.runAndWait()
    print(f"[OK] Saved to {output_file} using pyttsx3")


# Option 2: gTTS - Google Text-to-Speech (requires internet)
def use_gtts(text, output_file):
    """Online TTS using Google's service"""
    from gtts import gTTS

    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(output_file)
    print(f"[OK] Saved to {output_file} using gTTS")


# Option 3: edge-tts - Microsoft Edge TTS (requires internet, high quality)
async def use_edge_tts(text, output_file):
    """Online TTS using Microsoft Edge's service (best quality)"""
    import edge_tts

    # Available voices: en-US-AriaNeural, en-US-GuyNeural, etc.
    communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")
    await communicate.save(output_file)
    print(f"[OK] Saved to {output_file} using edge-tts")


# Synchronous wrapper for edge-tts
def use_edge_tts_sync(text, output_file):
    """Synchronous wrapper for edge-tts"""
    import asyncio
    asyncio.run(use_edge_tts(text, output_file))


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Example usage
    test_text = "Hello! This is a test of the text to speech system."
    output_dir = Path(__file__).parent.parent / "output" / "tts_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Testing TTS alternatives...\n")

    # Test 1: pyttsx3 (offline)
    print("1. Testing pyttsx3 (offline, Windows voices)...")
    try:
        use_pyttsx3(test_text, str(output_dir / "test_pyttsx3.mp3"))
    except Exception as e:
        print(f"   Error: {e}")

    # Test 2: gTTS (online)
    print("\n2. Testing gTTS (Google TTS, requires internet)...")
    try:
        use_gtts(test_text, str(output_dir / "test_gtts.mp3"))
    except Exception as e:
        print(f"   Error: {e}")

    # Test 3: edge-tts (online, best quality)
    print("\n3. Testing edge-tts (Microsoft Edge TTS, requires internet)...")
    try:
        use_edge_tts_sync(test_text, str(output_dir / "test_edge.mp3"))
    except Exception as e:
        print(f"   Error: {e}")

    print(f"\n[OK] All tests complete! Check {output_dir}")
