# Quick Start Guide - Sound_App with audio-venv

## ✅ Your Environment is Ready!

**Virtual Environment**: `audio-venv` with Python 3.14.0  
**Location**: `c:\Users\bhushan.bhosale\Documents\Sound_App\audio-venv`

---

## 🚀 Activate the Environment

**PowerShell:**
```powershell
cd c:\Users\bhushan.bhosale\Documents\Sound_App
.\audio-venv\Scripts\Activate.ps1
```

**Git Bash / WSL:**
```bash
cd /c/Users/bhushan.bhosale/Documents/Sound_App
source audio-venv/Scripts/activate
```

**When activated**, you'll see `(audio-venv)` at the start of your prompt.

---

## 📦 Installed Packages (38 total)

### Audio Processing Core:
- **NumPy** 2.5.2 - Array operations
- **SciPy** 1.18.1 - Signal processing  
- **Librosa** 1.0.0 - Audio analysis
- **Soundfile** 0.14.0 - Audio I/O
- **lameenc** 1.8.4 - MP3 encoding
- **pyloudnorm** 0.2.0 - Audio loudness

### Text-to-Speech:
- **pyttsx3** 2.99 - Offline TTS (Windows voices)
- **gTTS** 2.5.4 - Google TTS (online)
- **edge-tts** 7.2.8 - Microsoft Edge TTS (online)

### Supporting Libraries:
- numba, scikit-learn, joblib, and more...

---

## 🎵 Using Your Scripts

### Run Existing Scripts:
```bash
# Activate environment first
.\audio-venv\Scripts\Activate.ps1

# Run any script
python Sound_app/scripts/audio_dsp.py
python Sound_app/scripts/generate_story_audio.py
```

### Test TTS:
```bash
python Sound_app/scripts/tts_alternatives.py
```

### Import in Your Code:
```python
# Audio processing
import numpy as np
import librosa
import soundfile as sf
from audio_dsp import loop_with_crossfade

# Load audio
audio, sr = librosa.load('input.wav')

# Process
processed = loop_with_crossfade(audio, 44100, sr, 2.0)

# Save
sf.write('output.wav', processed, sr)
```

### TTS Examples:
```python
from tts_alternatives import use_pyttsx3, use_gtts

# Offline (Windows voices)
use_pyttsx3("Hello world", "output1.mp3")

# Online (Google)
use_gtts("Hello world", "output2.mp3")
```

---

## 📁 Project Structure

```
Sound_App/
├── audio-venv/              # Virtual environment (Python 3.14)
├── Sound_app/
│   ├── scripts/             # Your Python scripts
│   │   ├── audio_dsp.py    # DSP utilities
│   │   ├── tts_alternatives.py  # TTS examples
│   │   └── ...
│   ├── output/              # Generated files
│   └── requirements.txt     # Original requirements
├── INSTALLATION_STATUS.md   # Detailed setup info
└── QUICK_START.md          # This file
```

---

## 🔧 Common Commands

```bash
# Check installed packages
pip list

# Install additional package
pip install package-name

# Update a package
pip install --upgrade package-name

# Deactivate environment
deactivate

# Test Python is working
python -c "import librosa, numpy; print('Ready!')"
```

---

## 💡 Tips

1. **Always activate** the environment before running scripts
2. **pyttsx3** works offline - best for local development
3. **gTTS** requires internet but has natural voices
4. Your existing scripts using numpy, scipy, librosa will work as-is
5. See [tts_alternatives.py](Sound_app/scripts/tts_alternatives.py) for TTS usage

---

## 🆘 Troubleshooting

**Import errors?**
- Make sure environment is activated
- Check `(audio-venv)` appears in prompt

**Script not found?**
- Use full path: `python Sound_app/scripts/your_script.py`

**Need help?**
- Check [INSTALLATION_STATUS.md](INSTALLATION_STATUS.md) for details
- View package list: `pip list`

---

## ✅ You're All Set!

Your audio processing environment is fully configured and tested.  
Start coding! 🎉
