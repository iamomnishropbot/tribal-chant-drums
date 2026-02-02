import numpy as np
from scipy.io import wavfile
from scipy import signal

# -----------------------------
# Config – tweak for different rituals
# -----------------------------
SR = 44100              # Sample rate
DURATION = 20.0         # Total seconds
BPM = 85                # Slow, hypnotic tribal tempo
BEAT_SEC = 60 / BPM

# Drum types (procedural)
def djembe_boom(t, freq=80, decay=0.8):
    """Deep resonant hit like large djembe/base drum"""
    wave = np.sin(2 * np.pi * freq * t) * np.exp(-t / decay)
    noise = 0.15 * np.random.normal(0, 1, len(t)) * np.exp(-t / 0.4)
    return wave + noise

def slap(t):
    """Sharp attack slap/tone"""
    return 0.7 * signal.square(2 * np.pi * 220 * t) * np.exp(-t / 0.15) + \
           0.3 * np.random.normal(0, 1, len(t)) * np.exp(-t / 0.08)

def shaker(t):
    """Rattle/shaker texture"""
    return 0.4 * np.random.normal(0, 1, len(t)) * np.exp(-t / 0.6)

def frame_drum(t):
    """Medium taut drum slap"""
    return 0.6 * np.sin(2 * np.pi * 140 * t) * np.exp(-t / 0.25) + \
           0.2 * np.random.normal(0, 1, len(t)) * np.exp(-t / 0.12)

# Chant / overtone drone simulation (simple throat-singing style)
def overtone_chant(t, base_freq=110, overtone=3.0, vibrato=4.0):
    """Fundamental + strong overtone + slow vibrato for ancient monk/chant feel"""
    fund = np.sin(2 * np.pi * base_freq * t)
    overt = 0.6 * np.sin(2 * np.pi * base_freq * overtone * t + np.pi/4)
    vib = 1 + 0.02 * np.sin(2 * np.pi * vibrato * t)
    return (fund + overt) * vib * np.exp(-t * 0.05)  # slow fade

# -----------------------------
# Rhythm engine – procedural tribal patterns
# -----------------------------
def generate_rhythm(length_sec, subdivisions=16):
    """Euclidean-ish + random tribal feel: uneven spacing, accents"""
    beat_samples = int(BEAT_SEC * SR)
    total_samples = int(length_sec * SR)
    pattern = np.zeros(total_samples)
    
    # Polyrhythmic layers: 3-against-2 feel common in tribal/ancient
    for layer, density, offset in [
        ('djembe', 0.4, 0),           # Deep hits every ~2-3 beats
        ('slap',   0.6, 0.1),         # Sharp accents
        ('shaker', 0.8, 0.05),        # Constant texture
        ('frame',  0.5, 0.2)          # Mid-range fills
    ]:
        hits = np.random.choice([0,1], size=int(total_samples / beat_samples * subdivisions),
                                p=[1-density, density])
        for i, hit in enumerate(hits):
            if hit:
                start = int(i * beat_samples / subdivisions + offset * beat_samples)
                if start + 10000 > total_samples: continue
                end = min(start + 10000, total_samples)
                t_slice = np.linspace(0, (end-start)/SR, end-start)
                if layer == 'djembe':
                    pattern[start:end] += djembe_boom(t_slice)
                elif layer == 'slap':
                    pattern[start:end] += slap(t_slice)
                elif layer == 'shaker':
                    pattern[start:end] += shaker(t_slice)
                elif layer == 'frame':
                    pattern[start:end] += frame_drum(t_slice)
    
    # Normalize
    pattern /= np.max(np.abs(pattern)) * 1.1 if np.max(np.abs(pattern)) > 0 else 1
    return pattern

# -----------------------------
# Chant layer – sparse, evolving drone
# -----------------------------
def generate_chant(length_sec):
    total_samples = int(length_sec * SR)
    chant = np.zeros(total_samples)
    
    # Place 3-5 long chant "phrases"
    for start_sec in np.linspace(2, DURATION-5, 4):
        start = int(start_sec * SR)
        phrase_len = int(8 * SR)  # ~8 sec phrases
        t = np.linspace(0, phrase_len/SR, phrase_len)
        chant[start:start+phrase_len] += 0.5 * overtone_chant(t)
    
    chant /= np.max(np.abs(chant)) * 1.2 if np.max(np.abs(chant)) > 0 else 1
    return chant

# -----------------------------
# Mix & export
# -----------------------------
drums = generate_rhythm(DURATION)
chant = generate_chant(DURATION)

# Layer: drums louder, chant ghostly in background
mix = 0.8 * drums + 0.4 * chant

# Final normalize & save stereo (duplicate mono)
mix = mix / np.max(np.abs(mix)) * 0.95
stereo = np.column_stack((mix, mix)).astype(np.float32)

wavfile.write('ancient_tribal_ritual.wav', SR, stereo)
print("Saved: ancient_tribal_ritual.wav – 20s primal ceremony vibes")

## Enhanced Tribal Ritual Sound Generator 🥁🌑

Procedural audio synthesis for ancient/tribal vibes: deep djembes, pitch-bending talking drums, log thumps, polyrhythmic layers (Euclidean style), evolving overtone chants with formant "words," MIDI export, and live loop potential.

### Features
- Multiple drum types with organic variation
- True polyrhythms via Euclidean distribution
- Multi-voice throat-singing simulation (Tuvan/Mongolian-inspired)
- MIDI rhythm export for DAW use
- Outputs ~45s WAV file

### The Code
(See `enhanced_tribal_ritual.py` in this repo)

Run it locally (Python + numpy/scipy/mido):
```bash
pip install numpy scipy mido  # if needed
python enhanced_tribal_ritual.py
