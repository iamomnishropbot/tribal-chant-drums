import numpy as np
from scipy.io import wavfile
from scipy import signal
import mido  # For MIDI export (available in many STEM envs; if not, comment out MIDI part)
import random

# -----------------------------
# Config
# -----------------------------
SR = 44100
DURATION = 45.0          # Longer for full ritual feel
BPM = 78                 # Deep, meditative
BEAT_SEC = 60 / BPM

# -----------------------------
# Enhanced Drums
# -----------------------------
def djembe_boom(t, freq=75, decay=1.2):
    wave = np.sin(2 * np.pi * freq * t) * np.exp(-t / decay)
    noise = 0.18 * np.random.normal(0, 1, len(t)) * np.exp(-t / 0.5)
    return wave + noise

def talking_drum_glide(t, start_freq=90, end_freq=180, duration=0.6):
    """Pitch bend like talking drum"""
    freq = np.linspace(start_freq, end_freq, len(t))
    wave = np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-t / duration)
    return 0.8 * wave + 0.1 * np.random.normal(0, 1, len(t)) * np.exp(-t / 0.2)

def log_drum_thump(t):
    return 0.75 * np.sin(2 * np.pi * 50 * t) * np.exp(-t / 0.4) + \
           0.25 * signal.sawtooth(2 * np.pi * 30 * t) * np.exp(-t / 0.3)

def slap(t):
    return 0.7 * signal.square(2 * np.pi * 250 * t) * np.exp(-t / 0.12)

def shaker(t):
    return 0.45 * np.random.normal(0, 1, len(t)) * np.exp(-t / 0.7)

def frame_drum(t):
    return 0.65 * np.sin(2 * np.pi * 130 * t) * np.exp(-t / 0.28)

# -----------------------------
# Enhanced Chant: multi-voice + formant-like "words"
# -----------------------------
def overtone_chant_phrase(t, base_freq=105, overtones=[2.5, 4.1, 6.3], vibrato=3.8):
    fund = np.sin(2 * np.pi * base_freq * t)
    ovs = sum(0.4 * np.sin(2 * np.pi * base_freq * ot * t + np.pi * i / 3) for i, ot in enumerate(overtones))
    vib = 1 + 0.025 * np.sin(2 * np.pi * vibrato * t)
    formant_mod = 1 + 0.15 * np.sin(2 * np.pi * 0.8 * t)  # slow "vowel" shift
    return (fund + ovs) * vib * formant_mod * np.exp(-t * 0.04)

def generate_chant(length_sec):
    total_samples = int(length_sec * SR)
    chant = np.zeros(total_samples)
    phrase_starts = np.linspace(3, DURATION-8, 5)  # More phrases
    for start_sec in phrase_starts:
        start = int(start_sec * SR)
        phrase_len = int(random.uniform(6, 12) * SR)
        t = np.linspace(0, phrase_len/SR, phrase_len)
        chant[start:start+phrase_len] += 0.45 * overtone_chant_phrase(t)
        # Add subtle second voice harmony occasionally
        if random.random() > 0.6:
            chant[start:start+phrase_len] += 0.3 * overtone_chant_phrase(t, base_freq=base_freq*1.1)
    chant /= np.max(np.abs(chant)) * 1.3 if np.max(np.abs(chant)) > 0 else 1
    return chant

# -----------------------------
# Euclidean Rhythm Generator
# -----------------------------
def euclidean_pattern(steps, onsets):
    """Bresenham-like Euclidean distribution"""
    if onsets == 0: return [0] * steps
    pattern = [1] * onsets + [0] * (steps - onsets)
    # Rotate for variety
    rotate = random.randint(0, steps-1)
    pattern = pattern[rotate:] + pattern[:rotate]
    return pattern

def generate_polyrhythm(length_sec):
    beat_samples = int(BEAT_SEC * SR)
    total_samples = int(length_sec * SR)
    mix = np.zeros(total_samples)
    
    # Layers with Euclidean patterns
    layers = [
        {'name': 'djembe', 'pulses': 16, 'onsets': 5, 'density': 0.45, 'func': djembe_boom, 'decay': 1.2},
        {'name': 'talking', 'pulses': 12, 'onsets': 4, 'density': 0.35, 'func': talking_drum_glide},
        {'name': 'log', 'pulses': 8, 'onsets': 3, 'density': 0.3, 'func': log_drum_thump},
        {'name': 'slap', 'pulses': 24, 'onsets': 9, 'density': 0.55, 'func': slap},
        {'name': 'shaker', 'pulses': 32, 'onsets': 20, 'density': 0.75, 'func': shaker},
        {'name': 'frame', 'pulses': 16, 'onsets': 6, 'density': 0.5, 'func': frame_drum}
    ]
    
    for layer in layers:
        pat = euclidean_pattern(layer['pulses'], layer['onsets'])
        for beat in range(int(total_samples / beat_samples)):
            for sub in range(layer['pulses']):
                if pat[sub % len(pat)]:
                    offset = random.uniform(-0.02, 0.02) * beat_samples  # humanize
                    start = int(beat * beat_samples + sub * beat_samples / layer['pulses'] + offset)
                    dur_samples = int(0.6 * beat_samples) if 'talking' in layer['name'] else int(0.3 * beat_samples)
                    end = min(start + dur_samples, total_samples)
                    t_slice = np.linspace(0, (end-start)/SR, end-start)
                    hit = layer['func'](t_slice, decay=layer.get('decay', 0.4))
                    mix[start:end] += hit * layer['density']
    
    mix /= np.max(np.abs(mix)) * 1.15 if np.max(np.abs(mix)) > 0 else 1
    return mix

# -----------------------------
# MIDI Export (Drum Pattern Only)
# -----------------------------
def export_midi_rhythm(filename='tribal_ritual.mid'):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(BPM), time=0))
    
    # Simple mapping: djembe=36 (bass), slap=38 (snare), etc.
    drum_map = {'djembe': 36, 'talking': 41, 'log': 35, 'slap': 38, 'frame': 42}
    
    # Generate a short pattern for MIDI (one cycle example)
    for layer in ['djembe', 'talking', 'log', 'slap', 'frame']:
        pat = euclidean_pattern(16, random.randint(4,7))
        for i, hit in enumerate(pat):
            if hit:
                track.append(mido.Message('note_on', note=drum_map.get(layer, 36), velocity=100, time=i*240))  # 240 ticks/beat approx
                track.append(mido.Message('note_off', note=drum_map.get(layer, 36), velocity=0, time=120))
    
    mid.save(filename)
    print(f"MIDI saved: {filename}")

# -----------------------------
# Build & Export
# -----------------------------
drums = generate_polyrhythm(DURATION)
chant = generate_chant(DURATION)

mix = 0.85 * drums + 0.45 * chant
mix = mix / np.max(np.abs(mix)) * 0.98
stereo = np.column_stack((mix, mix)).astype(np.float32)

wavfile.write('enhanced_ancient_tribal_ritual.wav', SR, stereo)
print("WAV saved: enhanced_ancient_tribal_ritual.wav – 45s deep ritual with polyrhythms & evolving chants")

# Optional MIDI
try:
    export_midi_rhythm()
except Exception as e:
    print("MIDI export skipped (mido may not be available):", e)

# For live loop playback (if pygame available):
# import pygame
# pygame.mixer.init(frequency=SR)
# sound = pygame.sndarray.make_sound((mix * 32767).astype(np.int16))
# sound.play(-1)  # loop forever
