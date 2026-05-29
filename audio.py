import numpy as np
import sounddevice as sd
from enum import Enum

class Scale(Enum):
    """音階對應頻率（C大調）"""
    C4 = 261.63
    D4 = 293.66
    E4 = 329.63
    F4 = 349.23
    G4 = 392.00
    A4 = 440.00
    B4 = 493.88
    C5 = 523.25

SCALE_NOTES = list(Scale)

class Synthesizer:
    """音效合成器"""
    
    SAMPLE_RATE = 44100
    
    def __init__(self):
        self.current_freq: Optional[float] = None
        self.volume: float = 0.5
        self._stream: Optional[sd.OutputStream] = None
        self._phase: float = 0.0
    
    def y_to_note(self, y_normalized: float) -> Scale:
        """將 Y 軸位置映射到音符（上方 = 高音）"""
        idx = int((1 - y_normalized) * (len(SCALE_NOTES) - 1))
        idx = max(0, min(idx, len(SCALE_NOTES) - 1))
        return SCALE_NOTES[idx]
    
    def pinch_to_volume(self, pinch_dist: float) -> float:
        """捏合距離映射到音量（捏越緊越小聲）"""
        # pinch_dist 約在 0.02 ~ 0.3 之間
        vol = np.clip((pinch_dist - 0.02) / 0.28, 0.0, 1.0)
        return float(vol)
    
    def play_tone(self, frequency: float, volume: float, duration: float = 0.05):
        """播放單一音符"""
        t = np.linspace(0, duration, int(self.SAMPLE_RATE * duration), False)
        wave = volume * np.sin(2 * np.pi * frequency * t)
        # 加入泛音讓聲音更豐富
        wave += (volume * 0.3) * np.sin(4 * np.pi * frequency * t)
        wave *= np.hanning(len(wave))  # 淡入淡出
        sd.play(wave.astype(np.float32), self.SAMPLE_RATE, blocking=False)
