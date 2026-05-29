import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class HandLandmarks:
    """手部關鍵點資料結構"""
    landmarks: list
    handedness: str  # 'Left' or 'Right'
    
    @property
    def index_finger_tip(self):
        return self.landmarks[8]
    
    @property
    def thumb_tip(self):
        return self.landmarks[4]
    
    def pinch_distance(self) -> float:
        """計算拇指與食指的距離（捏合程度）"""
        t = self.thumb_tip
        i = self.index_finger_tip
        return np.sqrt((t.x - i.x)**2 + (t.y - i.y)**2)
    
    def index_y_normalized(self) -> float:
        """食指 Y 軸位置（0=頂 1=底）"""
        return self.index_finger_tip.y


class HandDetector:
    """手部偵測器（封裝 MediaPipe）"""
    
    def __init__(self, max_hands: int = 2, confidence: float = 0.7):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=confidence,
            min_tracking_confidence=confidence
        )
    
    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, list[HandLandmarks]]:
        """偵測畫面中的手部，回傳標註後的畫面與手部資料"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        detected = []
        
        if results.multi_hand_landmarks:
            for hand_lm, hand_info in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                self.mp_draw.draw_landmarks(
                    frame, hand_lm, self.mp_hands.HAND_CONNECTIONS
                )
                detected.append(HandLandmarks(
                    landmarks=hand_lm.landmark,
                    handedness=hand_info.classification[0].label
                ))
        
        return frame, detected
    
    def release(self):
        self.hands.close()
