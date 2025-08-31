# SwingDecode - Streamlit MVP
# Clean ASCII header to avoid Unicode parsing issues.
# 
# Features:
# - Extracts frames with OpenCV.
# - Runs MediaPipe Pose for body landmarks.
# - Finds rough key frames: address, top, impact, finish.
# - Computes 3 metrics: tempo ratio, hip sway (FO), head movement (vertical & lateral).
# - Draws overlays and returns an annotated video.
# - Summarizes with an LLM (OpenAI) into a short coaching note.
#
# Quick start:
# 1) Create a venv (optional) and install deps:
#    pip install streamlit mediapipe opencv-python-headless numpy scipy openai pillow
# 2) Set your OpenAI key:
#    export OPENAI_API_KEY=YOUR_KEY_HERE

import os
import io
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2
import mediapipe as mp
import streamlit as st
from scipy.signal import savgol_filter

# 3) Run app
streamlit run swingdecode_app.py

Notes
-----
# This is an MVP focused on clarity over completeness.
# Works best with 3–6s clips and a stable camera.
# Uses CPU-only models; processing may take ~a short while for longer clips.
"""

import os
import io
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2
import mediapipe as mp
import streamlit as st
from scipy.signal import savgol_filter

# --- Optional: LLM (set OPENAI_API_KEY or skip) ---
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

# -------------- Utilities --------------
@dataclass
class KeyFrames:
    address: int
    top: int
    impact: int
    finish: int

@dataclass
class Metrics:
    tempo_ratio: float  # backswing : downswing
    hip_sway_cm: float
    head_lateral_cm: float
    head_vertical_cm: float
    angle_hint: str  # 'FO' or 'DTL' or 'unknown'

mp_pose = mp.solutions.pose
POSE_LM = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def read_video_bytes_to_frames(file_bytes: bytes) -> Tuple[List[np.ndarray], float]:
    """Return frames (BGR) and fps."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    os.unlink(tmp_path)
    return frames, fps

# ---------------- Pose & landmarks ----------------
BODY_IDXS = {
    'nose': 0,
    'left_eye_inner': 1,
    'left_eye': 2,
    'left_eye_outer': 3,
    'right_eye_inner': 4,
    'right_eye': 5,
    'right_eye_outer': 6,
    'left_ear': 7,
    'right_ear': 8,
    'mouth_left': 9,
    'mouth_right': 10,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_pinky': 17,
    'right_pinky': 18,
    'left_index': 19,
    'right_index': 20,
    'left_thumb': 21,
    'right_thumb': 22,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
}

def pose_sequence(frames: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays: (N, 33, 3) landmarks and visibility (N, 33)."""
    H, W = frames[0].shape[:2]
    lms = []
    vis = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        res = POSE_LM.process(rgb)
        if not res.pose_landmarks:
            lms.append(np.full((33, 3), np.nan, dtype=np.float32))
            vis.append(np.zeros((33,), dtype=np.float32))
            continue
        pts = np.array([[lm.x * W, lm.y * H, lm.z] for lm in res.pose_landmarks.landmark], dtype=np.float32)
        v = np.array([lm.visibility for lm in res.pose_landmarks.landmark], dtype=np.float32)
        lms.append(pts)
        vis.append(v)
    return np.stack(lms), np.stack(vis)

# ---------------- Angle classifier (simple heuristic) ----------------

def classify_angle(lms: np.ndarray, vis: np.ndarray) -> str:
    """Rough face-on vs down-the-line based on shoulder width & pelvis orientation."""
    ls, rs = BODY_IDXS['left_shoulder'], BODY_IDXS['right_shoulder']
    lh, rh = BODY_IDXS['left_hip'], BODY_IDXS['right_hip']
    shoulders = lms[:, [ls, rs], :2]  # (N,2,2)
    hips = lms[:, [lh, rh], :2]
    d_sh = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1)
    d_hp = np.linalg.norm(hips[:, 0] - hips[:, 1], axis=1)
    ratio = np.nanmedian(d_sh / (d_hp + 1e-6))
    # Very rough: FO tends to show more lateral spread (ratio ~ similar), DTL often compresses one axis.
    return 'FO' if ratio > 0.9 else 'DTL'

# ---------------- Key frame detection ----------------

def smooth(x: np.ndarray, win: int = 11) -> np.ndarray:
    win = min(len(x) - (1 - len(x) % 2), win)  # ensure odd & <= len
    if win < 5:
        return x
    return savgol_filter(x, window_length=win if win % 2 == 1 else win - 1, polyorder=2)


def detect_key_frames(lms: np.ndarray, vis: np.ndarray) -> KeyFrames:
    """Heuristics: 
    - address: lowest hand speed window at start
    - top: max hand height
    - impact: min distance from lead hand to pelvis center during downswing
    - finish: last frame with stable pose
    """
    lw, rw = BODY_IDXS['left_wrist'], BODY_IDXS['right_wrist']
    lh, rh = BODY_IDXS['left_hip'], BODY_IDXS['right_hip']

    hands = np.nanmean(lms[:, [lw, rw], :2], axis=1)  # (N,2) average hands
    pelvis = np.nanmean(lms[:, [lh, rh], :2], axis=1)

    y_hand = hands[:, 1]
    y_hand_s = smooth(y_hand, 31)

    # top of backswing ~ highest hands (min y in image coords increasing downward)
    top_idx = int(np.nanargmin(y_hand_s))

    # address ~ start of motion: find first 20% segment where hand speed is minimal
    vel = np.linalg.norm(np.diff(hands, axis=0, prepend=hands[[0]]), axis=1)
    win = max(5, len(vel) // 30)
    mov_avg = np.convolve(vel, np.ones(win) / win, mode='same')
    start_window = slice(0, max(5, len(vel) // 5))
    address_idx = int(np.nanargmin(mov_avg[start_window]))

    # impact ~ closest hands to pelvis after top
    d = np.linalg.norm(hands - pelvis, axis=1)
    search = slice(top_idx + 1, len(d))
    impact_idx = int(np.nanargmin(d[search])) + (top_idx + 1)

    # finish ~ last third where velocity drops
    end_window = slice(int(len(vel) * 2 / 3), len(vel))
    finish_idx = int(np.nanargmin(mov_avg[end_window])) + int(len(vel) * 2 / 3)

    # safety clamps
    address_idx = max(0, min(address_idx, len(lms) - 1))
    top_idx = max(address_idx + 1, min(top_idx, len(lms) - 1))
    impact_idx = max(top_idx + 1, min(impact_idx, len(lms) - 1))
    finish_idx = max(impact_idx + 1, min(finish_idx, len(lms) - 1))

    return KeyFrames(address_idx, top_idx, impact_idx, finish_idx)

# ---------------- Metrics ----------------

def px_to_cm(px: float, reference_px: float) -> float:
    # Crude scale using shoulder width as ~40 cm proxy (adult). Adjust per height later.
    SHOULDER_CM = 40.0
    return (px / (reference_px + 1e-6)) * SHOULDER_CM


def compute_metrics(lms: np.ndarray, vis: np.ndarray, kf: KeyFrames) -> Metrics:
    lw, rw = BODY_IDXS['left_wrist'], BODY_IDXS['right_wrist']
    ls, rs = BODY_IDXS['left_shoulder'], BODY_IDXS['right_shoulder']
    lh, rh = BODY_IDXS['left_hip'], BODY_IDXS['right_hip']
    nose = BODY_IDXS['nose']

    hands = np.nanmean(lms[:, [lw, rw], :2], axis=1)
    pelvis = np.nanmean(lms[:, [lh, rh], :2], axis=1)
    head = lms[:, nose, :2]
    shoulders = lms[:, [ls, rs], :2]
    shoulder_width_px = np.nanmedian(np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1))

    # Tempo ratio
    backswing = max(1, kf.top - kf.address)
    downswing = max(1, kf.impact - kf.top)
    tempo_ratio = round(float(backswing) / float(downswing), 2)

    # Hip sway (FO): lateral pelvis x-shift address->impact
    hip_sway_px = float(pelvis[kf.impact, 0] - pelvis[kf.address, 0])
    hip_sway_cm = round(px_to_cm(abs(hip_sway_px), shoulder_width_px), 1)

    # Head movement address->impact
    head_lat_px = float(head[kf.impact, 0] - head[kf.address, 0])
    head_vert_px = float(head[kf.impact, 1] - head[kf.address, 1])
    head_lateral_cm = round(px_to_cm(abs(head_lat_px), shoulder_width_px), 1)
    head_vertical_cm = round(px_to_cm(abs(head_vert_px), shoulder_width_px), 1)

    angle_hint = classify_angle(lms, vis)

    return Metrics(
        tempo_ratio=tempo_ratio,
        hip_sway_cm=hip_sway_cm,
        head_lateral_cm=head_lateral_cm,
        head_vertical_cm=head_vertical_cm,
        angle_hint=angle_hint,
    )

# ---------------- Overlays ----------------

def draw_overlay(frame: np.ndarray, lm_row: np.ndarray, metrics: Metrics, kf: KeyFrames, idx: int) -> np.ndarray:
    out = frame.copy()
    H, W = out.shape[:2]
    color = (0, 255, 0)
    # Head + pelvis traces are optional for MVP; here just draw current landmarks for context
    # Draw key joints
    for name in ['left_shoulder','right_shoulder','left_hip','right_hip','left_wrist','right_wrist','nose']:
        i = BODY_IDXS[name]
        if np.isnan(lm_row[i, 0]):
            continue
        cv2.circle(out, (int(lm_row[i, 0]), int(lm_row[i, 1])), 4, color, -1)

    # Label
    cv2.rectangle(out, (10, 10), (360, 90), (0,0,0), -1)
    cv2.putText(out, f"Tempo: {metrics.tempo_ratio}:1", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(out, f"Hip sway: {metrics.hip_sway_cm} cm", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(out, f"Head move L/V: {metrics.head_lateral_cm}/{metrics.head_vertical_cm} cm", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # Keyframe markers
    if idx in [kf.address, kf.top, kf.impact, kf.finish]:
        tag = 'ADDRESS' if idx == kf.address else 'TOP' if idx == kf.top else 'IMPACT' if idx == kf.impact else 'FINISH'
        cv2.putText(out, tag, (W-160, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,100,0), 3)
    return out

# ---------------- Video compose ----------------

def annotate_video(frames: List[np.ndarray], lms: np.ndarray, metrics: Metrics, kf: KeyFrames, fps: float) -> bytes:
    """Compose annotated video using OpenCV VideoWriter (avoids moviepy).
    Returns mp4 bytes.
    """
    H, W = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # widely supported
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    vw = cv2.VideoWriter(tmp_path, fourcc, fps, (W, H))
    for i, f in enumerate(frames):
        frame = draw_overlay(f, lms[i], metrics, kf, i)
        vw.write(frame)  # already BGR
    vw.release()
    with open(tmp_path, "rb") as f:
        data = f.read()
    os.unlink(tmp_path)
    return data

# ---------------- LLM summary ----------------
COACH_PROMPT = """
You are a golf swing coach. Given swing metrics, write a concise, encouraging breakdown (max 120 words) with: 
- 1 sentence on what’s good
- 2 specific improvement points tied to the metrics
- 1 drill suggestion name with 1-line instruction
Avoid technical jargon. Be positive and practical.
"""

def coach_summary(metrics: Metrics) -> str:
    if not _OPENAI_AVAILABLE or not os.getenv("OPENAI_API_KEY"):
        return (
            "(LLM summary disabled) Tempo {t}:1. Hip sway {hs}cm. Head move L/V {hl}/{hv}cm. "
            "Suggested drill: Chair Hips — place a chair behind your hips and keep light contact through the downswing."
        ).format(t=metrics.tempo_ratio, hs=metrics.hip_sway_cm, hl=metrics.head_lateral_cm, hv=metrics.head_vertical_cm)
    client = OpenAI()
    content = {
        "tempo_ratio": metrics.tempo_ratio,
        "hip_sway_cm": metrics.hip_sway_cm,
        "head_lateral_cm": metrics.head_lateral_cm,
        "head_vertical_cm": metrics.head_vertical_cm,
        "angle_hint": metrics.angle_hint,
    }
    msg = f"Metrics: {content}"
    chat = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":COACH_PROMPT},{"role":"user","content":msg}],
        temperature=0.4,
    )
    return chat.choices[0].message.content.strip()

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="SwingDecode MVP", page_icon="🏌️", layout="wide")
st.title("🏌️ SwingDecode — AI Golf Swing Analyzer (MVP)")
st.caption("Upload a short swing clip (3–6s). We’ll analyze tempo, hip sway, and head stability, then return an annotated video and a quick coaching note.")

with st.sidebar:
    st.header("Settings")
    fps_override = st.slider("Output FPS", 24, 60, 30)
    st.write("LLM summary:", "ON" if os.getenv("OPENAI_API_KEY") else "OFF (set OPENAI_API_KEY to enable)")

uploaded = st.file_uploader("Upload MP4/MOV swing clip", type=["mp4","mov","m4v"], accept_multiple_files=False)

if uploaded is not None:
    st.info("Processing… this may take a moment for longer clips.")
    data = uploaded.read()
    frames, fps_in = read_video_bytes_to_frames(data)
    fps = fps_override or fps_in

    # Pose
    lms, vis = pose_sequence(frames)

    # Keyframes
    kf = detect_key_frames(lms, vis)

    # Metrics
    metrics = compute_metrics(lms, vis, kf)

    # Annotate
    video_bytes = annotate_video(frames, lms, metrics, kf, fps)

    # Summary
    summary = coach_summary(metrics)

    # --- UI output ---
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Annotated Swing")
        st.video(video_bytes)
    with col2:
        st.subheader("Metrics")
        st.metric("Tempo (backswing:downswing)", f"{metrics.tempo_ratio}:1")
        st.metric("Hip sway (address→impact)", f"{metrics.hip_sway_cm} cm")
        st.metric("Head move L/V (addr→impact)", f"{metrics.head_lateral_cm}/{metrics.head_vertical_cm} cm")
        st.caption(f"Angle hint: {metrics.angle_hint}")
        st.download_button("Download annotated video", data=video_bytes, file_name="swingdecode_annotated.mp4", mime="video/mp4")

    st.subheader("Coach Note")
    st.write(summary)

    with st.expander("Debug — keyframes"):
        st.write(kf)

else:
    st.info("Upload a short clip to begin.")

st.markdown("---")
st.caption("MVP prototype. Not a medical device or substitute for a qualified coach. For best results, record from face-on or down-the-line with a stable camera.")
