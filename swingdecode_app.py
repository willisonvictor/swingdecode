# SwingDecode - Streamlit MVP (ASCII safe)
# ---------------------------------------
# Sanitized, ASCII-only version. All notes are Python comments; no smart quotes or long dashes.
#
# Features:
# - Upload a short golf swing video (3-6s).
# - Extract frames with OpenCV.
# - Run MediaPipe Pose for body landmarks.
# - Detect rough key frames: address, top, impact, finish.
# - Compute metrics: tempo ratio, hip sway (FO), head movement (lateral, vertical).
# - Draw overlays and return an annotated video.
# - Optional LLM coach summary (uses OpenAI if OPENAI_API_KEY is set).

import os
import tempfile
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import cv2
import mediapipe as mp
import streamlit as st
from scipy.signal import savgol_filter


# ----------------------------- Data models -----------------------------
@dataclass
class KeyFrames:
    address: int
    top: int
    impact: int
    finish: int


@dataclass
class Metrics:
    tempo_ratio: float
    hip_sway_cm: float
    head_lateral_cm: float
    head_vertical_cm: float
    angle_hint: str  # FO, DTL, or unknown


# ----------------------------- Pose helper -----------------------------
_mp_pose = None  # created lazily so Streamlit reloads cleanly

def get_pose():
    global _mp_pose
    if _mp_pose is None:
        _mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _mp_pose


# ----------------------------- Video IO -----------------------------

def read_video_bytes_to_frames(file_bytes: bytes) -> Tuple[List[np.ndarray], float]:
    """Return list of frames (BGR) and fps."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        os.unlink(tmp_path)
        raise RuntimeError("Could not open video. Try a short MP4/MOV from a steady camera.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    os.unlink(tmp_path)
    if len(frames) == 0:
        raise RuntimeError("No frames decoded. Please re-encode your clip and try again.")
    return frames, float(fps)


# ----------------------------- Pose sequence -----------------------------
BODY = {
    'nose': 0,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_hip': 23,
    'right_hip': 24,
}

def pose_sequence(frames: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    pose = get_pose()
    H, W = frames[0].shape[:2]
    lms = []
    vis = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if not res.pose_landmarks:
            lms.append(np.full((33, 3), np.nan, dtype=np.float32))
            vis.append(np.zeros((33,), dtype=np.float32))
            continue
        pts = np.array([[lm.x * W, lm.y * H, lm.z] for lm in res.pose_landmarks.landmark], dtype=np.float32)
        v = np.array([lm.visibility for lm in res.pose_landmarks.landmark], dtype=np.float32)
        lms.append(pts)
        vis.append(v)
    return np.stack(lms), np.stack(vis)


# ----------------------------- Angle classifier -----------------------------

def classify_angle(lms: np.ndarray, vis: np.ndarray) -> str:
    ls, rs = BODY['left_shoulder'], BODY['right_shoulder']
    lh, rh = BODY['left_hip'], BODY['right_hip']
    shoulders = lms[:, [ls, rs], :2]
    hips = lms[:, [lh, rh], :2]
    d_sh = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1)
    d_hp = np.linalg.norm(hips[:, 0] - hips[:, 1], axis=1)
    ratio = np.nanmedian(d_sh / (d_hp + 1e-6))
    return 'FO' if ratio > 0.9 else 'DTL'


# ----------------------------- Key frames -----------------------------

def _smooth(x: np.ndarray, win: int = 11) -> np.ndarray:
    if len(x) < 7:
        return x
    win = min(len(x) - (1 - len(x) % 2), win)
    if win < 5:
        return x
    return savgol_filter(x, window_length=win if win % 2 == 1 else win - 1, polyorder=2)


def detect_key_frames(lms: np.ndarray, vis: np.ndarray) -> KeyFrames:
    lw, rw = BODY['left_wrist'], BODY['right_wrist']
    lh, rh = BODY['left_hip'], BODY['right_hip']

    hands = np.nanmean(lms[:, [lw, rw], :2], axis=1)
    pelvis = np.nanmean(lms[:, [lh, rh], :2], axis=1)

    y_hand = hands[:, 1]
    y_hand_s = _smooth(y_hand, 31)
    top_idx = int(np.nanargmin(y_hand_s))

    vel = np.linalg.norm(np.diff(hands, axis=0, prepend=hands[[0]]), axis=1)
    win = max(5, len(vel) // 30)
    mov_avg = np.convolve(vel, np.ones(win) / max(1, win), mode='same')
    start_window = slice(0, max(5, len(vel) // 5))
    address_idx = int(np.nanargmin(mov_avg[start_window]))

    d = np.linalg.norm(hands - pelvis, axis=1)
    search = slice(top_idx + 1, len(d))
    impact_idx = int(np.nanargmin(d[search])) + (top_idx + 1)

    end_window = slice(int(len(vel) * 2 / 3), len(vel))
    finish_idx = int(np.nanargmin(mov_avg[end_window])) + int(len(vel) * 2 / 3)

    address_idx = max(0, min(address_idx, len(lms) - 1))
    top_idx = max(address_idx + 1, min(top_idx, len(lms) - 1))
    impact_idx = max(top_idx + 1, min(impact_idx, len(lms) - 1))
    finish_idx = max(impact_idx + 1, min(finish_idx, len(lms) - 1))

    return KeyFrames(address_idx, top_idx, impact_idx, finish_idx)


# ----------------------------- Metrics -----------------------------

def _px_to_cm(px: float, reference_px: float) -> float:
    shoulder_cm = 40.0
    return (px / (reference_px + 1e-6)) * shoulder_cm


def compute_metrics(lms: np.ndarray, vis: np.ndarray, kf: KeyFrames) -> Metrics:
    lw, rw = BODY['left_wrist'], BODY['right_wrist']
    ls, rs = BODY['left_shoulder'], BODY['right_shoulder']
    lh, rh = BODY['left_hip'], BODY['right_hip']
    nose = BODY['nose']

    hands = np.nanmean(lms[:, [lw, rw], :2], axis=1)
    pelvis = np.nanmean(lms[:, [lh, rh], :2], axis=1)
    head = lms[:, nose, :2]
    shoulders = lms[:, [ls, rs], :2]
    shoulder_width_px = np.nanmedian(np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1))

    backswing = max(1, kf.top - kf.address)
    downswing = max(1, kf.impact - kf.top)
    tempo_ratio = round(float(backswing) / float(downswing), 2)

    hip_sway_px = float(pelvis[kf.impact, 0] - pelvis[kf.address, 0])
    hip_sway_cm = round(_px_to_cm(abs(hip_sway_px), shoulder_width_px), 1)

    head_lat_px = float(head[kf.impact, 0] - head[kf.address, 0])
    head_vert_px = float(head[kf.impact, 1] - head[kf.address, 1])
    head_lateral_cm = round(_px_to_cm(abs(head_lat_px), shoulder_width_px), 1)
    head_vertical_cm = round(_px_to_cm(abs(head_vert_px), shoulder_width_px), 1)

    angle_hint = classify_angle(lms, vis)

    return Metrics(
        tempo_ratio=tempo_ratio,
        hip_sway_cm=hip_sway_cm,
        head_lateral_cm=head_lateral_cm,
        head_vertical_cm=head_vertical_cm,
        angle_hint=angle_hint,
    )


# ----------------------------- Overlay and compose -----------------------------

def draw_overlay(frame: np.ndarray, lm_row: np.ndarray, metrics: Metrics, kf: KeyFrames, idx: int) -> np.ndarray:
    out = frame.copy()
    H, W = out.shape[:2]
    color = (0, 255, 0)
    for name in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'left_wrist', 'right_wrist', 'nose']:
        i = BODY[name]
        if np.isnan(lm_row[i, 0]):
            continue
        cv2.circle(out, (int(lm_row[i, 0]), int(lm_row[i, 1])), 4, color, -1)

    # Metrics label box
    cv2.rectangle(out, (10, 10), (420, 95), (0, 0, 0), -1)
    cv2.putText(out, f"Tempo: {metrics.tempo_ratio}:1", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(out, f"Hip sway: {metrics.hip_sway_cm} cm", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(out, f"Head L/V: {metrics.head_lateral_cm}/{metrics.head_vertical_cm} cm", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if idx in [kf.address, kf.top, kf.impact, kf.finish]:
        tag = 'ADDRESS' if idx == kf.address else 'TOP' if idx == kf.top else 'IMPACT' if idx == kf.impact else 'FINISH'
        cv2.putText(out, tag, (W - 160, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 0), 3)
    return out


def annotate_video(frames: List[np.ndarray], lms: np.ndarray, metrics: Metrics, kf: KeyFrames, fps: float) -> bytes:
    """Write video with OpenCV. Try MP4 first, fall back to AVI if needed."""
    H, W = frames[0].shape[:2]

    def _write(ext: str, fourcc: str) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            path = tmp.name
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, (W, H))
        if not writer.isOpened():
            try:
                os.unlink(path)
            except Exception:
                pass
            raise RuntimeError("VideoWriter could not open")
        for i, f in enumerate(frames):
            frame = draw_overlay(f, lms[i], metrics, kf, i)
            writer.write(frame)
        writer.release()
        with open(path, 'rb') as f:
            data = f.read()
        os.unlink(path)
        return data

    try:
        return _write('.mp4', 'mp4v')
    except Exception:
        return _write('.avi', 'MJPG')


# ----------------------------- LLM summary -----------------------------
try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    _OPENAI_OK = False

COACH_PROMPT = """You are a golf swing coach. Given swing metrics, write a concise, encouraging breakdown (max 120 words) that includes:
- 1 sentence on what's good
- 2 specific improvement points tied to the metrics
- 1 drill suggestion name with a 1-line instruction
Avoid technical jargon. Be positive and practical."""

)

def coach_summary(metrics: Metrics) -> str:
    if not _OPENAI_OK or not os.getenv("OPENAI_API_KEY"):
        return (
            f"Tempo {metrics.tempo_ratio}:1. Hip sway {metrics.hip_sway_cm} cm. "
            f"Head move L/V {metrics.head_lateral_cm}/{metrics.head_vertical_cm} cm. "
            "Suggested drill: Chair Hips - place a chair behind your hips and keep light contact through the downswing."
        )
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
        messages=[
            {"role": "system", "content": COACH_PROMPT},
            {"role": "user", "content": msg},
        ],
        temperature=0.4,
    )
    return chat.choices[0].message.content.strip()


# ----------------------------- Streamlit UI -----------------------------
st.set_page_config(page_title="SwingDecode MVP", layout="wide")
st.title("SwingDecode - AI Golf Swing Analyzer (MVP)")
st.caption("Upload a short swing clip (3-6 seconds). We analyze tempo, hip sway, and head stability, then return an annotated video and a quick coaching note.")

with st.sidebar:
    st.header("Settings")
    fps_override = st.slider("Output FPS", 24, 60, 30)
    st.write("LLM summary:", "ON" if os.getenv("OPENAI_API_KEY") else "OFF (set OPENAI_API_KEY to enable)")

uploaded = st.file_uploader("Upload MP4/MOV swing clip", type=["mp4", "mov", "m4v"], accept_multiple_files=False)

if uploaded is not None:
    try:
        if uploaded.size and uploaded.size > 80 * 1024 * 1024:
            st.error("File too large. Please upload a clip under 80 MB.")
        else:
            st.info("Processing... this may take a moment for longer clips.")
            data = uploaded.read()
            frames, fps_in = read_video_bytes_to_frames(data)
            fps = fps_override or fps_in

            lms, vis = pose_sequence(frames)
            if np.all(np.isnan(lms)):
                st.error("Could not detect a person in the video. Try better lighting and a steady camera.")
            else:
                kf = detect_key_frames(lms, vis)
                metrics = compute_metrics(lms, vis, kf)
                video_bytes = annotate_video(frames, lms, metrics, kf, fps)
                summary = coach_summary(metrics)

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.subheader("Annotated Swing")
                    st.video(video_bytes)
                with col2:
                    st.subheader("Metrics")
                    st.metric("Tempo (backswing:downswing)", f"{metrics.tempo_ratio}:1")
                    st.metric("Hip sway (address to impact)", f"{metrics.hip_sway_cm} cm")
                    st.metric("Head move L/V (addr to impact)", f"{metrics.head_lateral_cm}/{metrics.head_vertical_cm} cm")
                    st.caption(f"Angle hint: {metrics.angle_hint}")
                    st.download_button("Download annotated video", data=video_bytes, file_name="swingdecode_annotated.mp4", mime="video/mp4")

                st.subheader("Coach Note")
                st.write(summary)

                with st.expander("Debug - keyframes"):
                    st.write(kf)
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload a short clip to begin.")

st.markdown("---")
st.caption("MVP prototype. Not a medical device or substitute for a qualified coach. For best results, record from face-on or down-the-line with a stable camera.")
