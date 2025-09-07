import os
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple

# --- Configuration (tweak as needed) ---
# Bridge small gaps between consecutive segments of the SAME side (seconds)
GAP_SEC = 0.25
# Drop segments shorter than this duration after merging (seconds)
MIN_DUR_SEC = 0.10
# Force trial start and end markers to fixed bounds instead of using first/last interaction
FORCE_START_AT_ZERO = True
FORCE_END_AT_VIDEO_DURATION = True
# If FORCE_END_AT_VIDEO_DURATION is True, set your video duration (in seconds)
VIDEO_DURATION_SEC = 300.0  # 5 minutes
# ---------------------------------------

def detect_interaction_columns(df: pd.DataFrame) -> Dict[str, pd.Series]:
    tokens = {"interacting_right", "interacting_left"}
    string_cols = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
    candidates = []
    for c in string_cols:
        ser = df[c].astype(str).str.strip().str.lower()
        if ser.isin(tokens).any() or ser.str.contains("interacting_right|interacting_left", regex=True).any():
            candidates.append(c)

    preferred_order = ["interaction", "interact", "cls", "class", "label", "pred", "prediction", "state", "activity"]
    chosen = None
    for pref in preferred_order:
        for c in candidates:
            if pref in c.lower():
                chosen = c
                break
        if chosen:
            break
    if chosen is None and candidates:
        chosen = candidates[0]

    side = pd.Series([None]*len(df), index=df.index, dtype=object)
    if chosen is not None:
        ser = df[chosen].astype(str).str.strip().str.lower()
        side = np.where(ser.str.contains("interacting_right"), "interacting_right",
                        np.where(ser.str.contains("interacting_left"), "interacting_left", None))
        side = pd.Series(side, index=df.index, dtype=object)

    return {"side": side, "is_interacting": side.notna()}

def pick_time_vector(df: pd.DataFrame) -> Tuple[pd.Series, str, bool]:
    cols = {c.lower(): c for c in df.columns}
    def has(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    # seconds-like
    c = has("time")
    if c is not None:
        return pd.to_numeric(df[c], errors="coerce").fillna(method="ffill").fillna(0.0), c, True
    c = has("timestamp")
    if c is not None:
        return pd.to_numeric(df[c], errors="coerce").fillna(method="ffill").fillna(0.0), c, True
    c = has("sec", "seconds")
    if c is not None:
        return pd.to_numeric(df[c], errors="coerce").fillna(method="ffill").fillna(0.0), c, True
    c = has("ms", "millis", "milliseconds")
    if c is not None:
        s = pd.to_numeric(df[c], errors="coerce").fillna(method="ffill").fillna(0.0) / 1000.0
        return s, c, True

    # frames
    c = has("frame", "frame_index", "idx")
    if c is not None:
        return pd.to_numeric(df[c], errors="coerce").fillna(method="ffill").fillna(0).astype(int), c, False

    idx = pd.Series(np.arange(len(df)), index=df.index)
    return idx, "row_index", False

def contiguous_segments(mask: pd.Series) -> list[tuple[int,int]]:
    arr = mask.values.astype(int)
    change = np.diff(np.concatenate(([0], arr, [0])))
    starts = np.where(change == 1)[0]
    ends = np.where(change == -1)[0] - 1
    return list(zip(starts, ends))

def merge_segments(segs: list[tuple[int,int]], frames: pd.Series, fps: int, gap_sec: float, min_dur_sec: float) -> list[tuple[int,int]]:
    """
    Merge consecutive segments when the gap between them (in frames) is <= gap_sec*fps.
    After merging, drop any segments with duration < min_dur_sec.
    `segs` are in row-index coordinates (start_idx, end_idx). `frames` maps row-index -> frame number.
    """
    if not segs:
        return []

    # Convert to records with explicit frame numbers
    recs = []
    for s_idx, e_idx in sorted(segs, key=lambda t: t[0]):
        s_f = int(frames.iloc[s_idx])
        e_f = int(frames.iloc[e_idx])
        recs.append({"s_idx": s_idx, "e_idx": e_idx, "s_f": s_f, "e_f": e_f})

    merged = []
    cur = recs[0]
    max_gap_frames = int(round(gap_sec * fps))

    for nxt in recs[1:]:
        gap = nxt["s_f"] - cur["e_f"]
        if gap <= max_gap_frames:
            # extend current segment
            cur["e_idx"] = max(cur["e_idx"], nxt["e_idx"])
            cur["e_f"] = max(cur["e_f"], nxt["e_f"])
        else:
            merged.append(cur)
            cur = nxt
    merged.append(cur)

    # Drop tiny segments after merging
    min_frames = int(round(min_dur_sec * fps))
    filtered = [(m["s_idx"], m["e_idx"]) for m in merged if (m["e_f"] - m["s_f"]) >= min_frames]
    return filtered

def timecode_from_frame(frame_idx: int, fps: int) -> str:
    frame_idx = int(round(frame_idx))
    total_seconds = frame_idx // fps
    ff = frame_idx % fps
    hh = total_seconds // 3600
    mm = (total_seconds % 3600) // 60
    ss = total_seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"

def build_marker_table_from_prediction(pred_csv: str, out_csv: str, trial_name: str, fps: int) -> pd.DataFrame:
    df = pd.read_csv(pred_csv)
    det = detect_interaction_columns(df)
    side = det["side"]
    if not side.notna().any():
        raise RuntimeError("No 'interacting_left/right' found in the prediction file.")

    time_vec, time_label, is_seconds = pick_time_vector(df)

    # Choose frame index series for timecode generation
    if time_label in {"frame", "frame_index", "idx", "row_index"} and not is_seconds:
        frames = time_vec.astype(int)
    else:
        # Convert seconds to nearest frame count
        frames = (time_vec.astype(float) * fps).round().astype(int)

    rows = []
    segs_left = contiguous_segments(side == "interacting_left")
    segs_right = contiguous_segments(side == "interacting_right")

    # Merge small gaps within the same side and drop tiny segments
    segs_left = merge_segments(segs_left, frames, fps, GAP_SEC, MIN_DUR_SEC)
    segs_right = merge_segments(segs_right, frames, fps, GAP_SEC, MIN_DUR_SEC)

    # Trial start marker (forced to 00:00 if configured)
    if FORCE_START_AT_ZERO:
        start_tc = timecode_from_frame(0, fps)
        rows.append({
            "Marker Name": f"Trial,{trial_name},start",
            "Description": "",
            "In": start_tc, "Out": start_tc, "Duration": "00:00:00:00",
            "Marker Type": "Comment"
        })
    else:
        earliest_start = min([s for s,_ in segs_left+segs_right]) if (segs_left or segs_right) else None
        if earliest_start is not None:
            tc = timecode_from_frame(frames.iloc[earliest_start], fps)
            rows.append({
                "Marker Name": f"Trial,{trial_name},start",
                "Description": "",
                "In": tc, "Out": tc, "Duration": "00:00:00:00",
                "Marker Type": "Comment"
            })

    for label, segs in [("left", segs_left), ("right", segs_right)]:
        for s, e in segs:
            in_tc = timecode_from_frame(frames.iloc[s], fps)
            out_tc = timecode_from_frame(frames.iloc[e], fps)
            dur_frames = max(0, frames.iloc[e] - frames.iloc[s])  # exclusive-style out
            dur_tc = timecode_from_frame(dur_frames, fps)
            rows.append({
                "Marker Name": label,
                "Description": "",
                "In": in_tc,
                "Out": out_tc,
                "Duration": dur_tc,
                "Marker Type": "Comment"
            })

    # Trial end marker (forced to video duration if configured)
    if FORCE_END_AT_VIDEO_DURATION and (VIDEO_DURATION_SEC is not None):
        end_frame = int(round(VIDEO_DURATION_SEC * fps))
        end_tc = timecode_from_frame(end_frame, fps)
        rows.append({
            "Marker Name": f"Trial,{trial_name},end",
            "Description": "",
            "In": end_tc, "Out": end_tc, "Duration": "00:00:00:00",
            "Marker Type": "Comment"
        })
    else:
        latest_end = max([e for _,e in segs_left+segs_right]) if (segs_left or segs_right) else None
        if latest_end is not None:
            tc = timecode_from_frame(frames.iloc[latest_end], fps)
            rows.append({
                "Marker Name": f"Trial,{trial_name},end",
                "Description": "",
                "In": tc, "Out": tc, "Duration": "00:00:00:00",
                "Marker Type": "Comment"
            })

    # Sort rows (except keep start marker first and end marker last)
    def parse_tc(tc: str, fps: int) -> int:
        hh, mm, ss, ff = map(int, tc.split(":"))
        return ((hh*3600 + mm*60 + ss) * fps) + ff

    # Extract start/end markers separately
    start_rows = [r for r in rows if "Trial" in r["Marker Name"] and "start" in r["Marker Name"]]
    end_rows = [r for r in rows if "Trial" in r["Marker Name"] and "end" in r["Marker Name"]]
    mid_rows = [r for r in rows if r not in start_rows and r not in end_rows]

    mid_rows.sort(key=lambda r: parse_tc(r["In"], fps))
    rows = start_rows + mid_rows + end_rows

    markers = pd.DataFrame(rows, columns=["Marker Name","Description","In","Out","Duration","Marker Type"])
    markers.to_csv(out_csv, index=False)
    return markers

if __name__ == "__main__":
    PRED_PATH = "../ANSC/detected/F003_pred_inhance_head.csv"   # change to your file
    OUT_PATH = "../ANSC/detected/F003_pred_markers_60fps.csv"   # desired output filename
    TRIAL_NAME = "F003"                        # label in start/end markers
    FPS = 60
    markers = build_marker_table_from_prediction(PRED_PATH, OUT_PATH, TRIAL_NAME, FPS)
    print(f"Saved markers to {OUT_PATH}, rows={len(markers)}")
