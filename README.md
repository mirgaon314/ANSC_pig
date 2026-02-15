# ANSC_pig
# ANSC_pig

Pipeline and analysis notes for Novel Object Recognition (NOR) pig behavior videos.

## What this repo is

This project supports a two-stage computer vision pipeline for NOR video analysis:

- **Stage 1 (Detection)**: detect pigs and objects, then detect pig head within the pig box (YOLO-based). The model outputs bounding boxes and coordinates for pigs, objects, and pig heads. 
- **Stage 2 (Interaction classification)**: classify whether a pig is interacting with an object using outputs from Stage 1.

(See the “NOR Project Update” slides for the current model plan and early results.)

## Current modeling plan

### Stage 1: Detection

- Detect pigs and objects, returning bounding box coordinates.
- Detect the pig head inside the pig box, returning head coordinates.
- Training data was annotated with Label Studio (early set sizes were ~150+ for pig and object, ~200+ for head).

### Stage 2: Interaction classification (approaches tested)

1) **Coordinates only**
- Use pig, head, and object coordinates to compute spatial features such as distance.
- Early result: strong on “not interacting”, weak on “interacting” (No: 75.09%, Yes: 6.01%).

2) **Cropped image (5-frame sequence)**
- Use a short frame sequence so the model can learn motion cues.
- Early result (trained on one trial): ~31% accuracy.

3) **Annotated cropped image (next step)**
- Same as (2), but train on annotated crops.
- Needs improved head detection and more training data (slide notes mention a push toward ~1800 head annotations).

## NOR experiment workflow

This section is the lab-facing SOP for running NOR and producing analysis-ready files.

### 1) Object sets and counterbalancing

- Use object sets that are validated to have no innate preference.
- Use two different sets when possible. If only one validated set exists, counterbalance which object is novel.
- Counterbalance novel-object location (left vs right), considering both animal entry direction and camera view.

### 2) Arena setup tips (Vet Med)

- Flooring is assembled as 4×4 panels, while the pig is exposed to a 3×3 area.
- Tools: mallet, black screws (floor panels), zip ties (secure walls).
- Toys: keep taping consistent across sites (Vet Med and PNCL).
- Noise machines: place at opposite sides of the arena; they use 4× AAA batteries.

### 3) Cameras and recording (Vet Med)

- Verify the camera view matches the intended “In/Out” direction.
- Start recording in Axis Camera Station Client.
- It is easiest to download and copy videos to the lab share right after a trial or cohort.

### 4) During trials

- Double check recording is active.
- Ensure doors fully close; pigs should not be able to push doors open.
- Write the pig ID (or cage number) on the pig’s back to help later video analysis.
- Clean arena and objects between trials.

Cleaning
- Arena: spray, then quick mop with bleach water.
- Objects: dunk in bleach + soapy water, then rinse with regular water.
- Remove feces and urine thoroughly so pigs do not spend the trial sniffing residue.

## Manual annotation (Adobe Premiere Pro)

### Convert and import

- Convert `.wma` to `.mp4` with HandBrake.
- Create a Premiere project, import videos, and drag onto the timeline.

### Navigation and markers

- Use the timeline playhead or preview controls to move through video.
- If playback is jumpy, lower preview resolution.

Marker shortcuts and edits
- Add marker: select timeline panel, then press `M` (or right click → Add Marker).
- Resize marker: `Alt` + click marker to enable wings, then drag.
- Edit marker: double click marker.

### Event definitions

Define “exploration” with a clear rule. Example: mouth-based interactions (rooting, sniffing, mouthing) count, but standing nearby, hovering, or rubbing does not.

### Trial start and end markers

- Trial start: when the door opens and the pig enters the arena.
- Trial end: when the door opens and an experimenter enters the arena.

Label format (case-sensitive, required for the R script)
- `Trial,X,start`
- `Trial,X,end`

### Object exploration markers

- Create a marker at the first frame exploration begins.
- Extend the marker to cover the full event.
- If the pig breaks attention and returns later, record separate events.
- Label exploration events as `left` or `right` in Premiere, then map left/right to sample/novel later.

### Export markers

- File → Export → Markers
- Export as `.csv` into a folder that contains only marker export files (example folder name: `NOR Input`).

## R processing and index file

### Before running the R script

- Add a **frame rate** column to each exported marker `.csv`.
- Verify frame rate in the video (right click preview → Properties). Target is 30 fps.
- Re-save as `.csv` (R may fail to read Premiere exports unless re-saved).

### R script purpose

The R script cleans and summarizes marker events so results can be used for statistical testing (SAS). Outputs include left, right, and total measures per trial. Only **test trials** are used for final statistical analysis.

Common output suffixes
- `t`: total exploration
- `sam`: sample object exploration
- `test`: novel object exploration
- `_n`: number of visits
- `_cd`: cumulative duration (seconds)
- `_me`, `_md`: mean, median duration
- `_sd`, `_se`: standard deviation, standard error
- `_lf`, `_ll`: latency to first/last visit

### Manual edit and Recognition Index (RI)

- Build an index sheet with columns you need for SAS: trial, diet, pig ID, litter, rep, phase, object set, novel-object location, etc.
- Map `left/right` exploration to `sam/test` using the novel-object location column.
- Compute Recognition Index:

`RI = test_cd / t_cd`

NA handling
- Missing counts and durations are converted to 0.
- Mean, median, SD, SE, and latencies remain NA.
- In SAS, leave NA values blank so SAS reads them as missing.

## SAS and Prism

- SAS is used for one-way ANOVA and power analysis.
- Prism is used for plotting. Use the lab GraphPad computer license (room 2) and save often.

## Notes and next steps

- Improve head detection consistency.
- Expand annotated dataset for head and interaction events.
- Re-test Stage 2 with annotated crops and more trials.

## References

- NOR Project Update slides (Google Slides).
- Internal lab SOP: “Conducting NOR and analysis”.