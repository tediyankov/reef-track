# Annotating fish videos: quick start

You need the **video hard drive**, access to **RFS**, and about **5 GB free** on your laptop.

## One-time setup (about 15 minutes)

1. **Get the tool.** On the reef-track GitHub page click **Code → Download ZIP**, then unzip it
   into your **Documents** folder. Don't put it on a cloud-synced folder (OneDrive, iCloud, Dropbox).
2. **Install it.**
   - **Mac:** macOS blocks double-clicking downloaded scripts ("can't check it for malware";
     click **Done**, not *Move to Trash*). Run it from Terminal instead:
     open **Terminal** (Cmd+Space, type *Terminal*, Enter), type `bash ` (with a space), drag
     `install_mac.command` (unzipped folder → `scripts` → `annotator`) into the Terminal window,
     and press **Enter**. Wait for "Done".
     *(Alternative: after the warning, System Settings → Privacy & Security → **Open Anyway**.)*
   - **Windows:** open the unzipped folder → `scripts` → `annotator`, double-click
     `install_windows.bat`. If Windows warns you, choose **More info → Run anyway**. Wait for "Done".

   You now have an **Annotate fish videos** icon on your Desktop.
3. **Connect the hard drive and RFS** (on Mac: Finder → Go → Connect to Server…).
4. Double-click **Annotate fish videos**. The first time:
   - Type your **name** (it's saved with your annotations).
   - Set the three **library folders** (Browse… for each):
     - *Original videos (hard drive)*: the `originaldata` folder on the hard drive
     - *Compressed videos (RFS)*: `Triggerfish Navigation/fish_in_the_wild/compressed_videos`
     - *Annotations (RFS)*: `Triggerfish Navigation/fish_in_the_wild/annotations`

   On a Mac these are usually filled in already. If the Mac asks whether Python may access
   files on a removable or network volume, click **Allow**.

## Every time you annotate

1. Connect the **hard drive** and **RFS**, then double-click **Annotate fish videos**.
2. In the window that opens, pick the video from the hard drive
   (later: **File → Open video from hard drive…**).
3. The first time anyone opens a video it needs a compressed copy:
   - **Compress now** takes about as long as the video (≈20 min), happens once, and then the
     video opens smoothly for everyone. Leave the laptop plugged in and awake.
   - **Open original** starts straight away but is slower to jump around in.
4. Annotate (full guide: `docs/annotating_behaviour.md`):

   | to… | do this |
   |---|---|
   | play / pause | **Space** |
   | step one frame / one second | **←** **→** / **Shift+←** **Shift+→** |
   | turn a sideways (portrait) video upright | **R** (clockwise) / **Shift+R** (anticlockwise), or **View** menu; remembered for that video |
   | mark a behaviour | **Behaviour** tab: **I** at its first frame, **O** at its last frame, pick the behaviour (or press its number), optional comment, **OK** |
   | fill unlabelled gaps | **G** jumps to the next gap (hatched on the timeline) → **B** → pick behaviour; repeat until "✓ Every frame has a behaviour" |
   | find an annotation | click its coloured chunk on the timeline → its row is highlighted |
   | mark a loop | **Loops** tab: **I** where the loop starts, **O** when the fish is back at that spot, **OK** |
   | fix a mistake | click the row in the table → **Edit…** or **Delete** |

5. Close the window when you're done.

## Saving

Everything saves **automatically** to RFS after every change, into the video's own folder,
e.g. `fish_in_the_wild/annotations/Maldives/2024_02_14/AM/Cait/Left/GX010703/`, as
`behaviours_<your name>.csv` and `loops_<your name>.csv`. A copy is also kept on your laptop.

If a **red warning** appears, RFS isn't reachable. Keep working: nothing is lost. Reconnect
RFS and press **Ctrl+S** (Mac: **Cmd+S**); the warning disappears once it has saved.

## Working offline (no RFS / no hard drive)

You can annotate without RFS, using a compressed video someone sends you:

1. You receive **two files** per video, e.g. `GX010703.mp4` and `GX010703.reeftrack.json`. Keep
   them together in the same folder (e.g. `Documents/videos`). The `.json` file is needed,
   unless the hard drive is connected and **Compressed videos** points at the folder that
   contains `Maldives`: then the tool checks the `.mp4` against the original and recreates the
   `.json` itself.
2. Make a folder for your annotations, e.g. `Documents/fish_in_the_wild_annotations`.
3. Open the tool → **File → Library folders…** → set **Annotations** to that folder (the other
   two can stay empty or unreachable) → OK.
4. **File → Open compressed video…** → pick the `.mp4`. Annotate as usual. Files are saved in
   your annotations folder, in the same sub-folders they'll have on RFS
   (e.g. `Maldives/2024_02_14/AM/Cait/Left/GX010703/behaviours_<your name>.csv`).
5. When RFS works again, the maintainer copies everything across with one command
   (`reeftrack sync-annotations`), or you can zip your annotations folder and send it to them.

## Mask tracking (optional, for people who also outline the fish)

The **Annotate fish videos** icon is annotation-only: it installs a small toolkit without the AI
tracker. To track fish masks in the same window:

1. **Install once** (about 30 minutes, ~6 GB): same as step 2 above, but use
   `install_tracking_mac.command` (Mac, run from Terminal with `bash `) or
   `install_tracking_windows.bat` (Windows). You get a second Desktop icon, **Track fish videos**.
   The installer checks every package installed properly (it stops with a **MISSING** list if not:
   just run it again) and downloads the tracking model, so the first click on a fish is quick.
   For the better SAM 3 model, first request access at <https://huggingface.co/facebook/sam3> and run
   `hf auth login`; without it the tool uses SAM 2.1.
   Tracking is quick on an Apple Silicon Mac; on a laptop without an NVIDIA GPU (most Windows
   laptops, Intel Macs) it runs on the CPU and is slow.
2. Double-click **Track fish videos** and open a video exactly as above (from the hard drive, or
   **File → Open compressed video…**). The **Tracking** tab is now enabled.
3. Click the fish to start a mask, then let it track. For a tiny fish, drag a box around it to zoom
   (**Z**/**Esc** to go back), **E** cycles image enhancement, **V** lets you drag the mask's points
   and re-track from the corrected mask. Clicks work once the status bar says **Tracker ready**
   (a few seconds; several minutes the very first time if the installer couldn't download the model).
4. Masks save automatically into the video's annotation folder, next to your CSVs, as
   `tracking_<your name>/` (`tracklet.json`, `masks.npz`, and `tracklet.csv` with one row per frame).
   Each person has their own folder, so two people tracking the same video never overwrite each other.
   Like the CSVs, a copy is also kept on your laptop: if RFS drops, nothing is lost, and the masks
   reach RFS on your next change, Ctrl+S, or the next time you open the video.
5. **Other people's labels.** If someone has already labelled this video, their behaviours and
   loops appear read-only: an extra strip at the bottom of the timeline, and as "Name: label" on the
   video. **Display → Others' labels** picks everyone, one person, or Off (remembered).
   **J** / **Shift+J** jump to the next / previous stretch they marked **Out of view**, which is where
   the mask is expected to drop. Their files are never changed; yours stay in your own files.

**Prepare videos before an annotation session.** Opening a video that has no compressed copy yet
means waiting about as long as the video while it is compressed (once, then it's quick for everyone).
Don't use `reeftrack prepare` for this: it does the same slow work again for every clip. Instead,
compress the videos you'll work on in advance, e.g. overnight, with the hard drive and RFS connected:

```bash
reeftrack compress-batch "/Volumes/<drive>/originaldata/Maldives/2024_02_14" --dry-run
```

(drop `--dry-run` to do it). Compressing while you track works too, but both get slower, and on a
Mac with 16 GB of memory or less it can become sluggish.

**Where to save.** Setting **Annotations** to the RFS folder (`fish_in_the_wild/annotations`) is
simplest: masks and labels go straight to RFS, and you see everyone else's labels. Alternatively:

**Saving locally first, copying to RFS later.** Set **Annotations** (File → Library folders…) to a
folder on your laptop, e.g. `Documents/fish_in_the_wild_annotations`. Everything (CSVs and
`tracking_<name>/` folders) is saved there in the same sub-folders it will have on RFS. To copy it
across later (never overwrites anything newer on RFS; `--dry-run` first shows what it would do):

```bash
reeftrack sync-annotations ~/Documents/fish_in_the_wild_annotations "/Volumes/RFS/Triggerfish Navigation/fish_in_the_wild/annotations" --dry-run
```

Until then the laptop holds the only copy: back the folder up (e.g. zip it to OneDrive) now and then.

## If something goes wrong

* *"is not inside the original-videos folder"*: pick the video from the hard drive's
  `originaldata` folder, or fix it in **File → Library folders…**.
* *"0 bytes" / "could not read"*: that video file is damaged on the drive. Skip it and tell the
  maintainer.
* *"already being compressed"*: someone else is compressing it right now. Choose
  **Open original**, or try again later.
* Anything else: take a screenshot of the window (and the black Terminal window behind it)
  and send it to the maintainer.
