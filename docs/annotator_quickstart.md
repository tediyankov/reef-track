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

## If something goes wrong

* *"is not inside the original-videos folder"*: pick the video from the hard drive's
  `originaldata` folder, or fix it in **File → Library folders…**.
* *"0 bytes" / "could not read"*: that video file is damaged on the drive. Skip it and tell the
  maintainer.
* *"already being compressed"*: someone else is compressing it right now. Choose
  **Open original**, or try again later.
* Anything else: take a screenshot of the window (and the black Terminal window behind it)
  and send it to the maintainer.
