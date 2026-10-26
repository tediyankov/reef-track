# Behaviour and loop annotation guide

This guide is for annotators. You'll mark **when** the focal Picasso triggerfish shows each
behaviour, as start/end frames on a video. No 3D or tracking knowledge is needed.

## Start the tool

```bash
conda activate ~/envs/reef-track
reeftrack gui data/runs/<run name> --behaviour-only
```

`--behaviour-only` opens the video with the **Behaviour** tab and skips loading the AI tracking
model, so it starts in seconds on any laptop. The first time you add an event you'll be asked for
your name or initials; they're saved with every row.

If someone has already tracked the fish, it appears with a coloured outline, which helps you
follow the right individual. You can't change the outline in this mode.

## Mark an event

1. Play the video (**Space**) or step through it (**←/→** one frame, **Shift+←/→** one second).
2. At the **first** frame of a behaviour, press **I** (or click *Start [I]*).
3. Move to the **last** frame of that behaviour and press **O** (or click *End [O]*).
4. A window opens. Pick the behaviour by clicking it or pressing its key:

   | key | behaviour | key | behaviour |
   |---|---|---|---|
   | 1 | Foraging | 9 | Sheltering |
   | 2 | Near other Picasso triggerfish | 0 | Stationary |
   | 3 | Chasing | A | Erratic swimming |
   | 4 | Escaping | B | Calm swimming |
   | 5 | Fighting | C | Other (add a comment!) |
   | 6 | Cleaner fish interaction | D | Out of view |
   | 7 | Scratching | E | Calib |
   | 8 | Housekeeping | | |

   The definition of the highlighted behaviour (from the ethogram) is shown in the window.

5. Optionally write a **comment** (uncertainty, other fish involved, why it ended…), then **OK**.

The event is saved immediately. There is no save button and nothing is lost if the program
closes. A behaviour must be chosen before OK can be pressed.

## Rules of thumb

* Start and end frames are **inclusive** (the event covers both frames).
* Events **may overlap**: if the fish forages while near another Picasso triggerfish, mark both.
* If you press O before I by mistake, the frames are simply swapped.
* While the fish is out of view, don't annotate. Leave a gap, or mention it in a comment.

## Review and fix

* The coloured lane under the timeline shows all events; the labels on the video show what is
  active at the current frame.
* In the table: **click** a row to jump to its start, **double-click** (or *Edit…*) to change the
  behaviour, frames or comment, and use *Delete* to remove it.
* *Clear* discards a start/end you marked but haven't saved yet.

## Loops (Loops tab)

A **loop** is when the focal fish comes back to a place it has already been.

1. Open the **Loops** tab. **I**, **O** and **B** now act on loops instead of behaviours.
2. At the frame where the loop **starts** (the fish is at a spot you'll recognise, e.g. next to a
   particular coral head), press **I**. That frame now shows as a small picture in the top-right
   corner of the video.
3. Play on. When you can see the fish is **back at that same spot**, press **O**.
4. The window shows the start and return frames side by side. Choose your **confidence**
   (certain / probable / possible), optionally name the **landmark** (e.g. "big bommie") and add a
   comment, then **OK**.

Loops can overlap or sit inside each other. A loop must span at least two frames.

If the fish has been placed in 3D (`reeftrack fuse`), each loop row also gets measured values:
how close the fish really got back to its start point (`return_distance`), how far it swam
(`path_length`) and how far it strayed from the start (`max_excursion`). These are in metres
when the stereo trajectory exists. They're filled in automatically and updated whenever the
trajectory is recomputed. They stay empty if the fish wasn't tracked at the loop's start or
end frame.

## Where the data goes

`data/runs/<run name>/annotations/behaviours.csv` has one row per event:

| column | meaning |
|---|---|
| `event_id` | unique number within this run |
| `behaviour` | one of the categories above |
| `start_frame`, `end_frame` | frame numbers as shown in the tool (inclusive) |
| `start_source_frame`, `end_source_frame` | the same instants in the original video file |
| `start_time_s`, `end_time_s`, `duration_s` | seconds in the original video |
| `source_video`, `run` | which video and clip the row belongs to |
| `comment`, `annotator` | your comment and name |
| `created`, `modified` | timestamps |

The file opens directly in Excel, R or Python.

`data/runs/<run name>/annotations/loops.csv` has one row per loop: `loop_id`, `confidence`,
`landmark`, the same frame/time columns as above, `return_distance`, `path_length`,
`max_excursion`, `start_x/y/z` and `end_x/y/z` (fish positions), `units` (`m` = metres),
`trajectory_source`, then `source_video`, `run`, `comment`, `annotator`, `created` and `modified`.
