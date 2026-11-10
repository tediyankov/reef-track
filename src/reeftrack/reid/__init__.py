"""Re-identification dataset export: still frames of each individual, picked by a fish detector.

Separate from the tracking / 3D pipeline: it only reads the shared library (compressed videos
plus behaviour annotations) and writes a self-contained dataset folder.

    intervals.py   which stretch of footage to use (longest run without occlusion)
    select.py      which frames to keep (highest detector confidence, spread out in time)
    export.py      tracker spreadsheet -> detector -> frames/ + metadata.csv

Run it with `reeftrack reid-export --help`.
"""
