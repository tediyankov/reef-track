"""3D outputs for humans: a self-contained HTML viewer and a combined PLY.

* fusion/scene.html  open in any browser (works offline, shareable): reef points in true
                     colour, fish path coloured by time, camera (diver) path in grey.
* fusion/scene.ply   reef + fish path as bright points, for MeshLab / CloudCompare.
"""

from __future__ import annotations

import webbrowser

import numpy as np

from reeftrack.recon.geometry import camera_center

MAX_HTML_POINTS = 150_000


def _load_reef(run):
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(run.recon_dir / "reef.ply"))
    return np.asarray(pcd.points), np.asarray(pcd.colors)


def write_scene(run, rows: list[dict], html_path=None, write_ply: bool = True) -> None:
    import open3d as o3d
    import plotly.graph_objects as go

    P, C = _load_reef(run)
    cams = np.load(run.recon_dir / "cameras.npz")
    centres = np.stack([camera_center(E) for E in cams["extrinsic"]])
    fmap = run.frame_map()

    sel = np.random.default_rng(0).choice(len(P), min(len(P), MAX_HTML_POINTS), replace=False)
    rgb = [f"rgb({r},{g},{b})" for r, g, b in (C[sel] * 255).astype(int)]
    traces = [
        go.Scatter3d(x=P[sel, 0], y=P[sel, 1], z=P[sel, 2], mode="markers", name="reef",
                     marker=dict(size=1.5, color=rgb), hoverinfo="skip"),
        go.Scatter3d(x=centres[:, 0], y=centres[:, 1], z=centres[:, 2], mode="lines", name="camera (diver)",
                     line=dict(color="grey", width=3)),
    ]
    tracked = [r for r in rows if r["kind"] != "gap_interp"]
    if tracked:
        T = np.array([[r["x"], r["y"], r["z"]] for r in tracked])
        t = fmap.time_s(np.array([r["proxy_frame"] for r in tracked]))
        traces.append(go.Scatter3d(
            x=T[:, 0], y=T[:, 1], z=T[:, 2], mode="lines+markers", name="fish",
            marker=dict(size=2.5, color=t, colorscale="Plasma", colorbar=dict(title="time (s)")),
            line=dict(color="rgba(255,60,60,0.5)", width=2),
            text=[f"frame {fmap.to_source(r['proxy_frame'])} | seg {r['segment_id']} | {r['kind']}" for r in tracked],
            hoverinfo="text"))
    gaps = [r for r in rows if r["kind"] == "gap_interp"]
    if gaps:
        G = np.array([[r["x"], r["y"], r["z"]] for r in gaps])
        traces.append(go.Scatter3d(x=G[:, 0], y=G[:, 1], z=G[:, 2], mode="markers", name="fish (gap interp.)",
                                   marker=dict(size=2, color="white", symbol="circle-open")))
    fig = go.Figure(traces)
    fig.update_layout(scene=dict(aspectmode="data", bgcolor="rgb(8,30,40)", xaxis_title="x", yaxis_title="y",
                                 zaxis_title="z (up)"),
                      paper_bgcolor="rgb(8,30,40)", font_color="white", margin=dict(l=0, r=0, t=30, b=0),
                      title=f"{run.root.name}: reef" + (" + focal fish trajectory" if rows else ""))
    fig.write_html(html_path or run.fusion_dir / "scene.html", include_plotlyjs=True)
    if not write_ply:
        return

    # combined PLY: reef + fish path in red (time-ordered)
    pcd = o3d.geometry.PointCloud()
    if tracked:
        pts = np.vstack([P, T])
        cols = np.vstack([C, np.tile([1.0, 0.1, 0.1], (len(T), 1))])
    else:
        pts, cols = P, C
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    o3d.io.write_point_cloud(str(run.fusion_dir / "scene.ply"), pcd)


def open_scene(run, native: bool = False) -> None:
    if not native:
        html = run.fusion_dir / "scene.html"
        if not html.exists():  # no fusion yet: show the reef-only preview
            html = run.recon_dir / "reef.html"
        webbrowser.open(html.resolve().as_uri())
        return
    import csv

    import open3d as o3d

    geoms = [o3d.io.read_point_cloud(str(run.recon_dir / "reef.ply"))]
    traj = run.fusion_dir / "trajectory.csv"
    with open(traj) if traj.exists() else open("/dev/null") as fh:
        T = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in csv.DictReader(fh)
                      if r["kind"] != "gap_interp"])
    if len(T) > 1:
        ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(T),
                                  o3d.utility.Vector2iVector(np.c_[np.arange(len(T) - 1), np.arange(1, len(T))]))
        ls.paint_uniform_color([1, 0.1, 0.1])
        geoms.append(ls)
    o3d.visualization.draw_geometries(geoms, window_name=run.root.name)


def equal_density(clouds, max_points: int = 120_000) -> tuple[list, float]:
    """Downsample several clouds on the *same* metric voxel grid, so displays are comparable.

    A fixed point budget per cloud would show a denser cloud more sparsely (grainier),
    which made the stereo cloud look worse than it is.
    """
    v = 0.004
    while True:
        out = [c.voxel_down_sample(v) for c in clouds]
        if max(len(o.points) for o in out) <= max_points:
            return out, v
        v *= 1.25


def _scene_traces(pcd, cams, traj, name, colour):
    import plotly.graph_objects as go

    P, C = np.asarray(pcd.points), np.asarray(pcd.colors)
    rgb = [f"rgb({r},{g},{b})" for r, g, b in (C * 255).astype(int)]
    tr = [go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", marker=dict(size=1.6, color=rgb),
                       name=f"{name} reef", hoverinfo="skip", showlegend=False),
          go.Scatter3d(x=cams[:, 0], y=cams[:, 1], z=cams[:, 2], mode="lines", line=dict(color="grey", width=3),
                       name=f"{name} camera")]
    if traj is not None:
        tr.append(go.Scatter3d(x=traj[:, 0], y=traj[:, 1], z=traj[:, 2], mode="lines", line=dict(color=colour, width=5),
                               name=f"{name} fish"))
    return tr


def write_compare_html(path, title, pm, ps, cams_m, cams_s, t_centres, ratios, traj, report) -> None:
    """Side-by-side mono (aligned, metres) vs stereo 3D scenes + mono scale-drift curve."""
    import json as _json

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}], [{"type": "xy", "colspan": 2}, None]],
                        row_heights=[0.72, 0.28], vertical_spacing=0.06,
                        subplot_titles=("Mono (Wat3R, Left only) - aligned & scaled to metres",
                                        "Stereo (Wat3R, Left+Right, 7 cm baseline) - metres",
                                        "Mono scale drift: local / global scale (1.0 = no drift)"))
    (pm, ps), voxel = equal_density([pm, ps])
    report = {**report, "display_voxel_cm": round(100 * voxel, 2)}
    for t in _scene_traces(pm, cams_m, traj.get("mono"), "mono", "#3b82f6"):
        fig.add_trace(t, row=1, col=1)
    for t in _scene_traces(ps, cams_s, traj.get("stereo"), "stereo", "#f97316"):
        fig.add_trace(t, row=1, col=2)
    fig.add_trace(go.Scatter(x=t_centres, y=ratios, mode="lines+markers", line=dict(color="#3b82f6"), name="mono scale ratio"),
                  row=2, col=1)
    if len(t_centres):  # reference line (add_hline is unsupported next to 3D subplots)
        fig.add_trace(go.Scatter(x=[t_centres[0], t_centres[-1]], y=[1.0, 1.0], mode="lines",
                                 line=dict(color="#f97316", dash="dash"), name="no drift"), row=2, col=1)
    fig.update_xaxes(title_text="time in source video (s)", row=2, col=1)
    scene = dict(aspectmode="data", bgcolor="rgb(8,30,40)", xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z up (m)")
    cam = dict(eye=dict(x=1.3, y=-1.3, z=0.9))  # same initial viewpoint for both panels
    fig.update_layout(scene={**scene, "camera": cam}, scene2={**scene, "camera": cam}, paper_bgcolor="rgb(8,30,40)", plot_bgcolor="rgb(15,40,52)",
                      font_color="white", height=1000, title=f"{title}: mono vs stereo",
                      margin=dict(l=10, r=10, t=70, b=10))
    html = fig.to_html(include_plotlyjs=True, full_html=True)
    summary = _json.dumps(report, indent=1).replace("<", "&lt;")
    html = html.replace("</body>", f"<pre style='color:#dde6ea;background:#0f1a21;padding:16px;font-size:13px'>{summary}</pre></body>")
    path.write_text(html)
