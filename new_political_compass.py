#!/usr/bin/env python3
"""
New Political Compass

Required install:
    pip3 install matplotlib pandas numpy scipy shapely

Run:
    python3 new_political_compass.py

Edit CSV_PATH below if your CSV moves.
Expected CSV columns:
    name,x,y,group

Required: name, x, y
Optional: group
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import math
import sys
import tkinter as tk
from tkinter import messagebox

import numpy as np
import pandas as pd

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon

try:
    from scipy.spatial import Voronoi, QhullError
except ImportError as exc:  # pragma: no cover - user environment check
    raise SystemExit(
        "Missing dependency: scipy. Install with: pip3 install matplotlib pandas numpy scipy shapely"
    ) from exc

try:
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
except ImportError as exc:  # pragma: no cover - user environment check
    raise SystemExit(
        "Missing dependency: shapely. Install with: pip3 install matplotlib pandas numpy scipy shapely"
    ) from exc


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------

CSV_PATH = str(Path(__file__).with_name("ideology_coordinates.csv"))

APP_TITLE = "New Political Compass"
WORLD_MIN = -10.0
WORLD_MAX = 10.0
WORLD_SPAN = WORLD_MAX - WORLD_MIN
MIN_VIEW_SPAN = 0.75
SEARCH_FOCUS_SPAN = 6.0
MAX_SEARCH_RESULTS = 15
CLICK_DISTANCE_PIXELS = 16
PAN_CLICK_TOLERANCE_PIXELS = 5


# -----------------------------------------------------------------------------
# Visual settings
# -----------------------------------------------------------------------------

BACKGROUND = "#111111"
PANEL_BACKGROUND = "#171717"
GRID_MAJOR = "#3a3a3a"
GRID_MINOR = "#242424"
AXIS_COLOR = "#b7b7b7"
TEXT_COLOR = "#e7e7e7"
MUTED_TEXT = "#a8a8a8"
POINT_COLOR = "#f4f4f4"
POINT_EDGE = "#111111"
SELECTED_COLOR = "#fff176"
SELECTED_RING = "#ff4081"
VORONOI_EDGE = "#6d6d6d"
VORONOI_FILL = "#4fc3f7"
TOOLTIP_BACKGROUND = "#222222"
TOOLTIP_EDGE = "#eeeeee"


@dataclass(frozen=True)
class CompassPoint:
    """A single plotted point from the CSV."""

    name: str
    x: float
    y: float
    group: Optional[str]
    row_number: int


class DataLoadError(Exception):
    """Raised when the CSV cannot be loaded into valid graph data."""


# -----------------------------------------------------------------------------
# CSV loading
# -----------------------------------------------------------------------------


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() == ""


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with stripped/lowercase column names for friendlier CSVs."""
    renamed: Dict[str, str] = {}
    for col in df.columns:
        renamed[col] = str(col).strip().lower()
    return df.rename(columns=renamed)


def load_points(csv_path: str) -> Tuple[List[CompassPoint], List[str]]:
    path = Path(csv_path).expanduser()
    if not path.exists():
        raise DataLoadError(
            f"CSV file not found:\n{path}\n\nEdit CSV_PATH near the top of this file if the CSV moved."
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise DataLoadError(f"Could not read CSV file:\n{path}\n\n{exc}") from exc

    if df.empty:
        raise DataLoadError("The CSV file is empty.")

    df = _normalize_columns(df)
    df = df.dropna(how="all")

    required = {"name", "x", "y"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise DataLoadError(
            "The CSV is missing required column(s): "
            + ", ".join(missing)
            + "\n\nExpected at least: name,x,y"
        )

    points: List[CompassPoint] = []
    warnings: List[str] = []
    fallback_counter = 1

    for row_index, row in df.iterrows():
        row_number = int(row_index) + 2  # +2 because CSV header is row 1

        if all(_is_blank(row.get(col)) for col in df.columns):
            continue

        raw_x = row.get("x")
        raw_y = row.get("y")

        try:
            x = float(raw_x)
            y = float(raw_y)
        except (TypeError, ValueError):
            warnings.append(
                f"Skipped row {row_number}: x/y values must be numeric "
                f"(got x={raw_x!r}, y={raw_y!r})."
            )
            continue

        if not (math.isfinite(x) and math.isfinite(y)):
            warnings.append(
                f"Skipped row {row_number}: x/y values must be finite numbers "
                f"(got x={raw_x!r}, y={raw_y!r})."
            )
            continue

        if not (WORLD_MIN <= x <= WORLD_MAX and WORLD_MIN <= y <= WORLD_MAX):
            warnings.append(
                f"Skipped row {row_number}: point ({x:g}, {y:g}) is outside "
                f"the {WORLD_MIN:g} to {WORLD_MAX:g} bounds."
            )
            continue

        raw_name = row.get("name")
        if _is_blank(raw_name):
            name = f"Point {fallback_counter}"
            fallback_counter += 1
        else:
            name = str(raw_name).strip()

        group: Optional[str] = None
        if "group" in df.columns and not _is_blank(row.get("group")):
            group = str(row.get("group")).strip()

        points.append(CompassPoint(name=name, x=x, y=y, group=group, row_number=row_number))

    if not points:
        raise DataLoadError("No valid in-bounds points were found in the CSV.")

    if len(points) < 3:
        raise DataLoadError(
            "At least 3 valid in-bounds points are required to build a Voronoi diagram."
        )

    unique_coords = {(round(p.x, 12), round(p.y, 12)) for p in points}
    if len(unique_coords) < 3:
        raise DataLoadError(
            "At least 3 unique point locations are required to build a Voronoi diagram.\n"
            "Several rows may share the exact same x/y coordinates."
        )

    return points, warnings


# -----------------------------------------------------------------------------
# Voronoi helpers
# -----------------------------------------------------------------------------


def voronoi_finite_polygons_2d(
    vor: Voronoi, radius: Optional[float] = None
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Reconstruct infinite Voronoi regions into finite polygons.

    Based on the common SciPy Voronoi finite polygon recipe, adjusted for modern
    NumPy compatibility. The polygons are later clipped to the compass bounds.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("This helper only supports 2D Voronoi diagrams.")

    new_regions: List[List[int]] = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = float(np.ptp(vor.points, axis=0).max() * 4)
        radius = max(radius, WORLD_SPAN * 2)

    all_ridges: Dict[int, List[Tuple[int, int, int]]] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(int(p1), []).append((int(p2), int(v1), int(v2)))
        all_ridges.setdefault(int(p2), []).append((int(p1), int(v1), int(v2)))

    for p1, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]

        if not vertices:
            new_regions.append([])
            continue

        if all(v >= 0 for v in vertices):
            new_regions.append([int(v) for v in vertices])
            continue

        ridges = all_ridges.get(int(p1), [])
        new_region = [int(v) for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            tangent = vor.points[p2] - vor.points[p1]
            norm = np.linalg.norm(tangent)
            if norm == 0:
                continue
            tangent /= norm

            normal = np.array([-tangent[1], tangent[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        if not new_region:
            new_regions.append([])
            continue

        region_vertices = np.asarray([new_vertices[v] for v in new_region])
        centroid = region_vertices.mean(axis=0)
        angles = np.arctan2(
            region_vertices[:, 1] - centroid[1], region_vertices[:, 0] - centroid[0]
        )
        new_region = [int(v) for v in np.asarray(new_region)[np.argsort(angles)]]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


def _largest_polygon(geometry: object) -> Optional[Polygon]:
    """Return a usable Polygon from a possible Shapely geometry result."""
    if geometry is None or getattr(geometry, "is_empty", True):
        return None

    if isinstance(geometry, Polygon):
        return geometry

    if isinstance(geometry, MultiPolygon):
        polygons = list(geometry.geoms)
        return max(polygons, key=lambda poly: poly.area) if polygons else None

    if isinstance(geometry, GeometryCollection):
        polygons = [geom for geom in geometry.geoms if isinstance(geom, Polygon)]
        return max(polygons, key=lambda poly: poly.area) if polygons else None

    return None


def build_clipped_voronoi_cells(
    coords: np.ndarray,
) -> Tuple[List[Optional[List[Tuple[float, float]]]], Optional[str]]:
    """Build Voronoi polygons clipped to the -10 to 10 square."""
    bounds = box(WORLD_MIN, WORLD_MIN, WORLD_MAX, WORLD_MAX)

    try:
        vor = Voronoi(coords)
    except QhullError:
        # QJ joggles nearly-collinear points so the diagram can still be created.
        # This is useful for early datasets where many points may line up.
        try:
            vor = Voronoi(coords, qhull_options="Qbb Qc Qz QJ")
        except QhullError as exc:
            return [None for _ in coords], f"Could not build Voronoi diagram: {exc}"

    try:
        regions, vertices = voronoi_finite_polygons_2d(vor)
    except Exception as exc:
        return [None for _ in coords], f"Could not reconstruct Voronoi regions: {exc}"

    cells: List[Optional[List[Tuple[float, float]]]] = []
    for region in regions[: len(coords)]:
        if not region:
            cells.append(None)
            continue

        try:
            polygon = Polygon(vertices[region])
            if not polygon.is_valid:
                polygon = polygon.buffer(0)

            clipped = polygon.intersection(bounds)
            clipped_polygon = _largest_polygon(clipped)

            if clipped_polygon is None or clipped_polygon.is_empty:
                cells.append(None)
                continue

            exterior = list(clipped_polygon.exterior.coords)
            cells.append([(float(x), float(y)) for x, y in exterior])
        except Exception:
            cells.append(None)

    # SciPy should return one region per point, but this keeps drawing safe.
    while len(cells) < len(coords):
        cells.append(None)

    return cells[: len(coords)], None


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------


class NewPoliticalCompassApp:
    def __init__(self, root: tk.Tk, points: Sequence[CompassPoint], warnings: Sequence[str]):
        self.root = root
        self.points = list(points)
        self.warnings = list(warnings)

        self.selected_index: Optional[int] = None
        self.listbox_indices: List[int] = []
        self.drag_start: Optional[Dict[str, object]] = None
        self.is_dragging = False

        self.coord_to_point_indices = self._build_coord_index()
        self.unique_coords, self.unique_coord_to_point_index = self._build_unique_coords()
        self.voronoi_cells, self.voronoi_warning = build_clipped_voronoi_cells(self.unique_coords)

        self.root.title(APP_TITLE)
        self.root.configure(bg=BACKGROUND)
        self.root.geometry("1180x850")
        self.root.minsize(760, 560)

        self.search_var = tk.StringVar()
        self.status_var = tk.StringVar()

        self._build_ui()
        self._connect_events()
        self._update_status()
        self.draw_plot()

        if self.warnings:
            print("\nCSV warnings:")
            for warning in self.warnings:
                print("-", warning)
            print()

        if self.voronoi_warning:
            print("Voronoi warning:", self.voronoi_warning)
            messagebox.showwarning(
                APP_TITLE,
                "The points loaded, but the Voronoi diagram could not be built.\n\n"
                + self.voronoi_warning,
            )

    # ------------------------------------------------------------------
    # Data organization
    # ------------------------------------------------------------------

    def _build_coord_index(self) -> Dict[Tuple[float, float], List[int]]:
        coord_index: Dict[Tuple[float, float], List[int]] = {}
        for index, point in enumerate(self.points):
            key = (round(point.x, 12), round(point.y, 12))
            coord_index.setdefault(key, []).append(index)
        return coord_index

    def _build_unique_coords(self) -> Tuple[np.ndarray, List[int]]:
        coords: List[Tuple[float, float]] = []
        representative_indices: List[int] = []
        seen: set[Tuple[float, float]] = set()

        for index, point in enumerate(self.points):
            key = (round(point.x, 12), round(point.y, 12))
            if key in seen:
                continue
            seen.add(key)
            coords.append((point.x, point.y))
            representative_indices.append(index)

        return np.asarray(coords, dtype=float), representative_indices

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        top_frame = tk.Frame(self.root, bg=PANEL_BACKGROUND, padx=10, pady=8)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        title_label = tk.Label(
            top_frame,
            text=APP_TITLE,
            bg=PANEL_BACKGROUND,
            fg=TEXT_COLOR,
            font=("Helvetica Neue", 16, "bold"),
        )
        title_label.pack(side=tk.LEFT, padx=(0, 18))

        search_label = tk.Label(
            top_frame,
            text="Search point:",
            bg=PANEL_BACKGROUND,
            fg=TEXT_COLOR,
            font=("Helvetica Neue", 12),
        )
        search_label.pack(side=tk.LEFT, padx=(0, 6))

        self.search_entry = tk.Entry(
            top_frame,
            textvariable=self.search_var,
            width=34,
            bg="#252525",
            fg=TEXT_COLOR,
            insertbackground=TEXT_COLOR,
            relief=tk.FLAT,
            font=("Helvetica Neue", 12),
        )
        self.search_entry.pack(side=tk.LEFT, padx=(0, 8), ipady=4)

        self.clear_button = tk.Button(
            top_frame,
            text="Clear",
            command=self.clear_selection,
            bg="#2b2b2b",
            fg=TEXT_COLOR,
            activebackground="#3a3a3a",
            activeforeground=TEXT_COLOR,
            relief=tk.FLAT,
            font=("Helvetica Neue", 11),
            padx=10,
        )
        self.clear_button.pack(side=tk.LEFT, padx=(0, 12))

        status_label = tk.Label(
            top_frame,
            textvariable=self.status_var,
            bg=PANEL_BACKGROUND,
            fg=MUTED_TEXT,
            font=("Helvetica Neue", 11),
            anchor="w",
        )
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        body = tk.Frame(self.root, bg=BACKGROUND)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.results_frame = tk.Frame(body, bg=BACKGROUND, padx=8, pady=8)
        self.results_frame.pack(side=tk.RIGHT, fill=tk.Y)

        results_label = tk.Label(
            self.results_frame,
            text="Matches",
            bg=BACKGROUND,
            fg=TEXT_COLOR,
            font=("Helvetica Neue", 12, "bold"),
        )
        results_label.pack(anchor="w", pady=(0, 6))

        self.results_listbox = tk.Listbox(
            self.results_frame,
            width=34,
            height=18,
            bg="#181818",
            fg=TEXT_COLOR,
            selectbackground="#444444",
            selectforeground=SELECTED_COLOR,
            highlightthickness=1,
            highlightbackground="#333333",
            relief=tk.FLAT,
            activestyle="none",
            font=("Helvetica Neue", 11),
        )
        self.results_listbox.pack(side=tk.TOP, fill=tk.Y)

        help_text = (
            "Scroll: zoom\n"
            "Drag: pan\n"
            "Click point: info\n"
            "Esc: clear"
        )
        help_label = tk.Label(
            self.results_frame,
            text=help_text,
            bg=BACKGROUND,
            fg=MUTED_TEXT,
            justify=tk.LEFT,
            font=("Helvetica Neue", 10),
        )
        help_label.pack(anchor="w", pady=(12, 0))

        fig = Figure(figsize=(9, 8), dpi=100, facecolor=BACKGROUND)
        self.fig = fig
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, master=body)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.configure(bg=BACKGROUND, highlightthickness=0)
        self.canvas_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.search_var.trace_add("write", self._on_search_changed)
        self.results_listbox.bind("<<ListboxSelect>>", self._on_result_selected)
        self.search_entry.bind("<Return>", self._on_search_enter)
        self.search_entry.bind("<Escape>", self._on_escape)
        self.root.bind("<Escape>", self._on_escape)

    def _connect_events(self) -> None:
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _setup_axes(self) -> None:
        self.ax.set_facecolor(BACKGROUND)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlim(WORLD_MIN, WORLD_MAX)
        self.ax.set_ylim(WORLD_MIN, WORLD_MAX)

        major_ticks = np.arange(WORLD_MIN, WORLD_MAX + 1, 2)
        minor_ticks = np.arange(WORLD_MIN, WORLD_MAX + 0.5, 1)
        self.ax.set_xticks(major_ticks)
        self.ax.set_yticks(major_ticks)
        self.ax.set_xticks(minor_ticks, minor=True)
        self.ax.set_yticks(minor_ticks, minor=True)

        self.ax.grid(which="major", color=GRID_MAJOR, linewidth=0.75, alpha=0.75)
        self.ax.grid(which="minor", color=GRID_MINOR, linewidth=0.45, alpha=0.85)
        self.ax.axhline(0, color=AXIS_COLOR, linewidth=1.2, alpha=0.9, zorder=2)
        self.ax.axvline(0, color=AXIS_COLOR, linewidth=1.2, alpha=0.9, zorder=2)

        for spine in self.ax.spines.values():
            spine.set_color("#555555")
            spine.set_linewidth(1.0)

        self.ax.tick_params(axis="both", colors=MUTED_TEXT, labelsize=9)
        self.ax.set_xlabel("x", color=MUTED_TEXT)
        self.ax.set_ylabel("y", color=MUTED_TEXT)
        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.98, bottom=0.06)

    def draw_plot(self, keep_view: bool = False) -> None:
        old_xlim = self.ax.get_xlim() if keep_view else (WORLD_MIN, WORLD_MAX)
        old_ylim = self.ax.get_ylim() if keep_view else (WORLD_MIN, WORLD_MAX)

        self.ax.clear()
        self._setup_axes()
        self.ax.set_xlim(*old_xlim)
        self.ax.set_ylim(*old_ylim)

        self._draw_voronoi_cells()
        self._draw_points()
        self._draw_labels()
        self._draw_selected_point()

        self.canvas.draw_idle()

    def _draw_voronoi_cells(self) -> None:
        for unique_index, cell in enumerate(self.voronoi_cells):
            if not cell:
                continue

            patch = MplPolygon(
                cell,
                closed=True,
                facecolor=VORONOI_FILL,
                edgecolor=VORONOI_EDGE,
                linewidth=0.9,
                alpha=0.12,
                zorder=1,
            )
            self.ax.add_patch(patch)

            # Future group-coloring logic belongs here:
            # representative = self.points[self.unique_coord_to_point_index[unique_index]]
            # if representative.group is not None:
            #     use a stable color assigned to representative.group

        boundary = MplPolygon(
            [
                (WORLD_MIN, WORLD_MIN),
                (WORLD_MAX, WORLD_MIN),
                (WORLD_MAX, WORLD_MAX),
                (WORLD_MIN, WORLD_MAX),
            ],
            closed=True,
            fill=False,
            edgecolor="#888888",
            linewidth=1.2,
            alpha=0.8,
            zorder=3,
        )
        self.ax.add_patch(boundary)

        # Future umbrella-region logic belongs after individual cells are available:
        # merge cells whose CompassPoint.group values match, then draw broader outlines.

    def _draw_points(self) -> None:
        x_values = [point.x for point in self.points]
        y_values = [point.y for point in self.points]
        self.ax.scatter(
            x_values,
            y_values,
            s=34,
            c=POINT_COLOR,
            edgecolors=POINT_EDGE,
            linewidths=0.75,
            zorder=5,
        )

    def _draw_labels(self) -> None:
        for index, point in enumerate(self.points):
            is_selected = index == self.selected_index
            self.ax.text(
                point.x + 0.08,
                point.y + 0.08,
                point.name,
                color=SELECTED_COLOR if is_selected else TEXT_COLOR,
                fontsize=9.5 if is_selected else 8.5,
                fontweight="bold" if is_selected else "normal",
                alpha=1.0 if is_selected else 0.82,
                zorder=7 if is_selected else 6,
                clip_on=True,
            )

    def _draw_selected_point(self) -> None:
        if self.selected_index is None:
            return

        point = self.points[self.selected_index]
        self.ax.scatter(
            [point.x],
            [point.y],
            s=190,
            facecolors="none",
            edgecolors=SELECTED_RING,
            linewidths=2.4,
            zorder=9,
        )
        self.ax.scatter(
            [point.x],
            [point.y],
            s=62,
            c=SELECTED_COLOR,
            edgecolors="#000000",
            linewidths=1.0,
            zorder=10,
        )
        self.ax.annotate(
            self._format_point_info(self.selected_index),
            xy=(point.x, point.y),
            xytext=(14, 14),
            textcoords="offset points",
            color=TEXT_COLOR,
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": TOOLTIP_BACKGROUND,
                "edgecolor": TOOLTIP_EDGE,
                "alpha": 0.94,
            },
            arrowprops={"arrowstyle": "->", "color": TOOLTIP_EDGE, "lw": 1.0},
            zorder=11,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _on_search_changed(self, *_args: object) -> None:
        query = self.search_var.get().strip().lower()
        self.results_listbox.delete(0, tk.END)
        self.listbox_indices = []

        if not query:
            if self.selected_index is not None:
                self.selected_index = None
                self.draw_plot(keep_view=True)
            return

        matches = [
            (index, point)
            for index, point in enumerate(self.points)
            if query in point.name.lower()
            or (point.group is not None and query in point.group.lower())
        ]
        matches = matches[:MAX_SEARCH_RESULTS]

        for index, point in matches:
            group_text = f" — {point.group}" if point.group else ""
            self.results_listbox.insert(
                tk.END, f"{point.name} ({point.x:g}, {point.y:g}){group_text}"
            )
            self.listbox_indices.append(index)

    def _on_result_selected(self, _event: object = None) -> None:
        selection = self.results_listbox.curselection()
        if not selection:
            return

        listbox_index = int(selection[0])
        if listbox_index >= len(self.listbox_indices):
            return

        self.select_point(self.listbox_indices[listbox_index], focus=True)

    def _on_search_enter(self, _event: object = None) -> str:
        selection = self.results_listbox.curselection()
        if selection:
            self._on_result_selected()
            return "break"

        if self.listbox_indices:
            self.results_listbox.selection_clear(0, tk.END)
            self.results_listbox.selection_set(0)
            self.results_listbox.activate(0)
            self.select_point(self.listbox_indices[0], focus=True)
            return "break"

        return "break"

    def _on_escape(self, _event: object = None) -> str:
        self.clear_selection()
        return "break"

    def clear_selection(self) -> None:
        self.search_var.set("")
        self.results_listbox.delete(0, tk.END)
        self.listbox_indices = []
        self.selected_index = None
        self.draw_plot(keep_view=True)

    def select_point(self, index: int, focus: bool = False) -> None:
        if index < 0 or index >= len(self.points):
            return

        self.selected_index = index
        if focus:
            self.center_on_point(self.points[index])
        else:
            self.draw_plot(keep_view=True)
        self._update_status()

    # ------------------------------------------------------------------
    # Point info / selection
    # ------------------------------------------------------------------

    def _format_point_info(self, index: int) -> str:
        point = self.points[index]
        lines = [point.name, f"x: {point.x:g}", f"y: {point.y:g}"]
        if point.group:
            lines.append(f"group: {point.group}")

        same_coord_indices = self.coord_to_point_indices.get(
            (round(point.x, 12), round(point.y, 12)), []
        )
        other_names = [self.points[i].name for i in same_coord_indices if i != index]
        if other_names:
            preview = ", ".join(other_names[:3])
            if len(other_names) > 3:
                preview += f", +{len(other_names) - 3} more"
            lines.append(f"same spot: {preview}")

        return "\n".join(lines)

    def _nearest_point_index(self, event: object) -> Optional[int]:
        if event.x is None or event.y is None:
            return None

        display_coords = self.ax.transData.transform(
            np.asarray([(point.x, point.y) for point in self.points], dtype=float)
        )
        pointer = np.asarray([event.x, event.y], dtype=float)
        distances = np.linalg.norm(display_coords - pointer, axis=1)

        nearest_index = int(np.argmin(distances))
        if distances[nearest_index] <= CLICK_DISTANCE_PIXELS:
            return nearest_index
        return None

    # ------------------------------------------------------------------
    # Zoom / pan
    # ------------------------------------------------------------------

    def _on_scroll(self, event: object) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        current_width = current_xlim[1] - current_xlim[0]
        current_height = current_ylim[1] - current_ylim[0]

        step = getattr(event, "step", 0)
        if step > 0 or getattr(event, "button", None) == "up":
            scale = 0.82
        else:
            scale = 1.18

        new_width = float(np.clip(current_width * scale, MIN_VIEW_SPAN, WORLD_SPAN))
        new_height = float(np.clip(current_height * scale, MIN_VIEW_SPAN, WORLD_SPAN))

        rel_x = (event.xdata - current_xlim[0]) / current_width if current_width else 0.5
        rel_y = (event.ydata - current_ylim[0]) / current_height if current_height else 0.5

        new_xmin = event.xdata - rel_x * new_width
        new_ymin = event.ydata - rel_y * new_height
        new_xlim = self._clamp_limits(new_xmin, new_xmin + new_width)
        new_ylim = self._clamp_limits(new_ymin, new_ymin + new_height)

        self.ax.set_xlim(*new_xlim)
        self.ax.set_ylim(*new_ylim)
        self.canvas.draw_idle()

    def _on_mouse_press(self, event: object) -> None:
        if event.inaxes != self.ax or getattr(event, "button", None) != 1:
            return

        self.drag_start = {
            "x_pixel": event.x,
            "y_pixel": event.y,
            "xlim": self.ax.get_xlim(),
            "ylim": self.ax.get_ylim(),
            "bbox_width": self.ax.bbox.width,
            "bbox_height": self.ax.bbox.height,
        }
        self.is_dragging = False

    def _on_mouse_move(self, event: object) -> None:
        if self.drag_start is None or event.x is None or event.y is None:
            return

        start_x = float(self.drag_start["x_pixel"])
        start_y = float(self.drag_start["y_pixel"])
        dx_pixels = event.x - start_x
        dy_pixels = event.y - start_y

        if abs(dx_pixels) > PAN_CLICK_TOLERANCE_PIXELS or abs(dy_pixels) > PAN_CLICK_TOLERANCE_PIXELS:
            self.is_dragging = True

        if not self.is_dragging:
            return

        xlim = self.drag_start["xlim"]
        ylim = self.drag_start["ylim"]
        bbox_width = max(float(self.drag_start["bbox_width"]), 1.0)
        bbox_height = max(float(self.drag_start["bbox_height"]), 1.0)

        width = xlim[1] - xlim[0]
        height = ylim[1] - ylim[0]

        dx_data = dx_pixels / bbox_width * width
        dy_data = dy_pixels / bbox_height * height

        new_xlim = self._clamp_limits(xlim[0] - dx_data, xlim[1] - dx_data)
        new_ylim = self._clamp_limits(ylim[0] - dy_data, ylim[1] - dy_data)

        self.ax.set_xlim(*new_xlim)
        self.ax.set_ylim(*new_ylim)
        self.canvas.draw_idle()

    def _on_mouse_release(self, event: object) -> None:
        if self.drag_start is None:
            return

        was_dragging = self.is_dragging
        self.drag_start = None
        self.is_dragging = False

        if was_dragging:
            return

        if event.inaxes != self.ax:
            return

        nearest_index = self._nearest_point_index(event)
        if nearest_index is not None:
            self.select_point(nearest_index, focus=False)

    def _clamp_limits(self, low: float, high: float) -> Tuple[float, float]:
        span = high - low
        span = float(np.clip(span, MIN_VIEW_SPAN, WORLD_SPAN))

        low = float(low)
        high = low + span

        if low < WORLD_MIN:
            low = WORLD_MIN
            high = WORLD_MIN + span
        if high > WORLD_MAX:
            high = WORLD_MAX
            low = WORLD_MAX - span

        return low, high

    def center_on_point(self, point: CompassPoint) -> None:
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()

        width = min(current_xlim[1] - current_xlim[0], SEARCH_FOCUS_SPAN)
        height = min(current_ylim[1] - current_ylim[0], SEARCH_FOCUS_SPAN)

        xlim = self._clamp_limits(point.x - width / 2, point.x + width / 2)
        ylim = self._clamp_limits(point.y - height / 2, point.y + height / 2)

        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.draw_plot(keep_view=True)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _update_status(self) -> None:
        group_count = sum(1 for point in self.points if point.group)
        warning_text = f" | {len(self.warnings)} skipped row(s)" if self.warnings else ""
        self.status_var.set(
            f"{len(self.points)} point(s) loaded | {group_count} with group values{warning_text}"
        )


def main() -> None:
    root = tk.Tk()
    root.withdraw()

    try:
        points, warnings = load_points(CSV_PATH)
    except DataLoadError as exc:
        print(exc, file=sys.stderr)
        messagebox.showerror(APP_TITLE, str(exc))
        root.destroy()
        return

    root.deiconify()
    app = NewPoliticalCompassApp(root, points, warnings)
    root.mainloop()


if __name__ == "__main__":
    main()
