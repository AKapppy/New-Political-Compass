#!/usr/bin/env python3
"""
New Political Compass

Required install:
    pip3 install matplotlib pandas numpy scipy shapely

Run:
    python3 new_political_compass.py

Edit the CSV_PATH or CSV_CATS values below if a CSV moves.
Expected CSV columns for all point/category files:
    name,x,y,group,hex

Required: name, x, y
Optional: group, hex
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypedDict
import math
import sys
import tkinter as tk
from tkinter import messagebox

import numpy as np
import pandas as pd

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import Event, MouseEvent
from matplotlib.figure import Figure
from matplotlib.patches import Circle as MplCircle, Polygon as MplPolygon, Wedge as MplWedge
from matplotlib.colors import to_rgba
from matplotlib.collections import PolyCollection

try:
    from scipy.spatial import Voronoi, QhullError, cKDTree
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
CSV_PATH2 = str(Path(__file__).with_name("ideology_coordinates2.csv"))
CSV_PATH3 = str(Path(__file__).with_name("ideology_coordinates3.csv"))
CSV_PATH4 = str(Path(__file__).with_name("ideology_coordinates4.csv"))
CSV_PATH5 = str(Path(__file__).with_name("ideology_coordinates5.csv"))
CSV_PATH6 = str(Path(__file__).with_name("ideology_coordinates6.csv"))
CSV_PATH7 = str(Path(__file__).with_name("ideology_coordinates7.csv"))
CSV_PATH8 = str(Path(__file__).with_name("ideology_coordinates8.csv"))
CSV_PATH9 = str(Path(__file__).with_name("ideology_coordinates9.csv"))
CSV_PATH10 = str(Path(__file__).with_name("ideology_coordinates10.csv"))
CSV_CATS = str(Path(__file__).with_name("ideology_categories.csv"))
CSV_CATS2 = str(Path(__file__).with_name("ideology_categories2.csv"))
CSV_CATS3 = str(Path(__file__).with_name("ideology_categories3.csv"))
CSV_CATS4 = str(Path(__file__).with_name("ideology_categories4.csv"))
CSV_CATS5 = str(Path(__file__).with_name("ideology_categories5.csv"))
CSV_CATS6 = str(Path(__file__).with_name("ideology_categories6.csv"))
CSV_CATS7 = str(Path(__file__).with_name("ideology_categories7.csv"))
CSV_CATS8 = str(Path(__file__).with_name("ideology_categories8.csv"))
CSV_CATS9 = str(Path(__file__).with_name("ideology_categories9.csv"))
CSV_CATS10 = str(Path(__file__).with_name("ideology_categories10.csv"))

APP_TITLE = "New Political Compass"
GRAPH_MIN = -11.0
GRAPH_MAX = 11.0
COMPASS_MIN = -10.0
COMPASS_MAX = 10.0
WORLD_MIN = GRAPH_MIN
WORLD_MAX = GRAPH_MAX
WORLD_SPAN = WORLD_MAX - WORLD_MIN
MIN_VIEW_SPAN = 0.75
SEARCH_FOCUS_SPAN = 12.5
OVERTON_MAX_RADIUS = 5.0
OVERTON_FOCUS_MARGIN = 1.0
MAX_SEARCH_RESULTS = 15
CLICK_DISTANCE_PIXELS = 16
PAN_CLICK_TOLERANCE_PIXELS = 5
SCROLL_ZOOM_IN_SCALE = 0.90
SCROLL_ZOOM_OUT_SCALE = 1.0 / SCROLL_ZOOM_IN_SCALE
LABEL_EDGE_MARGIN_PIXELS = 2.0
LABEL_PADDING_PIXELS = 2.5
LABEL_OFFSET_PIXELS = 4.0
CATEGORY_LABEL_OFFSET_PIXELS = 5.0
LABEL_ZOOMED_OUT_FRACTION = 0.32
LABEL_ZOOM_CURVE = 1.15
LABEL_MIN_VISIBLE = 28
LABEL_WIDTH_FACTOR = 0.82
LABEL_HEIGHT_FACTOR = 1.15
LABEL_PRIORITY_FRACTION = 0.45
LABEL_OPEN_SPACE_MIN_PIXELS = 48.0
MANDATORY_LABEL_VORONOI_AREA = 4.0


# -----------------------------------------------------------------------------
# Visual settings
# -----------------------------------------------------------------------------

BACKGROUND = "#101518"
PANEL_BACKGROUND = "#182024"
GRID_MAJOR = "#314044"
GRID_MINOR = "#202c30"
AXIS_COLOR = "#b7b7b7"
OUTER_BAND = "#050505"
TEXT_COLOR = "#fbfbfb"
MUTED_TEXT = "#a8a8a8"
POINT_COLOR = "#f4f4f4"
POINT_EDGE = "#111111"
SELECTED_COLOR = "#fff176"
SELECTED_RING = "#ff4081"
VORONOI_EDGE = "#9be7ff"
VORONOI_FILL = "#4fc3f7"
VORONOI_FILL_ALPHA = 0.055
VORONOI_EDGE_ALPHA = 0.42
CATEGORY_EDGE = "#39c462"
CATEGORY_FILL = "#27c35a"
CATEGORY_FILL_ALPHA = 0.035
CATEGORY_EDGE_ALPHA = 0.56
CATEGORY_COLORFUL_FILL_ALPHA = 0.3
CATEGORY_COLORFUL_EDGE = "#f2f2f2"
CATEGORY_POINT = "#9ff6b4"
CATEGORY_TEXT = "#8fdca3"
CATEGORY_DARK_TEXT = "#d8d8d8"
TOOLTIP_BACKGROUND = "#222222"
TOOLTIP_EDGE = "#eeeeee"

# Colors apply to ideological proximity, not measured public approval.
OVERTON_BANDS = (
    ("Policy", "r < 1", "#39d2c0"),
    ("Popular", "1 <= r < 2", "#8bd671"),
    ("Sensible", "2 <= r < 3", "#66b7ed"),
    ("Acceptable", "3 <= r < 4", "#edc36b"),
    ("Radical", "4 <= r <= 5", "#f49375"),
    ("Unthinkable", "r > 5", "#aaaaaa"),
)


def parse_coordinate_query(query: str) -> Optional[Tuple[float, float]]:
    """Accept 'x,y' or '(x, y)', without treating ideology names as coordinates."""
    raw = query.strip()
    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1].strip()
    parts = raw.split(",")
    if len(parts) != 2:
        return None
    try:
        x, y = (float(value.strip()) for value in parts)
    except ValueError:
        return None
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def overton_band_index(distance: float) -> int:
    """Map nonnegative radius to a band, including exact boundary values."""
    if not math.isfinite(distance) or distance < 0:
        raise ValueError("Distance must be a finite, nonnegative number")
    if distance > OVERTON_MAX_RADIUS:
        return 5
    if distance == OVERTON_MAX_RADIUS:
        return 4
    return min(int(distance), 4)



@dataclass(frozen=True)
class CompassPoint:
    """A single plotted point from the CSV."""

    name: str
    x: float
    y: float
    group: Optional[str]
    hex_color: Optional[str]
    row_number: int

def overton_counts(
    center: Tuple[float, float], points: Sequence[CompassPoint]
) -> List[int]:
    """Count ideologies in each distance band, not people or popularity."""
    counts = [0] * len(OVERTON_BANDS)
    for point in points:
        distance = math.hypot(point.x - center[0], point.y - center[1])
        counts[overton_band_index(distance)] += 1
    return counts


@dataclass(frozen=True)
class CompassDataset:
    """A matched point/category CSV pair."""

    label: str
    points: Sequence[CompassPoint]
    warnings: Sequence[str]
    category_points: Sequence[CompassPoint]
    category_warnings: Sequence[str]


@dataclass(frozen=True)
class LabelBox:
    """A label rectangle in canvas pixels."""

    left: float
    bottom: float
    right: float
    top: float


@dataclass(frozen=True)
class LabelPlacement:
    """A label position plus its collision rectangle."""

    x: float
    y: float
    ha: str
    va: str
    fontsize: float
    box: LabelBox


class DataLoadError(Exception):
    """Raised when the CSV cannot be loaded into valid graph data."""


class DragStart(TypedDict):
    """Mouse-drag state captured when pan begins."""

    x_pixel: float
    y_pixel: float
    xlim: Tuple[float, float]
    ylim: Tuple[float, float]
    bbox_width: float
    bbox_height: float


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


def _parse_csv_float(value: object) -> float:
    """Parse a CSV scalar into a float while keeping type checkers honest."""
    return float(str(value).strip())


def _parse_hex_color(value: object) -> str:
    color = str(value).strip()
    hex_digits = "0123456789abcdefABCDEF"
    if (
        color.startswith("#")
        and len(color) in (4, 7)
        and all(char in hex_digits for char in color[1:])
    ):
        return color.upper()
    raise ValueError(f"{color!r} is not a valid #RGB or #RRGGBB color.")


def is_in_graph_bounds(x: float, y: float) -> bool:
    return WORLD_MIN <= x <= WORLD_MAX and WORLD_MIN <= y <= WORLD_MAX


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

    for row_number, (_, row) in enumerate(df.iterrows(), start=2):

        if all(_is_blank(row.get(col)) for col in df.columns):
            continue

        raw_x: object = row["x"]
        raw_y: object = row["y"]

        try:
            x = _parse_csv_float(raw_x)
            y = _parse_csv_float(raw_y)
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

        raw_name = row.get("name")
        if _is_blank(raw_name):
            name = f"Point {fallback_counter}"
            fallback_counter += 1
        else:
            name = str(raw_name).strip()

        group: Optional[str] = None
        if "group" in df.columns and not _is_blank(row.get("group")):
            group = str(row.get("group")).strip()

        hex_color: Optional[str] = None
        if "hex" in df.columns and not _is_blank(row.get("hex")):
            try:
                hex_color = _parse_hex_color(row.get("hex"))
            except ValueError as exc:
                warnings.append(f"Row {row_number}: ignored invalid hex color: {exc}")

        points.append(
            CompassPoint(
                name=name,
                x=x,
                y=y,
                group=group,
                hex_color=hex_color,
                row_number=row_number,
            )
        )

    if not points:
        raise DataLoadError("No valid finite points were found in the CSV.")

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


def polygon_area(coords: Sequence[Tuple[float, float]]) -> float:
    """Return polygon area in data-square units."""
    if len(coords) < 3:
        return 0.0

    area = 0.0
    for index, (x1, y1) in enumerate(coords):
        x2, y2 = coords[(index + 1) % len(coords)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def build_clipped_voronoi_cells(
    coords: np.ndarray,
    clip_min: float = WORLD_MIN,
    clip_max: float = WORLD_MAX,
) -> Tuple[List[Optional[List[Tuple[float, float]]]], Optional[str]]:
    """Build Voronoi polygons clipped to the graph square."""
    if len(coords) < 3:
        return [None for _ in coords], "At least 3 graph-bounds points are required to build a Voronoi diagram."

    unique_coords = {(round(float(x), 12), round(float(y), 12)) for x, y in coords}
    if len(unique_coords) < 3:
        return (
            [None for _ in coords],
            "At least 3 unique graph-bounds point locations are required to build a Voronoi diagram.",
        )

    bounds = box(clip_min, clip_min, clip_max, clip_max)

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
    def __init__(
        self,
        root: tk.Tk,
        datasets: Sequence[CompassDataset],
    ):
        if not datasets:
            raise ValueError("At least one dataset is required.")

        self.root = root
        self.datasets = list(datasets)
        self.active_dataset_index = len(self.datasets) - 1
        self.warning_notified_dataset_indices: set[int] = set()
        self.ideology_name_frequency = self._build_dataset_name_frequency(
            dataset.points for dataset in self.datasets
        )
        self.category_name_frequency = self._build_dataset_name_frequency(
            dataset.category_points for dataset in self.datasets
        )

        self.points: List[CompassPoint] = []
        self.warnings: List[str] = []
        self.category_points: List[CompassPoint] = []
        self.category_warnings: List[str] = []
        self.coord_to_point_indices: Dict[Tuple[float, float], List[int]] = {}
        self.category_coord_keys: set[Tuple[float, float]] = set()
        self.unique_coords = np.empty((0, 2), dtype=float)
        self.unique_coord_to_point_index: List[int] = []
        self.voronoi_cells: List[Optional[List[Tuple[float, float]]]] = []
        self.voronoi_warning: Optional[str] = None
        self.category_unique_coords = np.empty((0, 2), dtype=float)
        self.category_unique_coord_to_point_index: List[int] = []
        self.category_voronoi_cells: List[Optional[List[Tuple[float, float]]]] = []
        self.category_voronoi_warning: Optional[str] = None
        self.view_min_x = WORLD_MIN
        self.view_max_x = WORLD_MAX
        self.view_min_y = WORLD_MIN
        self.view_max_y = WORLD_MAX

        self.selected_index: Optional[int] = None
        self.custom_center: Optional[Tuple[float, float]] = None
        self.listbox_indices: List[int] = []
        self.drag_start: Optional[DragStart] = None
        self.is_dragging = False
        # A pin is independent of the currently selected detail level. Keep its
        # location even if that ideology disappears at another level.
        self.pinned_center: Optional[Tuple[float, float]] = None
        self.pinned_label: Optional[str] = None
        self.geometry_cache: dict[Tuple[int, bool], tuple] = {}
        self._navigation_after_id: Optional[str] = None
        self._resize_after_id: Optional[str] = None
        self._detail_after_id: Optional[str] = None
        self._pending_detail_index: Optional[int] = None
        self._details_visible = False
        self._pan_image_item: Optional[int] = None
        self._pan_image_origin: Optional[Tuple[float, float]] = None

        self.root.title(APP_TITLE)
        self.root.configure(bg=BACKGROUND)
        self.root.geometry("1360x900")
        self.root.minsize(900, 590)

        self.search_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.search_feedback_var = tk.StringVar(value="Type an ideology or coordinates (x, y).")
        self.center_info_var = tk.StringVar(value="Select an ideology or enter x,y to show proximity rings.")
        self.band_count_vars = [tk.StringVar(value="—") for _ in OVERTON_BANDS]
        self.show_overton_var = tk.BooleanVar(value=True)
        self.detail_var = tk.IntVar(value=len(self.datasets))
        self.show_extremes_var = tk.BooleanVar(value=False)
        self.colorful_categories_var = tk.BooleanVar(value=False)

        self._apply_dataset(self.active_dataset_index)
        self._build_ui()
        self._connect_events()
        self._update_status()
        self._update_detail_slider()
        self.draw_plot()
        self._notify_active_dataset_warnings()

    # ------------------------------------------------------------------
    # Data organization
    # ------------------------------------------------------------------

    def _normalize_label_key(self, text: str) -> str:
        return " ".join(text.strip().lower().split())

    def _build_dataset_name_frequency(
        self, point_groups: Iterable[Sequence[CompassPoint]]
    ) -> Dict[str, int]:
        frequencies: Dict[str, int] = {}
        for points in point_groups:
            keys = {self._normalize_label_key(point.name) for point in points}
            for key in keys:
                frequencies[key] = frequencies.get(key, 0) + 1
        return frequencies

    def _point_coord_key(self, point: CompassPoint) -> Tuple[float, float]:
        return round(point.x, 12), round(point.y, 12)

    def _current_graph_bounds(self) -> Tuple[float, float]:
        if self.show_extremes_var.get():
            return WORLD_MIN, WORLD_MAX
        return COMPASS_MIN, COMPASS_MAX

    def _point_is_in_current_graph_bounds(self, point: CompassPoint) -> bool:
        graph_min, graph_max = self._current_graph_bounds()
        return graph_min <= point.x <= graph_max and graph_min <= point.y <= graph_max

    def _build_coord_index(self) -> Dict[Tuple[float, float], List[int]]:
        coord_index: Dict[Tuple[float, float], List[int]] = {}
        for index, point in enumerate(self.points):
            key = self._point_coord_key(point)
            coord_index.setdefault(key, []).append(index)
        return coord_index

    def _build_unique_coords(
        self, points: Sequence[CompassPoint],
        in_bounds_only: bool = False,
    ) -> Tuple[np.ndarray, List[int]]:
        coords: List[Tuple[float, float]] = []
        representative_indices: List[int] = []
        seen: set[Tuple[float, float]] = set()

        for index, point in enumerate(points):
            if in_bounds_only and not self._point_is_in_current_graph_bounds(point):
                continue

            key = self._point_coord_key(point)
            if key in seen:
                continue
            seen.add(key)
            coords.append((point.x, point.y))
            representative_indices.append(index)

        return np.asarray(coords, dtype=float), representative_indices

    def _update_view_bounds(self) -> None:
        graph_min, graph_max = self._current_graph_bounds()
        graph_points = [
            point
            for point in [*self.points, *self.category_points]
            if self._point_is_in_current_graph_bounds(point)
        ]
        x_values = [graph_min, graph_max, *[point.x for point in graph_points]]
        y_values = [graph_min, graph_max, *[point.y for point in graph_points]]

        min_x = min(x_values)
        max_x = max(x_values)
        min_y = min(y_values)
        max_y = max(y_values)

        x_padding = 0.0 if (min_x == graph_min and max_x == graph_max) else 0.25
        y_padding = 0.0 if (min_y == graph_min and max_y == graph_max) else 0.25

        min_x -= x_padding
        max_x += x_padding
        min_y -= y_padding
        max_y += y_padding

        span = max(max_x - min_x, max_y - min_y)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        self.view_min_x = center_x - span / 2
        self.view_max_x = center_x + span / 2
        self.view_min_y = center_y - span / 2
        self.view_max_y = center_y + span / 2

    def _apply_dataset(self, dataset_index: int) -> None:
        dataset = self.datasets[dataset_index]
        self.active_dataset_index = dataset_index
        self.points = list(dataset.points)
        self.warnings = list(dataset.warnings)
        self.category_points = list(dataset.category_points)
        self.category_warnings = list(dataset.category_warnings)

        self.coord_to_point_indices = self._build_coord_index()
        self.category_coord_keys = {self._point_coord_key(point) for point in self.category_points}
        self._update_view_bounds()
        cache_key = (dataset_index, self.show_extremes_var.get())
        if cache_key not in self.geometry_cache:
            graph_min, graph_max = self._current_graph_bounds()
            unique_coords, unique_indices = self._build_unique_coords(
                self.points, in_bounds_only=True
            )
            cells, warning = build_clipped_voronoi_cells(unique_coords, graph_min, graph_max)
            cat_coords, cat_indices = self._build_unique_coords(
                self.category_points, in_bounds_only=True
            )
            cat_cells, cat_warning = build_clipped_voronoi_cells(cat_coords, graph_min, graph_max)
            self.geometry_cache[cache_key] = (
                unique_coords, unique_indices, cells, warning,
                cat_coords, cat_indices, cat_cells, cat_warning,
            )
        (
            self.unique_coords, self.unique_coord_to_point_index,
            self.voronoi_cells, self.voronoi_warning,
            self.category_unique_coords, self.category_unique_coord_to_point_index,
            self.category_voronoi_cells, self.category_voronoi_warning,
        ) = self.geometry_cache[cache_key]
        self._resolve_pinned_selection()

    def _resolve_pinned_selection(self) -> None:
        """Keep the map pin fixed, matching a current data point only if identical."""
        self.selected_index = None
        self.custom_center = self.pinned_center
        if self.pinned_center is None or not self.pinned_label:
            return
        wanted = self._normalize_label_key(self.pinned_label)
        x, y = self.pinned_center
        for i, point in enumerate(self.points):
            if (self._normalize_label_key(point.name) == wanted
                    and math.isclose(point.x, x, abs_tol=1e-8)
                    and math.isclose(point.y, y, abs_tol=1e-8)):
                self.selected_index = i
                self.custom_center = None
                return

    def _notify_active_dataset_warnings(self) -> None:
        if self.active_dataset_index in self.warning_notified_dataset_indices:
            return

        dataset_label = self.datasets[self.active_dataset_index].label
        self.warning_notified_dataset_indices.add(self.active_dataset_index)

        if self.warnings:
            print(f"\n{dataset_label} CSV warnings:")
            for warning in self.warnings:
                print("-", warning)
            print()

        if self.category_warnings:
            print(f"\n{dataset_label} category CSV warnings:")
            for warning in self.category_warnings:
                print("-", warning)
            print()

        if self.voronoi_warning:
            print(f"{dataset_label} Voronoi warning:", self.voronoi_warning)
            messagebox.showwarning(
                APP_TITLE,
                f"{dataset_label} points loaded, but the Voronoi diagram could not be built.\n\n"
                + self.voronoi_warning,
            )

        if self.category_voronoi_warning:
            print(f"{dataset_label} category Voronoi warning:", self.category_voronoi_warning)
            messagebox.showwarning(
                APP_TITLE,
                f"{dataset_label} category points loaded, but the category Voronoi diagram could not be built.\n\n"
                + self.category_voronoi_warning,
            )

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, bg=PANEL_BACKGROUND, padx=14, pady=7)
        header.pack(side=tk.TOP, fill=tk.X)
        top_frame = tk.Frame(header, bg=PANEL_BACKGROUND)
        top_frame.pack(fill=tk.X)
        search_frame = tk.Frame(header, bg=PANEL_BACKGROUND)
        search_frame.pack(fill=tk.X, pady=(5, 0))

        tk.Label(
            top_frame, text=APP_TITLE, bg=PANEL_BACKGROUND, fg=TEXT_COLOR,
            font=("Helvetica Neue", 15, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 17))

        detail_frame = tk.Frame(top_frame, bg=PANEL_BACKGROUND)
        detail_frame.pack(side=tk.LEFT, padx=(0, 16))
        tk.Label(
            detail_frame, text="Detail", bg=PANEL_BACKGROUND,
            fg=MUTED_TEXT, font=("Helvetica Neue", 10),
        ).pack(side=tk.LEFT, padx=(0, 7))
        detail_slider_frame = tk.Frame(detail_frame, bg=PANEL_BACKGROUND)
        detail_slider_frame.pack(side=tk.LEFT)
        self.detail_scale = tk.Scale(
            detail_slider_frame, from_=len(self.datasets), to=1, orient=tk.HORIZONTAL,
            variable=self.detail_var, command=self._on_detail_slider_changed,
            bg=PANEL_BACKGROUND, fg=TEXT_COLOR, activebackground=SELECTED_RING,
            troughcolor="#344044", highlightthickness=0, relief=tk.FLAT,
            showvalue=False, resolution=1, length=165,
            sliderlength=15, bd=0, font=("Helvetica Neue", 10),
        )
        self.detail_scale.pack(side=tk.TOP, fill=tk.X)
        ticks = tk.Frame(detail_slider_frame, bg=PANEL_BACKGROUND)
        ticks.pack(side=tk.TOP, fill=tk.X)
        for index in reversed(range(len(self.datasets))):
            tk.Label(
                ticks, text=str(index + 1), bg=PANEL_BACKGROUND,
                fg=MUTED_TEXT, font=("Helvetica Neue", 8),
            ).pack(side=tk.LEFT, expand=True)
        self.detail_scale.bind("<ButtonRelease-1>", self._flush_detail_switch)

        for title, var, callback in (
            ("Extremes", self.show_extremes_var, self._on_extremes_toggled),
            ("Colorful", self.colorful_categories_var, self._on_colorful_categories_toggled),
            ("Proximity rings", self.show_overton_var, self._on_overton_toggled),
        ):
            tk.Checkbutton(
                top_frame, text=title, variable=var, command=callback,
                bg=PANEL_BACKGROUND, fg=TEXT_COLOR, activebackground=PANEL_BACKGROUND,
                activeforeground=TEXT_COLOR, selectcolor="#253237", relief=tk.FLAT,
                highlightthickness=0, font=("Helvetica Neue", 10),
            ).pack(side=tk.LEFT, padx=(0, 11))

        self.details_button = tk.Button(
            top_frame, text="Details  ▸", command=self._toggle_details,
            bg="#28373d", fg=TEXT_COLOR, activebackground="#3b5057",
            activeforeground=TEXT_COLOR, relief=tk.FLAT,
            font=("Helvetica Neue", 10, "bold"), padx=12, pady=3,
        )
        self.details_button.pack(side=tk.RIGHT, padx=(10, 0))

        tk.Label(
            search_frame, text="Find ideology / x,y", bg=PANEL_BACKGROUND,
            fg=TEXT_COLOR, font=("Helvetica Neue", 11),
        ).pack(side=tk.LEFT, padx=(0, 8))
        self.search_entry = tk.Entry(
            search_frame, textvariable=self.search_var, width=29,
            bg="#263238", fg=TEXT_COLOR, insertbackground=TEXT_COLOR,
            relief=tk.FLAT, font=("Helvetica Neue", 11),
            highlightthickness=1, highlightbackground="#394a50",
            highlightcolor="#69bed0",
        )
        self.search_entry.pack(side=tk.LEFT, padx=(0, 8), ipady=4)
        self.go_button = tk.Button(
            search_frame, text="Go / Select", command=self._on_search_enter,
            bg="#85cced", fg="#111111", activebackground="#bbecff",
            relief=tk.FLAT, font=("Helvetica Neue", 10, "bold"), padx=12, pady=3,
        )
        self.go_button.pack(side=tk.LEFT, padx=(0, 7))
        self.clear_button = tk.Button(
            search_frame, text="Clear", command=self.clear_selection,
            bg="#36464d", fg=TEXT_COLOR, activebackground="#50626a",
            activeforeground=TEXT_COLOR, relief=tk.FLAT,
            font=("Helvetica Neue", 10), padx=11, pady=3,
        )
        self.clear_button.pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(
            search_frame, textvariable=self.status_var, bg=PANEL_BACKGROUND,
            fg=MUTED_TEXT, font=("Helvetica Neue", 10), anchor="w",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Search suggestions sit below the search bar, only while needed.
        # The graph therefore doesn't permanently sacrifice 250+ px of width.
        self.matches_frame = tk.Frame(header, bg=PANEL_BACKGROUND)
        self.matches_title_var = tk.StringVar()
        tk.Label(
            self.matches_frame, textvariable=self.matches_title_var,
            bg=PANEL_BACKGROUND, fg=MUTED_TEXT, font=("Helvetica Neue", 9),
            anchor="w",
        ).pack(anchor="w", pady=(1, 3))
        self.results_listbox = tk.Listbox(
            self.matches_frame, width=63, height=5, bg="#223037", fg=TEXT_COLOR,
            selectbackground="#3a5962", selectforeground=SELECTED_COLOR,
            highlightthickness=1, highlightbackground="#35494f", relief=tk.FLAT,
            activestyle="none", exportselection=False,
            font=("Helvetica Neue", 10),
        )
        self.results_listbox.pack(anchor="w", fill=tk.X)

        body = tk.Frame(self.root, bg=BACKGROUND)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.results_frame = tk.Frame(body, bg=PANEL_BACKGROUND, padx=13, pady=14, width=225)
        self.results_frame.pack_propagate(False)
        tk.Label(
            self.results_frame, text="SELECTED POINT", bg=PANEL_BACKGROUND,
            fg=MUTED_TEXT, font=("Helvetica Neue", 10, "bold"),
        ).pack(anchor="w", pady=(0, 6))
        tk.Label(
            self.results_frame, textvariable=self.center_info_var,
            bg=PANEL_BACKGROUND, fg=TEXT_COLOR, font=("Helvetica Neue", 11),
            justify=tk.LEFT, wraplength=197,
        ).pack(anchor="w", pady=(0, 15))
        tk.Frame(self.results_frame, bg="#35444b", height=1).pack(fill=tk.X, pady=(0, 12))
        tk.Label(
            self.results_frame, text="PROXIMITY BANDS", bg=PANEL_BACKGROUND,
            fg=MUTED_TEXT, font=("Helvetica Neue", 10, "bold"),
        ).pack(anchor="w", pady=(0, 8))
        for index, (name, rule, color) in enumerate(OVERTON_BANDS):
            band_row = tk.Frame(self.results_frame, bg=PANEL_BACKGROUND)
            band_row.pack(fill=tk.X, pady=4)
            tk.Label(
                band_row, text="●", bg=PANEL_BACKGROUND, fg=color,
                font=("Helvetica Neue", 13),
            ).pack(side=tk.LEFT)
            tk.Label(
                band_row, text=name, bg=PANEL_BACKGROUND, fg=TEXT_COLOR,
                font=("Helvetica Neue", 10), anchor="w",
            ).pack(side=tk.LEFT, padx=(5, 3))
            tk.Label(
                band_row, textvariable=self.band_count_vars[index],
                bg=PANEL_BACKGROUND, fg=TEXT_COLOR,
                font=("Helvetica Neue", 10, "bold"), anchor="e",
            ).pack(side=tk.RIGHT)
            tk.Label(
                band_row, text=rule, bg=PANEL_BACKGROUND, fg=MUTED_TEXT,
                font=("Helvetica Neue", 8), anchor="e",
            ).pack(side=tk.RIGHT, padx=(0, 7))
        tk.Label(
            self.results_frame,
            text="Counts are plotted ideologies—not public opinion or policy viability.",
            bg=PANEL_BACKGROUND, fg=MUTED_TEXT, wraplength=195,
            justify=tk.LEFT, font=("Helvetica Neue", 9),
        ).pack(anchor="w", pady=(17, 6))
        tk.Label(
            self.results_frame,
            text="Scroll to zoom · Drag to pan\nClick a dot to select · Esc to clear",
            bg=PANEL_BACKGROUND, fg=MUTED_TEXT, wraplength=195,
            justify=tk.LEFT, font=("Helvetica Neue", 9),
        ).pack(anchor="w", pady=(12, 0))

        fig = Figure(figsize=(10, 9), dpi=100, facecolor=BACKGROUND)
        self.fig = fig
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, master=body)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.configure(bg=BACKGROUND, highlightthickness=0)
        self.canvas_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.footer_frame = tk.Frame(self.root, bg=PANEL_BACKGROUND, padx=15, pady=4)
        self.footer_frame.pack(side=tk.BOTTOM, fill=tk.X)
        # Pack the expanding graph last; otherwise Tk can clip the footer.
        body.pack_forget()
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        tk.Label(
            self.footer_frame, textvariable=self.search_feedback_var,
            bg=PANEL_BACKGROUND, fg=MUTED_TEXT,
            font=("Helvetica Neue", 9), anchor="w",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(
            self.footer_frame, text="Scroll: zoom   •   Drag: pan   •   Esc: clear",
            bg=PANEL_BACKGROUND, fg=MUTED_TEXT,
            font=("Helvetica Neue", 9), anchor="e",
        ).pack(side=tk.RIGHT)

        self.search_var.trace_add("write", self._on_search_changed)
        self.results_listbox.bind("<<ListboxSelect>>", self._on_result_selected)
        self.search_entry.bind("<Return>", self._on_search_enter)
        self.search_entry.bind("<Escape>", self._on_escape)
        self.root.bind("<Escape>", self._on_escape)

    def _toggle_details(self) -> None:
        self._details_visible = not self._details_visible
        if self._details_visible:
            # The sidebar must be packed before the expanding Matplotlib canvas
            # or the canvas claims the sidebar's horizontal space.
            self.canvas_widget.pack_forget()
            self.results_frame.pack(side=tk.RIGHT, fill=tk.Y)
            self.canvas_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.details_button.config(text="Details  ◂")
        else:
            self.results_frame.pack_forget()
            self.details_button.config(text="Details  ▸")
        self._queue_resize_relayout()

    def _queue_resize_relayout(self, _event: object = None) -> None:
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        # Tk/Matplotlib update canvas dimensions asynchronously. Place labels
        # only after the new axes bbox has stabilized.
        self._resize_after_id = self.root.after(120, self._finish_resize_relayout)

    def _finish_resize_relayout(self) -> None:
        self._resize_after_id = None
        if not self.is_dragging:
            self.draw_plot(keep_view=True)

    def _set_matches_visible(self, visible: bool) -> None:
        if visible and not self.matches_frame.winfo_manager():
            self.matches_frame.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        elif not visible and self.matches_frame.winfo_manager():
            self.matches_frame.pack_forget()

    def _on_detail_slider_changed(self, value: str) -> None:
        # Coalesce slider scrubbing: a Voronoi redraw is needed only for the
        # most recently requested level, not every intermediate mouse event.
        self._pending_detail_index = int(round(float(value))) - 1
        if self._detail_after_id is not None:
            self.root.after_cancel(self._detail_after_id)
        self._detail_after_id = self.root.after(85, self._flush_detail_switch)

    def _flush_detail_switch(self, _event: object = None) -> None:
        if self._detail_after_id is not None:
            self.root.after_cancel(self._detail_after_id)
            self._detail_after_id = None
        if self._pending_detail_index is not None:
            index = self._pending_detail_index
            self._pending_detail_index = None
            self.switch_dataset(index)

    def switch_dataset(self, dataset_index: int) -> None:
        if dataset_index == self.active_dataset_index:
            return
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        # _apply_dataset will resolve the existing pin into the new data if
        # the exact point exists, otherwise keep it as a fixed coordinate pin.
        self._apply_dataset(dataset_index)
        self.ax.set_xlim(*self._clamp_x_limits(*map(float, current_xlim)))
        self.ax.set_ylim(*self._clamp_y_limits(*map(float, current_ylim)))
        self._on_search_changed(show_matches=False)
        self._update_overton_panel()
        self._update_status()
        self._update_detail_slider()
        self.draw_plot(keep_view=True)
        self._notify_active_dataset_warnings()

    def _update_detail_slider(self) -> None:
        self.detail_var.set(self.active_dataset_index + 1)

    def _on_extremes_toggled(self) -> None:
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        self._apply_dataset(self.active_dataset_index)
        self.ax.set_xlim(*self._clamp_x_limits(*map(float, current_xlim)))
        self.ax.set_ylim(*self._clamp_y_limits(*map(float, current_ylim)))
        self._on_search_changed(show_matches=False)
        self._update_overton_panel()
        self._update_status()
        self.draw_plot(keep_view=True)
        self._notify_active_dataset_warnings()

    def _on_colorful_categories_toggled(self) -> None:
        self.draw_plot(keep_view=True)

    def _on_overton_toggled(self) -> None:
        self.draw_plot(keep_view=True)

    def _connect_events(self) -> None:
        self.canvas.mpl_connect("resize_event", self._queue_resize_relayout)
        self.canvas.mpl_connect("scroll_event", self._handle_scroll_event)
        self.canvas.mpl_connect("button_press_event", self._handle_mouse_press_event)
        self.canvas.mpl_connect("motion_notify_event", self._handle_mouse_move_event)
        self.canvas.mpl_connect("button_release_event", self._handle_mouse_release_event)

    def _handle_scroll_event(self, event: Event) -> None:
        if isinstance(event, MouseEvent):
            self._on_scroll(event)

    def _handle_mouse_press_event(self, event: Event) -> None:
        if isinstance(event, MouseEvent):
            self._on_mouse_press(event)

    def _handle_mouse_move_event(self, event: Event) -> None:
        if isinstance(event, MouseEvent):
            self._on_mouse_move(event)

    def _handle_mouse_release_event(self, event: Event) -> None:
        if isinstance(event, MouseEvent):
            self._on_mouse_release(event)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _cancel_navigation_refine(self) -> None:
        if self._navigation_after_id is not None:
            self.root.after_cancel(self._navigation_after_id)
            self._navigation_after_id = None

    def _refine_navigation(self) -> None:
        self._navigation_after_id = None
        if not self.is_dragging:
            self.draw_plot(keep_view=True)

    def _preview_navigation(self) -> None:
        # Pan/zoom only changes the axes' affine transform. Reuse all existing
        # collections, text and circles rather than reconstructing the graph.
        self.canvas.draw_idle()
        self._cancel_navigation_refine()
        self._navigation_after_id = self.root.after(170, self._refine_navigation)

    def _setup_axes(self) -> None:
        graph_min, graph_max = self._current_graph_bounds()
        self.ax.set_facecolor(BACKGROUND)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlim(self.view_min_x, self.view_max_x)
        self.ax.set_ylim(self.view_min_y, self.view_max_y)

        major_ticks = np.arange(COMPASS_MIN, COMPASS_MAX + 1, 2)
        minor_ticks = np.arange(graph_min, graph_max + 0.5, 1)
        self.ax.set_xticks(major_ticks)
        self.ax.set_yticks(major_ticks)
        self.ax.set_xticks(minor_ticks, minor=True)
        self.ax.set_yticks(minor_ticks, minor=True)

        if self.show_extremes_var.get():
            self.ax.axvspan(WORLD_MIN, COMPASS_MIN, color=OUTER_BAND, alpha=0.28, zorder=0.1)
            self.ax.axvspan(COMPASS_MAX, WORLD_MAX, color=OUTER_BAND, alpha=0.28, zorder=0.1)
            self.ax.axhspan(WORLD_MIN, COMPASS_MIN, color=OUTER_BAND, alpha=0.28, zorder=0.1)
            self.ax.axhspan(COMPASS_MAX, WORLD_MAX, color=OUTER_BAND, alpha=0.28, zorder=0.1)
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
        self.fig.subplots_adjust(left=0.055, right=0.985, top=0.992, bottom=0.058)

    def draw_plot(self, keep_view: bool = False) -> None:
        old_xlim = self.ax.get_xlim() if keep_view else (self.view_min_x, self.view_max_x)
        old_ylim = self.ax.get_ylim() if keep_view else (self.view_min_y, self.view_max_y)

        self._cancel_navigation_refine()
        self.ax.clear()
        self._setup_axes()
        self.ax.set_xlim(*old_xlim)
        self.ax.set_ylim(*old_ylim)

        self._draw_voronoi_cells()
        self._draw_overton_rings()
        visible_ideology_points = self._visible_ideology_points_by_priority()
        occupied_label_boxes, visible_category_points = self._draw_category_labels()
        self._draw_category_points(visible_category_points)
        visible_label_indices = self._draw_labels(visible_ideology_points, occupied_label_boxes)
        self._draw_points(visible_label_indices)
        self._draw_selected_point()

        self.canvas.draw_idle()

    def _active_center(self) -> Optional[Tuple[float, float]]:
        return self.pinned_center

    def _draw_overton_rings(self) -> None:
        center = self._active_center()
        if center is None or not self.show_overton_var.get():
            return

        # True annuli: paint each band once so its color doesn't mix with others.
        # The region beyond radius 5 remains the ordinary map background.
        for radius in range(5, 0, -1):
            _name, _rule, color = OVERTON_BANDS[radius - 1]
            self.ax.add_patch(MplWedge(
                center, r=radius, theta1=0, theta2=360, width=1,
                facecolor=to_rgba(color, 0.17),
                edgecolor="none",
                zorder=2.8,
                clip_on=True,
            ))
            self.ax.add_patch(MplCircle(
                center, radius=radius, fill=False,
                edgecolor=to_rgba(color, 0.95),
                linewidth=1.9,
                zorder=2.9,
                clip_on=True,
            ))

    def _update_overton_panel(self) -> None:
        center = self._active_center()
        if center is None:
            self.center_info_var.set("Select an ideology or enter x,y to show rings.")
            for variable in self.band_count_vars:
                variable.set("—")
            return
        label = self.pinned_label or "Custom coordinates"
        self.center_info_var.set(f"{label}\n({center[0]:g}, {center[1]:g})")
        candidates = [point for point in self.points if self._point_is_in_current_graph_bounds(point)]
        for variable, count in zip(self.band_count_vars, overton_counts(center, candidates)):
            variable.set(str(count))

    def _draw_voronoi_cells(self) -> None:
        self._draw_cell_patches(
            self.voronoi_cells,
            VORONOI_FILL,
            VORONOI_EDGE,
            1.25,
            VORONOI_FILL_ALPHA,
            VORONOI_EDGE_ALPHA,
            1.8,
        )
        if self.colorful_categories_var.get():
            self._draw_category_cell_patches()
        else:
            self._draw_cell_patches(
                self.category_voronoi_cells,
                CATEGORY_FILL,
                CATEGORY_EDGE,
                1.0,
                CATEGORY_FILL_ALPHA,
                CATEGORY_EDGE_ALPHA,
                2.1,
            )

        boundary = MplPolygon(
            [
                (COMPASS_MIN, COMPASS_MIN),
                (COMPASS_MAX, COMPASS_MIN),
                (COMPASS_MAX, COMPASS_MAX),
                (COMPASS_MIN, COMPASS_MAX),
            ],
            closed=True,
            fill=False,
            edgecolor="#d0d0d0",
            linewidth=2.3,
            alpha=0.9,
            zorder=4,
        )
        self.ax.add_patch(boundary)

    def _draw_cell_patches(
        self,
        cells: Sequence[Optional[List[Tuple[float, float]]]],
        fill_color: str,
        edge_color: str,
        linewidth: float,
        fill_alpha: float,
        edge_alpha: float,
        zorder: float,
    ) -> None:
        # One batched GPU/renderer artist instead of hundreds of Polygon artists.
        valid_cells = [cell for cell in cells if cell]
        if not valid_cells:
            return
        self.ax.add_collection(PolyCollection(
            valid_cells, closed=True,
            facecolors=to_rgba(fill_color, fill_alpha),
            edgecolors=to_rgba(edge_color, edge_alpha),
            linewidths=linewidth, zorder=zorder,
        ))

    def _draw_category_cell_patches(self) -> None:
        valid = [
            (cell, self.category_points[index])
            for cell, index in zip(
                self.category_voronoi_cells, self.category_unique_coord_to_point_index
            ) if cell
        ]
        if not valid:
            return
        self.ax.add_collection(PolyCollection(
            [cell for cell, point in valid], closed=True,
            facecolors=[to_rgba(self._category_color(point), CATEGORY_COLORFUL_FILL_ALPHA)
                        for cell, point in valid],
            edgecolors=to_rgba(CATEGORY_COLORFUL_EDGE, 0.92),
            linewidths=1.0, zorder=2.1,
        ))

    def _category_color(self, point: CompassPoint) -> str:
        return point.hex_color or CATEGORY_FILL

    def _color_is_dark(self, color: str) -> bool:
        red, green, blue, _alpha = to_rgba(color)
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        return luminance < 0.24

    def _category_text_color(self, point: CompassPoint) -> str:
        if not self.colorful_categories_var.get():
            return CATEGORY_TEXT

        color = self._category_color(point)
        if self._color_is_dark(color):
            return CATEGORY_DARK_TEXT
        return color

    def _visible_ideology_points_by_priority(self) -> List[Tuple[int, CompassPoint]]:
        visible_points = [
            (index, point)
            for index, point in enumerate(self.points)
            if self._point_is_visible(point)
            and self._point_coord_key(point) not in self.category_coord_keys
        ]
        visible_points.sort(key=lambda item: self._ideology_label_sort_key(item[0], item[1]))
        return visible_points

    def _draw_points(self, point_indices: set[int]) -> None:
        if self.selected_index is not None and self._point_is_visible(self.points[self.selected_index]):
            point_indices.add(self.selected_index)

        points = [self.points[index] for index in sorted(point_indices)]
        if not points:
            return

        x_values = [point.x for point in points]
        y_values = [point.y for point in points]
        self.ax.scatter(
            x_values,
            y_values,
            s=34,
            c=POINT_COLOR,
            edgecolors=POINT_EDGE,
            linewidths=0.75,
            zorder=5,
        )

    def _draw_category_points(self, points: Sequence[CompassPoint]) -> None:
        if not points:
            return

        x_values = [point.x for point in points]
        y_values = [point.y for point in points]
        face_colors: object = CATEGORY_POINT
        edge_colors: object = "#06230d"
        if self.colorful_categories_var.get():
            face_colors = [self._category_color(point) for point in points]
            edge_colors = CATEGORY_COLORFUL_EDGE
        self.ax.scatter(
            x_values,
            y_values,
            s=58,
            c=face_colors,
            edgecolors=edge_colors,
            linewidths=1.0,
            zorder=4.4,
        )

    def _point_is_visible(self, point: CompassPoint) -> bool:
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        return xlim[0] <= point.x <= xlim[1] and ylim[0] <= point.y <= ylim[1]

    def _label_boxes_overlap(self, first: LabelBox, second: LabelBox) -> bool:
        padding = LABEL_PADDING_PIXELS
        return not (
            first.right + padding <= second.left
            or first.left >= second.right + padding
            or first.top + padding <= second.bottom
            or first.bottom >= second.top + padding
        )

    def _label_limit(self, visible_count: int) -> int:
        if visible_count <= 12:
            return visible_count

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        current_span = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
        max_span = max(self.view_max_x - self.view_min_x, self.view_max_y - self.view_min_y)
        span_range = max(max_span - MIN_VIEW_SPAN, 1.0)
        zoom_progress = float(np.clip((max_span - current_span) / span_range, 0.0, 1.0))
        label_fraction = LABEL_ZOOMED_OUT_FRACTION + (
            1.0 - LABEL_ZOOMED_OUT_FRACTION
        ) * (zoom_progress**LABEL_ZOOM_CURVE)
        return min(visible_count, max(LABEL_MIN_VISIBLE, int(math.ceil(visible_count * label_fraction))))

    def _ideology_label_sort_key(self, index: int, point: CompassPoint) -> Tuple[int, int, int, str]:
        selected_rank = 0 if index == self.selected_index else 1
        frequency = self.ideology_name_frequency.get(self._normalize_label_key(point.name), 0)
        return selected_rank, -frequency, point.row_number, point.name.lower()

    def _category_label_sort_key(self, point: CompassPoint) -> Tuple[int, int, str]:
        frequency = self.category_name_frequency.get(self._normalize_label_key(point.name), 0)
        return -frequency, point.row_number, point.name.lower()

    def _category_voronoi_area_by_key(self) -> Dict[Tuple[float, float], float]:
        areas: Dict[Tuple[float, float], float] = {}

        for cell, point_index in zip(
            self.category_voronoi_cells, self.category_unique_coord_to_point_index
        ):
            if not cell:
                continue

            point = self.category_points[point_index]
            areas[self._point_coord_key(point)] = polygon_area(cell)

        return areas

    def _category_label_priority_key(
        self,
        point: CompassPoint,
        area_by_key: Dict[Tuple[float, float], float],
    ) -> Tuple[int, float, int, int, str]:
        area = area_by_key.get(self._point_coord_key(point), 0.0)
        mandatory_rank = 0 if area > MANDATORY_LABEL_VORONOI_AREA else 1
        frequency, row_number, name = self._category_label_sort_key(point)
        return mandatory_rank, -area if mandatory_rank == 0 else 0.0, frequency, row_number, name

    def _mandatory_voronoi_label_indices(self) -> Dict[int, float]:
        label_indices: Dict[int, float] = {}

        for cell, point_index in zip(self.voronoi_cells, self.unique_coord_to_point_index):
            if not cell:
                continue

            point = self.points[point_index]
            area = polygon_area(cell)
            if (
                area > MANDATORY_LABEL_VORONOI_AREA
                and self._point_is_visible(point)
                and self._point_coord_key(point) not in self.category_coord_keys
            ):
                label_indices[point_index] = area

        return label_indices

    def _point_screen_openness(
        self,
        point: CompassPoint,
        neighbor_points: Sequence[CompassPoint],
    ) -> float:
        point_x, point_y = self.ax.transData.transform((point.x, point.y))
        nearest_distance = math.inf

        for neighbor in neighbor_points:
            if neighbor is point:
                continue
            neighbor_x, neighbor_y = self.ax.transData.transform((neighbor.x, neighbor.y))
            distance = math.hypot(point_x - neighbor_x, point_y - neighbor_y)
            nearest_distance = min(nearest_distance, distance)

        return nearest_distance

    def _clamp_label_box(self, box: LabelBox, width: float, height: float) -> LabelBox:
        axes_left = float(self.ax.bbox.x0) + LABEL_EDGE_MARGIN_PIXELS
        axes_right = float(self.ax.bbox.x1) - LABEL_EDGE_MARGIN_PIXELS
        axes_bottom = float(self.ax.bbox.y0) + LABEL_EDGE_MARGIN_PIXELS
        axes_top = float(self.ax.bbox.y1) - LABEL_EDGE_MARGIN_PIXELS

        if width >= axes_right - axes_left:
            left = axes_left
        else:
            left = min(max(box.left, axes_left), axes_right - width)

        if height >= axes_top - axes_bottom:
            bottom = axes_bottom
        else:
            bottom = min(max(box.bottom, axes_bottom), axes_top - height)

        return LabelBox(left=left, bottom=bottom, right=left + width, top=bottom + height)

    def _label_placement(
        self,
        point: CompassPoint,
        text: str,
        fontsize: float,
        offset_pixels: float,
        occupied_boxes: Sequence[LabelBox],
    ) -> Optional[LabelPlacement]:
        if not self._point_is_visible(point):
            return None

        axes_width = max(float(self.ax.bbox.width) - 2 * LABEL_EDGE_MARGIN_PIXELS, 1.0)
        font_pixels = fontsize * self.fig.dpi / 72.0
        max_font_pixels = max(
            3.0, (axes_width - 8) / (max(len(text), 1) * LABEL_WIDTH_FACTOR)
        )
        font_pixels = min(font_pixels, max_font_pixels)
        fontsize = font_pixels * 72.0 / self.fig.dpi
        label_width = max(len(text), 1) * font_pixels * LABEL_WIDTH_FACTOR
        label_height = font_pixels * LABEL_HEIGHT_FACTOR

        point_x, point_y = self.ax.transData.transform((point.x, point.y))
        midpoint_x = (float(self.ax.bbox.x0) + float(self.ax.bbox.x1)) / 2
        midpoint_y = (float(self.ax.bbox.y0) + float(self.ax.bbox.y1)) / 2

        horizontal_order = ["left", "right"] if point_x <= midpoint_x else ["right", "left"]
        vertical_order = ["bottom", "top"] if point_y <= midpoint_y else ["top", "bottom"]
        orientations = [
            (horizontal_order[0], vertical_order[0]),
            (horizontal_order[1], vertical_order[0]),
            ("center", vertical_order[0]),
            (horizontal_order[0], vertical_order[1]),
            (horizontal_order[1], vertical_order[1]),
            ("center", vertical_order[1]),
        ]

        for ha, va in orientations:
            if ha == "left":
                left = point_x + offset_pixels
                right = left + label_width
            elif ha == "right":
                right = point_x - offset_pixels
                left = right - label_width
            else:
                left = point_x - label_width / 2
                right = point_x + label_width / 2

            if va == "bottom":
                bottom = point_y + offset_pixels
                top = bottom + label_height
            else:
                top = point_y - offset_pixels
                bottom = top - label_height

            box = self._clamp_label_box(
                LabelBox(left=left, bottom=bottom, right=right, top=top),
                label_width,
                label_height,
            )
            if any(self._label_boxes_overlap(box, occupied) for occupied in occupied_boxes):
                continue

            if ha == "left":
                anchor_x = box.left
            elif ha == "right":
                anchor_x = box.right
            else:
                anchor_x = (box.left + box.right) / 2
            anchor_y = box.bottom if va == "bottom" else box.top
            data_x, data_y = self.ax.transData.inverted().transform((anchor_x, anchor_y))
            return LabelPlacement(
                x=float(data_x),
                y=float(data_y),
                ha=ha,
                va=va,
                fontsize=fontsize,
                box=box,
            )

        return None

    def _draw_labels(
        self,
        visible_points: Sequence[Tuple[int, CompassPoint]],
        occupied_boxes: Sequence[LabelBox],
    ) -> set[int]:
        label_boxes = list(occupied_boxes)
        label_limit = self._label_limit(len(visible_points))
        drawn_indices: set[int] = set()
        mandatory_label_areas = self._mandatory_voronoi_label_indices()
        neighbor_points = [
            *[point for _, point in visible_points],
            *[point for point in self.category_points if self._point_is_visible(point)],
        ]

        def try_draw_label(index: int, point: CompassPoint, respect_limit: bool = True) -> bool:
            if (respect_limit and len(drawn_indices) >= label_limit) or index in drawn_indices:
                return False

            is_selected = index == self.selected_index
            fontsize = 9.5 if is_selected else 8.5
            placement = self._label_placement(
                point, point.name, fontsize, LABEL_OFFSET_PIXELS, label_boxes
            )
            if placement is None:
                return False

            self.ax.text(
                placement.x,
                placement.y,
                point.name,
                color=SELECTED_COLOR if is_selected else TEXT_COLOR,
                fontsize=placement.fontsize,
                fontweight="bold" if is_selected else "normal",
                alpha=1.0 if is_selected else 0.82,
                ha=placement.ha,
                va=placement.va,
                zorder=7 if is_selected else 6,
                clip_on=True,
            )
            label_boxes.append(placement.box)
            drawn_indices.add(index)
            return True

        mandatory_points = [
            (mandatory_label_areas[index], index, point)
            for index, point in visible_points
            if index in mandatory_label_areas
        ]
        mandatory_points.sort(
            key=lambda item: (-item[0], self._ideology_label_sort_key(item[1], item[2]))
        )

        for _area, index, point in mandatory_points:
            try_draw_label(index, point, respect_limit=False)

        priority_target = min(label_limit, max(1, int(math.ceil(label_limit * LABEL_PRIORITY_FRACTION))))
        for index, point in visible_points:
            if len(drawn_indices) >= priority_target:
                break
            try_draw_label(index, point)

        # Pixel-space nearest neighbor distances in one KDTree query. This
        # replaces an O(N^2) Python loop with hundreds of transform calls.
        neighbor_pixels = self.ax.transData.transform(
            np.asarray([(p.x, p.y) for p in neighbor_points], dtype=float)
        ) if neighbor_points else np.empty((0, 2))
        if len(neighbor_pixels) > 1:
            neighbor_distances = cKDTree(neighbor_pixels).query(neighbor_pixels, k=2)[0][:, 1]
        else:
            neighbor_distances = np.full(len(neighbor_pixels), math.inf)
        open_space_points = [
            (float(neighbor_distances[pos]), index, point)
            for pos, (index, point) in enumerate(visible_points)
            if index not in drawn_indices
        ]
        open_space_points = [
            item for item in open_space_points if item[0] >= LABEL_OPEN_SPACE_MIN_PIXELS
        ]
        open_space_points.sort(
            key=lambda item: (-item[0], self._ideology_label_sort_key(item[1], item[2]))
        )

        for _openness, index, point in open_space_points:
            if len(drawn_indices) >= label_limit:
                break
            try_draw_label(index, point)

        for index, point in visible_points:
            if len(drawn_indices) >= label_limit:
                break
            try_draw_label(index, point)

        return drawn_indices

    def _draw_category_labels(self) -> Tuple[List[LabelBox], List[CompassPoint]]:
        label_boxes: List[LabelBox] = []
        labeled_points: List[CompassPoint] = []
        category_area_by_key = self._category_voronoi_area_by_key()
        visible_points = [point for point in self.category_points if self._point_is_visible(point)]
        visible_points.sort(
            key=lambda point: self._category_label_priority_key(point, category_area_by_key)
        )

        for point in visible_points:
            placement = self._label_placement(
                point, point.name, 10, CATEGORY_LABEL_OFFSET_PIXELS, label_boxes
            )
            if placement is None:
                continue

            self.ax.text(
                placement.x,
                placement.y,
                point.name,
                color=self._category_text_color(point),
                fontsize=placement.fontsize,
                fontweight="bold",
                alpha=0.98,
                ha=placement.ha,
                va=placement.va,
                zorder=4.5,
                clip_on=True,
            )
            label_boxes.append(placement.box)
            labeled_points.append(point)

        return label_boxes, labeled_points

    def _draw_selected_point(self) -> None:
        center = self._active_center()
        if center is None:
            return
        x, y = center
        if not (self.ax.get_xlim()[0] <= x <= self.ax.get_xlim()[1]
                and self.ax.get_ylim()[0] <= y <= self.ax.get_ylim()[1]):
            return
        self.ax.scatter(
            [x], [y], s=190, facecolors="none", edgecolors=SELECTED_RING,
            linewidths=2.4, zorder=9,
        )
        self.ax.scatter(
            [x], [y], s=62, c=SELECTED_COLOR, edgecolors="#000000",
            linewidths=1.0, zorder=10,
        )
        info = (self._format_point_info(self.selected_index)
                if self.selected_index is not None
                else f"{self.pinned_label or 'Custom coordinates'}\nx: {x:g}\ny: {y:g}")
        self.ax.annotate(
            info, xy=(x, y), xytext=(14, 14), textcoords="offset points",
            color=TEXT_COLOR, fontsize=9,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": TOOLTIP_BACKGROUND,
                  "edgecolor": TOOLTIP_EDGE, "alpha": 0.94},
            arrowprops={"arrowstyle": "->", "color": TOOLTIP_EDGE, "lw": 1.0},
            zorder=11,
        )

    def _on_search_changed(self, *_args: object, show_matches: bool = True) -> None:
        query = self.search_var.get().strip()
        self.results_listbox.delete(0, tk.END)
        self.listbox_indices = []
        if not query:
            # Editing a search isn't an explicit request to clear the pin.
            # Only Clear or Esc clears selected coordinates.
            self.search_feedback_var.set("Search by ideology name or type x,y coordinates.")
            self._set_matches_visible(False)
            return
        coordinates = parse_coordinate_query(query)
        if coordinates is not None:
            self.search_feedback_var.set(
                f"Coordinates ({coordinates[0]:g}, {coordinates[1]:g}) ready. Press Enter or Go."
            )
            self._set_matches_visible(False)
            return

        matches = [
            (index, point) for index, point in enumerate(self.points)
            if query.casefold() in point.name.casefold()
            or (point.group is not None and query.casefold() in point.group.casefold())
        ]
        matches.sort(key=lambda item: (
            item[1].name.casefold() != query.casefold(),
            not item[1].name.casefold().startswith(query.casefold()),
            item[1].name.casefold(),
        ))
        for index, point in matches[:MAX_SEARCH_RESULTS]:
            group_text = f" — {point.group}" if point.group else ""
            self.results_listbox.insert(
                tk.END, f"{point.name} ({point.x:g}, {point.y:g}){group_text}"
            )
            self.listbox_indices.append(index)
        if matches:
            self.results_listbox.config(height=min(5, len(self.listbox_indices)))
            self.matches_title_var.set(f"{len(matches)} match(es) • click one or press Enter to select")
            self.search_feedback_var.set(f"{len(matches)} matches • Enter selects the first")
            self._set_matches_visible(show_matches)
        else:
            self.search_feedback_var.set(
                "No matches at this detail level. Try a different level or x,y coordinates."
            )
            self._set_matches_visible(False)

    def _on_result_selected(self, _event: object = None) -> None:
        selection = self.results_listbox.curselection()
        if not selection:
            return
        listbox_index = int(selection[0])
        if listbox_index >= len(self.listbox_indices):
            return
        self.select_point(self.listbox_indices[listbox_index], focus=True)

    def _on_search_enter(self, _event: object = None) -> str:
        query = self.search_var.get().strip()
        coordinates = parse_coordinate_query(query)
        if coordinates is not None:
            if not all(WORLD_MIN <= value <= WORLD_MAX for value in coordinates):
                self.search_feedback_var.set(
                    f"Coordinates must be between {WORLD_MIN:g} and {WORLD_MAX:g} "
                    "on both axes."
                )
                return "break"
            if not all(COMPASS_MIN <= value <= COMPASS_MAX for value in coordinates):
                self._enable_extremes_for_focus()
            self.pinned_center = coordinates
            self.pinned_label = "Custom coordinates"
            self.custom_center = coordinates
            self.selected_index = None
            self.search_feedback_var.set(
                f"Showing rings at ({coordinates[0]:g}, {coordinates[1]:g})."
            )
            self._update_overton_panel()
            self._update_status()
            self._set_matches_visible(False)
            self.center_on_coordinate(*coordinates)
            return "break"

        selection = self.results_listbox.curselection()
        if selection:
            self._on_result_selected()
        elif self.listbox_indices:
            self.results_listbox.selection_clear(0, tk.END)
            self.results_listbox.selection_set(0)
            self.results_listbox.activate(0)
            self.select_point(self.listbox_indices[0], focus=True)
        elif query:
            self.search_feedback_var.set(
                "No ideology found. Enter an ideology name or coordinates as x,y."
            )
        return "break"

    def _enable_extremes_for_focus(self) -> None:
        if not self.show_extremes_var.get():
            previous_result_indices = list(self.listbox_indices)
            self.show_extremes_var.set(True)
            # Rebuild cells to include coordinates in the outer band.
            self._apply_dataset(self.active_dataset_index)
            self.listbox_indices = previous_result_indices
            self._update_status()

    def _on_escape(self, _event: object = None) -> str:
        self.clear_selection()
        return "break"

    def clear_selection(self) -> None:
        self.pinned_center = None
        self.pinned_label = None
        self.selected_index = None
        self.custom_center = None
        self.search_var.set("")
        self.results_listbox.delete(0, tk.END)
        self.listbox_indices = []
        self._set_matches_visible(False)
        self._update_overton_panel()
        self._update_status()
        self.draw_plot(keep_view=True)

    def select_point(self, index: int, focus: bool = False) -> None:
        if index < 0 or index >= len(self.points):
            return
        point = self.points[index]
        if not self._point_is_in_current_graph_bounds(point):
            self._enable_extremes_for_focus()
        self.pinned_center = (point.x, point.y)
        self.pinned_label = point.name
        self.custom_center = None
        self.selected_index = index
        self.search_feedback_var.set(f"Selected: {point.name}")
        self._set_matches_visible(False)
        self._update_overton_panel()
        self._update_status()
        if focus:
            self.center_on_coordinate(point.x, point.y)
        else:
            self.draw_plot(keep_view=True)

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

    def _nearest_point_index(self, event: MouseEvent) -> Optional[int]:
        if event.x is None or event.y is None:
            return None

        candidate_indices = [
            index
            for index, point in enumerate(self.points)
            if self._point_is_visible(point)
            and self._point_coord_key(point) not in self.category_coord_keys
        ]
        if not candidate_indices:
            return None

        x_pixel = float(event.x)
        y_pixel = float(event.y)
        display_coords = self.ax.transData.transform(
            np.asarray(
                [(self.points[index].x, self.points[index].y) for index in candidate_indices],
                dtype=float,
            )
        )
        pointer = np.asarray([x_pixel, y_pixel], dtype=float)
        distances = np.linalg.norm(display_coords - pointer, axis=1)

        nearest_candidate_index = int(np.argmin(distances))
        if distances[nearest_candidate_index] <= CLICK_DISTANCE_PIXELS:
            return candidate_indices[nearest_candidate_index]
        return None

    # ------------------------------------------------------------------
    # Zoom / pan
    # ------------------------------------------------------------------

    def _on_scroll(self, event: MouseEvent) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        xdata = float(event.xdata)
        ydata = float(event.ydata)
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        current_width = current_xlim[1] - current_xlim[0]
        current_height = current_ylim[1] - current_ylim[0]

        step = getattr(event, "step", 0)
        if step > 0 or getattr(event, "button", None) == "up":
            scale = SCROLL_ZOOM_IN_SCALE
        else:
            scale = SCROLL_ZOOM_OUT_SCALE

        new_width = float(np.clip(current_width * scale, MIN_VIEW_SPAN, self.view_max_x - self.view_min_x))
        new_height = float(np.clip(current_height * scale, MIN_VIEW_SPAN, self.view_max_y - self.view_min_y))

        rel_x = (xdata - current_xlim[0]) / current_width if current_width else 0.5
        rel_y = (ydata - current_ylim[0]) / current_height if current_height else 0.5

        new_xmin = xdata - rel_x * new_width
        new_ymin = ydata - rel_y * new_height
        new_xlim = self._clamp_x_limits(new_xmin, new_xmin + new_width)
        new_ylim = self._clamp_y_limits(new_ymin, new_ymin + new_height)

        self.ax.set_xlim(*new_xlim)
        self.ax.set_ylim(*new_ylim)
        self._preview_navigation()

    def _on_mouse_press(self, event: MouseEvent) -> None:
        if (
            event.inaxes != self.ax
            or getattr(event, "button", None) != 1
            or event.x is None
            or event.y is None
        ):
            return

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.drag_start = {
            "x_pixel": float(event.x),
            "y_pixel": float(event.y),
            "xlim": (float(xlim[0]), float(xlim[1])),
            "ylim": (float(ylim[0]), float(ylim[1])),
            "bbox_width": float(self.ax.bbox.width),
            "bbox_height": float(self.ax.bbox.height),
        }
        self.is_dragging = False
        self._cancel_navigation_refine()
        # The already rendered Tk image can be translated immediately, without
        # re-running Matplotlib for every mousemove. This also keeps text sharp
        # because the final full draw happens only on mouse release.
        self._pan_image_item = getattr(self.canvas, "_tkcanvas_image_region", None)
        if self._pan_image_item is not None:
            coords = self.canvas_widget.coords(self._pan_image_item)
            self._pan_image_origin = (float(coords[0]), float(coords[1])) if len(coords) == 2 else None
        else:
            self._pan_image_origin = None

    def _on_mouse_move(self, event: MouseEvent) -> None:
        if self.drag_start is None or event.x is None or event.y is None:
            return

        x_pixel = float(event.x)
        y_pixel = float(event.y)
        start_x = self.drag_start["x_pixel"]
        start_y = self.drag_start["y_pixel"]
        dx_pixels = x_pixel - start_x
        dy_pixels = y_pixel - start_y

        if abs(dx_pixels) > PAN_CLICK_TOLERANCE_PIXELS or abs(dy_pixels) > PAN_CLICK_TOLERANCE_PIXELS:
            self.is_dragging = True

        if not self.is_dragging:
            return

        xlim = self.drag_start["xlim"]
        ylim = self.drag_start["ylim"]
        bbox_width = max(self.drag_start["bbox_width"], 1.0)
        bbox_height = max(self.drag_start["bbox_height"], 1.0)

        width = xlim[1] - xlim[0]
        height = ylim[1] - ylim[0]

        dx_data = dx_pixels / bbox_width * width
        dy_data = dy_pixels / bbox_height * height

        new_xlim = self._clamp_x_limits(xlim[0] - dx_data, xlim[1] - dx_data)
        new_ylim = self._clamp_y_limits(ylim[0] - dy_data, ylim[1] - dy_data)

        self.ax.set_xlim(*new_xlim)
        self.ax.set_ylim(*new_ylim)
        if self._pan_image_origin is not None and self._pan_image_item is not None:
            # Move the raster to match the updated axes transformation.
            actual_dx = (xlim[0] - new_xlim[0]) / width * bbox_width
            actual_dy = -(ylim[0] - new_ylim[0]) / height * bbox_height
            ox, oy = self._pan_image_origin
            self.canvas_widget.coords(self._pan_image_item, ox + actual_dx, oy + actual_dy)
        else:
            self._preview_navigation()

    def _on_mouse_release(self, event: MouseEvent) -> None:
        if self.drag_start is None:
            return

        was_dragging = self.is_dragging
        self.drag_start = None
        self.is_dragging = False
        if self._pan_image_origin is not None and self._pan_image_item is not None:
            self.canvas_widget.coords(self._pan_image_item, *self._pan_image_origin)
        self._pan_image_item = None
        self._pan_image_origin = None

        if was_dragging:
            self.draw_plot(keep_view=True)
            return

        if event.inaxes != self.ax:
            return

        nearest_index = self._nearest_point_index(event)
        if nearest_index is not None:
            self.select_point(nearest_index, focus=False)

    def _clamp_limits(
        self,
        low: float,
        high: float,
        view_min: float,
        view_max: float,
    ) -> Tuple[float, float]:
        span = high - low
        max_span = view_max - view_min
        span = float(np.clip(span, MIN_VIEW_SPAN, max_span))

        low = float(low)
        high = low + span

        if low < view_min:
            low = view_min
            high = view_min + span
        if high > view_max:
            high = view_max
            low = view_max - span

        return low, high

    def _focus_bounds(self) -> Tuple[float, float, float, float]:
        x_low, x_high = self.view_min_x, self.view_max_x
        y_low, y_high = self.view_min_y, self.view_max_y
        center = self._active_center()
        if center is not None and self.show_overton_var.get():
            padding = OVERTON_MAX_RADIUS + OVERTON_FOCUS_MARGIN
            x_low = min(x_low, center[0] - padding)
            x_high = max(x_high, center[0] + padding)
            y_low = min(y_low, center[1] - padding)
            y_high = max(y_high, center[1] + padding)
        return x_low, x_high, y_low, y_high

    def _clamp_x_limits(self, low: float, high: float) -> Tuple[float, float]:
        x_low, x_high, _y_low, _y_high = self._focus_bounds()
        return self._clamp_limits(low, high, x_low, x_high)

    def _clamp_y_limits(self, low: float, high: float) -> Tuple[float, float]:
        _x_low, _x_high, y_low, y_high = self._focus_bounds()
        return self._clamp_limits(low, high, y_low, y_high)

    def center_on_coordinate(self, x: float, y: float) -> None:
        width = SEARCH_FOCUS_SPAN if self.show_overton_var.get() else 6.0
        height = width
        xlim = self._clamp_x_limits(x - width / 2, x + width / 2)
        ylim = self._clamp_y_limits(y - height / 2, y + height / 2)
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.draw_plot(keep_view=True)

    def center_on_point(self, point: CompassPoint) -> None:
        self.center_on_coordinate(point.x, point.y)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _update_status(self) -> None:
        dataset_label = self.datasets[self.active_dataset_index].label
        skipped_count = len(self.warnings) + len(self.category_warnings)
        warning_text = f" | {skipped_count} skipped row(s)" if skipped_count else ""
        center = self._active_center()
        center_text = f" | Center ({center[0]:g}, {center[1]:g})" if center else ""
        self.status_var.set(
            f"{dataset_label} | {len(self.points)} ideologies | "
            f"{len(self.category_points)} categories{warning_text}{center_text}"
        )


def main() -> None:
    root = tk.Tk()
    root.withdraw()

    try:
        points1, warnings1 = load_points(CSV_PATH)
        category_points1, category_warnings1 = load_points(CSV_CATS)
        points2, warnings2 = load_points(CSV_PATH2)
        category_points2, category_warnings2 = load_points(CSV_CATS2)
        points3, warnings3 = load_points(CSV_PATH3)
        category_points3, category_warnings3 = load_points(CSV_CATS3)
        points4, warnings4 = load_points(CSV_PATH4)
        category_points4, category_warnings4 = load_points(CSV_CATS4)
        points5, warnings5 = load_points(CSV_PATH5)
        category_points5, category_warnings5 = load_points(CSV_CATS5)
        points6, warnings6 = load_points(CSV_PATH6)
        category_points6, category_warnings6 = load_points(CSV_CATS6)
        points7, warnings7 = load_points(CSV_PATH7)
        category_points7, category_warnings7 = load_points(CSV_CATS7)
        points8, warnings8 = load_points(CSV_PATH8)
        category_points8, category_warnings8 = load_points(CSV_CATS8)
        points9, warnings9 = load_points(CSV_PATH9)
        category_points9, category_warnings9 = load_points(CSV_CATS9)
        points10, warnings10 = load_points(CSV_PATH10)
        category_points10, category_warnings10 = load_points(CSV_CATS10)
    except DataLoadError as exc:
        print(exc, file=sys.stderr)
        messagebox.showerror(APP_TITLE, str(exc))
        root.destroy()
        return

    datasets = [
        CompassDataset(
            label="Level 1",
            points=points1,
            warnings=warnings1,
            category_points=category_points1,
            category_warnings=category_warnings1,
        ),
        CompassDataset(
            label="Level 2",
            points=points2,
            warnings=warnings2,
            category_points=category_points2,
            category_warnings=category_warnings2,
        ),
        CompassDataset(
            label="Level 3",
            points=points3,
            warnings=warnings3,
            category_points=category_points3,
            category_warnings=category_warnings3,
        ),
        CompassDataset(
            label="Level 4",
            points=points4,
            warnings=warnings4,
            category_points=category_points4,
            category_warnings=category_warnings4,
        ),
        CompassDataset(
            label="Level 5",
            points=points5,
            warnings=warnings5,
            category_points=category_points5,
            category_warnings=category_warnings5,
        ),
        CompassDataset(
            label="Level 6",
            points=points6,
            warnings=warnings6,
            category_points=category_points6,
            category_warnings=category_warnings6,
        ),
        CompassDataset(
            label="Level 7",
            points=points7,
            warnings=warnings7,
            category_points=category_points7,
            category_warnings=category_warnings7,
        ),
        CompassDataset(
            label="Level 8",
            points=points8,
            warnings=warnings8,
            category_points=category_points8,
            category_warnings=category_warnings8,
        ),
        CompassDataset(
            label="Level 9",
            points=points9,
            warnings=warnings9,
            category_points=category_points9,
            category_warnings=category_warnings9,
        ),
        CompassDataset(
            label="Level 10",
            points=points10,
            warnings=warnings10,
            category_points=category_points10,
            category_warnings=category_warnings10,
        ),
    ]

    root.deiconify()
    app = NewPoliticalCompassApp(root, datasets)
    root.mainloop()


if __name__ == "__main__":
    main()
