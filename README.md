# New Political Compass

Interactive political compass visualization for the ideology coordinate dataset in `ideology_coordinates.csv`.

The project includes two versions:

- `new_political_compass.py`: desktop Python/Tk app with Matplotlib.
- `index.html`: shareable browser version with the same CSV data, search, point selection, pan, zoom, labels, and Voronoi cells.

## Browser Version

Open `index.html` through a web server so the page can load `ideology_coordinates.csv`.

```bash
python3 -m http.server 8000
```

Then visit:

```text
http://localhost:8000/
```

The HTML version is ready for GitHub Pages. After pushing the repo to GitHub, enable Pages for the branch that contains `index.html`, then share the generated Pages URL.

## Python Version

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the desktop app:

```bash
python3 new_political_compass.py
```

## CSV Format

Required columns:

```text
name,x,y
```

Optional column:

```text
group
```

Coordinates must be finite numbers between `-10` and `10`. Rows outside those bounds are skipped by both versions.
