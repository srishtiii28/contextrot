import pytest
from app.services.graph_service import (
    render_cartesian_graph,
    render_geometry_diagram,
    process_message_graphs,
    synthesize_graph_from_text,
)


def test_render_cartesian_graph_coordinates_in_labels():
    spec = {
        "title": "Pair of Linear Equations: 2x + 3y = 9 & 4x + 6y = 18",
        "xLabel": "x (Cost of pencil in Rs)",
        "yLabel": "y (Cost of eraser in Rs)",
        "nature": "coincident",
        "lines": [
            {"name": "2x + 3y = 9", "points": [{"x": 0, "y": 3}, {"x": 4.5, "y": 0}], "color": "#2563eb"},
            {"name": "4x + 6y = 18", "points": [{"x": 3, "y": 1}, {"x": 6, "y": -1}], "color": "#dc2626", "coincident": True},
        ],
        "points": [
            {"x": 0, "y": 3, "label": "A/C"},
            {"x": 4.5, "y": 0, "label": "B"},
            {"x": 3, "y": 1, "label": "D"},
        ],
    }

    svg = render_cartesian_graph(spec)
    assert "<svg" in svg
    assert "</svg>" in svg
    assert "<?xml" not in svg
    assert "<!DOCTYPE" not in svg

    # Ensure point labels ALWAYS contain actual numerical coordinates
    assert "A/C (0, 3)" in svg
    assert "B (4.5, 0)" in svg
    assert "D (3, 1)" in svg

    # Ensure coincident overlay line is present
    assert "Coincident Overlay" in svg


def test_render_geometry_diagram():
    spec = {
        "type": "geometry",
        "title": "Tangents from an External Point",
        "elements": [
            {"type": "circle", "x": 5, "y": 5, "r": 3, "label": "Circle O"},
            {"type": "point", "x": 10, "y": 5, "label": "P(10, 5)"},
            {"type": "line", "from": [10, 5], "to": [5, 8], "label": "PT1"},
            {"type": "line", "from": [10, 5], "to": [5, 2], "label": "PT2"},
            {"type": "triangle", "points": [[0, 0], [4, 0], [0, 3]], "labels": ["A", "B", "C"]},
        ],
    }

    svg = render_geometry_diagram(spec)
    assert "<svg" in svg
    assert "</svg>" in svg
    assert "<?xml" not in svg
    assert "Tangents from an External Point" in svg


def test_process_message_graphs_nested_fences():
    text = """Here is the complete solution:

```graph
{
  "title": "Romila's Stationary Purchase",
  "nature": "coincident",
  "lines": [
    {
      "name": "2x + 3y = 9",
      "points": [{"x": 0, "y": 3}, {"x": 4.5, "y": 0}]
    }
  ],
  "points": [
    {"x": 0, "y": 3, "label": "A/C"},
    {"x": 4.5, "y": 0, "label": "B"}
  ]
}
```

Both lines coincide everywhere.
"""
    processed = process_message_graphs(text)
    assert "<svg" in processed
    assert "A/C (0, 3)" in processed
    assert "Romila's Stationary Purchase" in processed
    assert "```graph" not in processed


def test_process_message_graphs_fallback_synthesis():
    text = """Let us represent the pair of linear equations graphically:
Equation 1: 2x + 3y = 9 with points (0, 3) and (4.5, 0).
Equation 2: 4x + 6y = 18 with points (3, 1) and (6, -1).
Since the lines coincide everywhere, there are infinitely many solutions.
"""
    processed = process_message_graphs(text)
    assert "<svg" in processed
    assert "(0, 3)" in processed
    assert "(4.5, 0)" in processed
