import io
import json
import re
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _extract_coord(p: Any) -> tuple[float, float] | None:
    if p is None:
        return None
    if isinstance(p, (list, tuple)) and len(p) >= 2:
        try:
            return float(p[0]), float(p[1])
        except (ValueError, TypeError):
            return None
    if isinstance(p, dict):
        x = p.get("x", p.get("X"))
        y = p.get("y", p.get("Y"))
        if x is not None and y is not None:
            try:
                return float(x), float(y)
            except (ValueError, TypeError):
                return None
    return None


def _parse_color(c: Any, default: Any = "#2563eb") -> Any:
    if not c:
        return default
    if isinstance(c, (list, tuple)):
        return c
    c_str = str(c).strip()
    m = re.match(r"rgba?\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*([\d\.]+))?\s*\)", c_str, re.I)
    if m:
        r, g, b = int(m.group(1)) / 255.0, int(m.group(2)) / 255.0, int(m.group(3)) / 255.0
        a = float(m.group(4)) if m.group(4) is not None else 1.0
        return (r, g, b, a)
    if c_str.lower() in ("none", "transparent"):
        return "none"
    return c_str


def _get_line_points_from_equation(eq_str: str) -> list[tuple[float, float]]:
    if not eq_str:
        return []

    # First check ax + by + c = 0 format
    m_zero = re.search(
        r"([+\-]?[0-9]*\.?[0-9]*)\s*x\s*([+\-]?\s*[0-9]*\.?[0-9]*)\s*y\s*([+\-]\s*[0-9]*\.?[0-9]+)\s*=\s*0",
        eq_str,
        re.IGNORECASE,
    )
    if m_zero:
        a_str = (m_zero.group(1) or "1").replace(" ", "")
        if a_str in ("+", ""):
            a_str = "1"
        elif a_str == "-":
            a_str = "-1"
        b_str = (m_zero.group(2) or "+1").replace(" ", "")
        if b_str in ("+", ""):
            b_str = "1"
        elif b_str == "-":
            b_str = "-1"
        try:
            a = float(a_str)
            b = float(b_str)
            c = -float(m_zero.group(3).replace(" ", ""))
            if a != 0 or b != 0:
                pts = []
                if b != 0:
                    pts.append((0.0, c / b))
                if a != 0:
                    pts.append((c / a, 0.0))
                if len(pts) >= 2 and abs(pts[0][0] - pts[1][0]) > 1e-4:
                    return pts
                if b != 0:
                    pts.append((2.0, (c - 2.0 * a) / b))
                return pts
        except (ValueError, TypeError):
            pass

    # Standard ax + by = c format
    m = re.search(
        r"([+\-]?[0-9]*\.?[0-9]*)\s*x\s*([+\-]?\s*[0-9]*\.?[0-9]*)\s*y\s*=\s*([+\-]?[0-9]*\.?[0-9]+)",
        eq_str,
        re.IGNORECASE,
    )
    if m:
        a_str = (m.group(1) or "1").replace(" ", "")
        if a_str in ("+", ""):
            a_str = "1"
        elif a_str == "-":
            a_str = "-1"
        b_str = (m.group(2) or "+1").replace(" ", "")
        if b_str in ("+", ""):
            b_str = "1"
        elif b_str == "-":
            b_str = "-1"
        try:
            a = float(a_str)
            b = float(b_str)
            c = float(m.group(3))
            if a != 0 or b != 0:
                pts = []
                if b != 0:
                    pts.append((0.0, c / b))
                if a != 0:
                    pts.append((c / a, 0.0))
                if len(pts) >= 2 and abs(pts[0][0] - pts[1][0]) > 1e-4:
                    return pts
                if b != 0:
                    pts.append((2.0, (c - 2.0 * a) / b))
                return pts
        except (ValueError, TypeError):
            pass
    return []


def render_cartesian_graph(spec: dict[str, Any]) -> str:
    """Renders an accurate, publication-quality 2D Cartesian SVG graph using Matplotlib."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=100)

    title = spec.get("title") or "Cartesian Coordinate Graph"
    xlabel = spec.get("xLabel") or "x"
    ylabel = spec.get("yLabel") or "y"
    nature = str(spec.get("nature", "")).lower()
    raw_lines = spec.get("lines", []) if isinstance(spec.get("lines"), list) else []
    raw_points = spec.get("points", []) if isinstance(spec.get("points"), list) else []

    all_x: list[float] = []
    all_y: list[float] = []

    for l in raw_lines:
        for p in l.get("points", []):
            pt = _extract_coord(p)
            if pt:
                all_x.append(pt[0])
                all_y.append(pt[1])

    for p in raw_points:
        pt = _extract_coord(p)
        if pt:
            all_x.append(pt[0])
            all_y.append(pt[1])

    min_x = min(all_x) if all_x else 0.0
    max_x = max(all_x) if all_x else 5.0
    min_y = min(all_y) if all_y else 0.0
    max_y = max(all_y) if all_y else 5.0

    x_span = max(max_x - min_x, 1.0)
    y_span = max(max_y - min_y, 1.0)
    x_pad = max(1.5, x_span * 0.22)
    y_pad = max(1.5, y_span * 0.22)

    plot_min_x = min(min_x, 0.0) - x_pad
    plot_max_x = max(max_x, 2.0) + x_pad
    plot_min_y = min(min_y, 0.0) - y_pad
    plot_max_y = max(max_y, 2.0) + y_pad

    xs = np.linspace(plot_min_x, plot_max_x, 300)

    is_coincident = "coincid" in nature or any(
        "coincid" in str(l.get("nature", "")).lower() or l.get("coincident")
        for l in raw_lines
    )

    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c"]

    for idx, line in enumerate(raw_lines):
        line_pts = [_extract_coord(p) for p in line.get("points", []) if _extract_coord(p)]
        eq_name = line.get("name") or line.get("equation") or line.get("label") or f"Line {idx+1}"

        if len(line_pts) < 2:
            line_pts = _get_line_points_from_equation(eq_name)

        if len(line_pts) < 2 and len(raw_points) >= 2:
            c1 = _extract_coord(raw_points[0])
            c2 = _extract_coord(raw_points[1])
            if c1 and c2:
                line_pts = [c1, c2]

        color = line.get("color") or colors[idx % len(colors)]

        if len(line_pts) >= 2:
            x1, y1 = line_pts[0]
            x2, y2 = line_pts[1]
            if abs(x2 - x1) < 1e-9:
                ax.axvline(x1, color=color, linewidth=2.6, label=eq_name, zorder=3)
            else:
                m = (y2 - y1) / (x2 - x1)
                c = y1 - m * x1
                ys = m * xs + c
                if is_coincident and idx == 1:
                    ax.plot(
                        xs,
                        ys,
                        color="#dc2626",
                        linewidth=2.8,
                        linestyle="--",
                        label=f"{eq_name} (Coincident Overlay)",
                        zorder=5,
                    )
                elif is_coincident and idx == 0:
                    ax.plot(
                        xs,
                        ys,
                        color="#2563eb",
                        linewidth=4.2,
                        label=f"{eq_name} (Base Line)",
                        zorder=4,
                    )
                else:
                    dash_style = "--" if line.get("style") == "dashed" else "-"
                    ax.plot(xs, ys, color=color, linewidth=2.6, linestyle=dash_style, label=eq_name, zorder=3)

                # Label equation along the line in authentic NCERT style
                if eq_name and not is_coincident:
                    lx = min_x + 0.65 * (max_x - min_x) if (max_x - min_x) > 1.5 else (x1 + x2) / 2
                    ly = m * lx + c
                    if plot_min_y <= ly <= plot_max_y:
                        angle_deg = float(np.degrees(np.arctan(m)))
                        angle_deg = max(-65.0, min(65.0, angle_deg))
                        ax.text(
                            lx,
                            ly,
                            f" {eq_name} ",
                            fontsize=9,
                            fontweight="bold",
                            color=color,
                            rotation=angle_deg,
                            rotation_mode="anchor",
                            ha="center",
                            va="bottom",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.9, lw=0.8),
                            zorder=5,
                        )

    # Support shaded polygon region (e.g. for triangles bounded by lines and an axis)
    shaded_pts = spec.get("shaded_region") or spec.get("shade_region") or spec.get("shade")
    if isinstance(shaded_pts, list) and len(shaded_pts) >= 3:
        poly_pts = [_extract_coord(p) for p in shaded_pts if _extract_coord(p)]
        if len(poly_pts) >= 3:
            poly = plt.Polygon(
                poly_pts,
                closed=True,
                facecolor=(0.15, 0.45, 0.9, 0.22),
                edgecolor="#1d4ed8",
                hatch="//",
                linewidth=1.8,
                zorder=2,
                label=spec.get("shaded_label") or "Shaded Triangular Region",
            )
            ax.add_patch(poly)

    # Plot points with ALWAYS guaranteed actual coordinates (x, y)
    seen_points: set[tuple[float, float]] = set()
    for p in raw_points:
        pt = _extract_coord(p)
        if not pt:
            continue
        px, py = pt
        if (px, py) in seen_points:
            continue
        seen_points.add((px, py))

        raw_label = str(p.get("label", "")).strip() if isinstance(p, dict) else ""
        coord_repr = f"({px:g}, {py:g})"

        # Ensure the actual coordinates (x, y) are ALWAYS part of the label
        if raw_label and coord_repr not in raw_label:
            full_label = f"{raw_label} {coord_repr}"
        elif raw_label:
            full_label = raw_label
        else:
            full_label = coord_repr

        ax.scatter(px, py, color="#0f172a", s=55, zorder=7, edgecolors="white", linewidth=1.6)
        ax.annotate(
            full_label,
            (px, py),
            textcoords="offset points",
            xytext=(10, 8),
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#94a3b8", alpha=0.94, lw=1.0),
            fontsize=9.5,
            fontweight="bold",
            color="#0f172a",
            zorder=8,
        )

    # Clean axes through origin with directional arrow style
    ax.axhline(0, color="#334155", linewidth=1.5, zorder=2)
    ax.axvline(0, color="#334155", linewidth=1.5, zorder=2)
    ax.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1", zorder=1)

    ax.set_xlim(plot_min_x, plot_max_x)
    ax.set_ylim(plot_min_y, plot_max_y)

    # Axis arrows and directional labels X, X', Y, Y' in authentic NCERT textbook style
    arrow_len_x = (plot_max_x - plot_min_x) * 0.035
    arrow_len_y = (plot_max_y - plot_min_y) * 0.035

    # Positive X axis (right)
    ax.annotate("", xy=(plot_max_x, 0), xytext=(plot_max_x - arrow_len_x, 0),
                arrowprops=dict(arrowstyle="->", color="#1e293b", lw=1.8, mutation_scale=12), zorder=6)
    ax.text(plot_max_x, 0, "  X", fontsize=11, fontweight="bold", fontstyle="italic",
            color="#0f172a", va="center", ha="left", zorder=6)

    # Negative X axis (left)
    ax.annotate("", xy=(plot_min_x, 0), xytext=(plot_min_x + arrow_len_x, 0),
                arrowprops=dict(arrowstyle="->", color="#1e293b", lw=1.8, mutation_scale=12), zorder=6)
    ax.text(plot_min_x, 0, "X'  ", fontsize=11, fontweight="bold", fontstyle="italic",
            color="#0f172a", va="center", ha="right", zorder=6)

    # Positive Y axis (up)
    ax.annotate("", xy=(0, plot_max_y), xytext=(0, plot_max_y - arrow_len_y),
                arrowprops=dict(arrowstyle="->", color="#1e293b", lw=1.8, mutation_scale=12), zorder=6)
    ax.text(0, plot_max_y, "  Y\n", fontsize=11, fontweight="bold", fontstyle="italic",
            color="#0f172a", va="bottom", ha="center", zorder=6)

    # Negative Y axis (down)
    ax.annotate("", xy=(0, plot_min_y), xytext=(0, plot_min_y + arrow_len_y),
                arrowprops=dict(arrowstyle="->", color="#1e293b", lw=1.8, mutation_scale=12), zorder=6)
    ax.text(0, plot_min_y, "\nY'", fontsize=11, fontweight="bold", fontstyle="italic",
            color="#0f172a", va="top", ha="center", zorder=6)

    # Origin label O
    ax.text(-0.35 * (plot_max_x - plot_min_x) / 10.0, -0.35 * (plot_max_y - plot_min_y) / 10.0,
            "O", fontsize=10.5, fontweight="bold", fontstyle="italic", color="#334155", ha="right", va="top", zorder=6)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=14, color="#0f172a")
    ax.set_xlabel(xlabel, fontsize=10.5, fontweight="bold", labelpad=8, color="#1e293b")
    ax.set_ylabel(ylabel, fontsize=10.5, fontweight="bold", labelpad=8, color="#1e293b")

    if raw_lines or (isinstance(shaded_pts, list) and len(shaded_pts) >= 3):
        ax.legend(loc="upper right", framealpha=0.95, edgecolor="#e2e8f0", fontsize=9.5)

    buf = io.StringIO()
    plt.tight_layout()
    plt.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    raw_svg = buf.getvalue()
    raw_svg = re.sub(r"<\?xml[^>]*\?>", "", raw_svg).strip()
    raw_svg = re.sub(r"<!DOCTYPE[^>]*>", "", raw_svg).strip()
    return raw_svg


def render_geometry_diagram(spec: dict[str, Any]) -> str:
    """Renders an accurate geometric diagram (triangles, circles, tangents, angles) via Matplotlib."""
    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=100)

    title = spec.get("title") or "Geometric Diagram"
    elements = spec.get("elements", []) if isinstance(spec.get("elements"), list) else []

    all_x = [0.0, 10.0]
    all_y = [0.0, 10.0]

    for el in elements:
        t = str(el.get("type", "")).lower()
        if t in ("triangle", "polygon"):
            raw_pts = el.get("points") or el.get("vertices") or []
            pts = [_extract_coord(p) for p in raw_pts if _extract_coord(p)]
            if len(pts) >= 3:
                poly = plt.Polygon(
                    pts,
                    closed=True,
                    facecolor=_parse_color(el.get("fill"), (0.15, 0.39, 0.92, 0.08)),
                    edgecolor=_parse_color(el.get("color"), "#2563eb"),
                    linewidth=el.get("strokeWidth", 2.2),
                    zorder=2,
                )
                ax.add_patch(poly)
                for pt in pts:
                    all_x.append(pt[0])
                    all_y.append(pt[1])

            # Vertex labels
            labels = el.get("labels", [])
            for i, lbl in enumerate(labels):
                if i < len(pts):
                    px, py = pts[i]
                    ax.scatter(px, py, color="#1e293b", s=40, zorder=6)
                    ax.annotate(
                        str(lbl),
                        (px, py),
                        textcoords="offset points",
                        xytext=(7, 7),
                        fontsize=11,
                        fontweight="bold",
                        color="#0f172a",
                        zorder=7,
                    )

        elif t == "circle":
            cx, cy = _extract_coord(el) or (5.0, 5.0)
            r = float(el.get("r") or el.get("radius") or 3.0)
            circ = plt.Circle(
                (cx, cy),
                r,
                facecolor=_parse_color(el.get("fill"), (0.15, 0.39, 0.92, 0.05)),
                edgecolor=_parse_color(el.get("color"), "#2563eb"),
                linewidth=el.get("strokeWidth", 2.2),
                zorder=2,
            )
            ax.add_patch(circ)
            ax.scatter(cx, cy, color="#1e293b", s=35, zorder=5)
            if el.get("label"):
                ax.annotate(
                    str(el["label"]),
                    (cx, cy),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=10.5,
                    fontweight="bold",
                    color="#0f172a",
                )
            all_x.extend([cx - r - 1, cx + r + 1])
            all_y.extend([cy - r - 1, cy + r + 1])

        elif t in ("line", "segment", "ray"):
            p1 = _extract_coord(el.get("from")) or (float(el.get("x1", 0)), float(el.get("y1", 0)))
            p2 = _extract_coord(el.get("to")) or (float(el.get("x2", 0)), float(el.get("y2", 0)))
            style = "--" if el.get("style") == "dashed" else "-"
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color=_parse_color(el.get("color"), "#2563eb"),
                linewidth=el.get("strokeWidth", 2.2),
                linestyle=style,
                zorder=4,
            )
            if el.get("label"):
                mx = (p1[0] + p2[0]) / 2
                my = (p1[1] + p2[1]) / 2
                ax.annotate(
                    str(el["label"]),
                    (mx, my),
                    textcoords="offset points",
                    xytext=(0, 6),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cbd5e1", alpha=0.9),
                    fontsize=9,
                    fontweight="bold",
                    color="#334155",
                    ha="center",
                )
            all_x.extend([p1[0], p2[0]])
            all_y.extend([p1[1], p2[1]])

        elif t == "point":
            pt = _extract_coord(el)
            if pt:
                px, py = pt
                ax.scatter(px, py, color="#1e293b", s=45, zorder=6, edgecolors="white", linewidth=1.5)
                lbl = el.get("label") or f"({px:g}, {py:g})"
                ax.annotate(
                    str(lbl),
                    (px, py),
                    textcoords="offset points",
                    xytext=(8, 8),
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cbd5e1", alpha=0.9),
                    fontsize=9.5,
                    fontweight="bold",
                    color="#0f172a",
                    zorder=7,
                )
                all_x.append(px)
                all_y.append(py)

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    pad = max(1.0, max(max_x - min_x, max_y - min_y) * 0.15)
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12, color="#0f172a")

    buf = io.StringIO()
    plt.tight_layout()
    plt.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    raw_svg = buf.getvalue()
    raw_svg = re.sub(r"<\?xml[^>]*\?>", "", raw_svg).strip()
    raw_svg = re.sub(r"<!DOCTYPE[^>]*>", "", raw_svg).strip()
    return raw_svg


def synthesize_graph_from_text(text: str) -> dict[str, Any] | None:
    """Extracts equations and coordinates from mathematical text if no explicit graph block was provided."""
    lowered = text.lower()
    has_trigger = bool(
        re.search(r"\b(graphical|graphically|plot|coordinate\s+table|geometrical)\b", lowered)
        or re.search(r"\|\s*x\s*\|\s*y\s*\|", lowered)
    )
    if not has_trigger:
        return None

    raw_eq_matches = re.findall(
        r"([0-9]*\.?[0-9]*\s*[a-zA-Z]\s*[+\-]\s*[0-9]*\.?[0-9]*\s*[a-zA-Z]\s*=\s*[0-9]+(?:\.[0-9]+)?)",
        text,
    )
    seen_eq: set[str] = set()
    eq_matches = []
    for eq in raw_eq_matches:
        cleaned_eq = re.sub(r"\s+", " ", eq.strip())
        if cleaned_eq not in seen_eq:
            seen_eq.add(cleaned_eq)
            eq_matches.append(cleaned_eq)

    pt_matches = re.findall(
        r"\(\s*(-?[0-9]+(?:\.[0-9]+)?)\s*,\s*(-?[0-9]+(?:\.[0-9]+)?)\s*\)",
        text,
    )
    table_matches = re.findall(
        r"\|\s*(-?[0-9]+(?:\.[0-9]+)?)\s*\|\s*(-?[0-9]+(?:\.[0-9]+)?)\s*\|",
        text,
    )
    pts = [(float(x), float(y)) for x, y in pt_matches + table_matches]

    if len(pts) < 2 and not eq_matches:
        return None

    nature = "intersecting"
    if "coincid" in lowered or "infinitely many" in lowered:
        nature = "coincident"
    elif "parallel" in lowered or "no solution" in lowered or "inconsistent" in lowered:
        nature = "parallel"
    elif len(eq_matches) >= 2:
        # Ratio comparison fallback
        def _get_abc(eq_str: str) -> tuple[float, float, float] | None:
            m = re.search(r"([+\-]?[0-9]*\.?[0-9]*)\s*x\s*([+\-]?\s*[0-9]*\.?[0-9]*)\s*y\s*=\s*([+\-]?[0-9]*\.?[0-9]+)", eq_str, re.I)
            if not m:
                return None
            g1 = (m.group(1) or "").replace(" ", "")
            a = 1.0 if g1 in ("+", "") else (-1.0 if g1 == "-" else float(g1))
            g2 = (m.group(2) or "").replace(" ", "")
            b = 1.0 if g2 in ("+", "") else (-1.0 if g2 == "-" else float(g2))
            c = float(m.group(3))
            return a, b, c

        abc1 = _get_abc(eq_matches[0])
        abc2 = _get_abc(eq_matches[1])
        if abc1 and abc2:
            a1, b1, c1 = abc1
            a2, b2, c2 = abc2
            if a2 != 0 and b2 != 0:
                r1, r2 = a1 / a2, b1 / b2
                if abs(r1 - r2) < 1e-4:
                    if c2 != 0 and abs(r1 - (c1 / c2)) < 1e-4:
                        nature = "coincident"
                    else:
                        nature = "parallel"

    lines = []
    if len(eq_matches) >= 2 and len(pts) >= 4:
        lines.append({
            "name": eq_matches[0],
            "points": [pts[0], pts[1]],
            "color": "#2563eb",
        })
        lines.append({
            "name": eq_matches[1],
            "points": [pts[2], pts[3]],
            "color": "#dc2626",
            "coincident": nature == "coincident",
        })
    elif len(eq_matches) >= 2 and len(pts) >= 2 and nature == "coincident":
        lines.append({
            "name": eq_matches[0],
            "points": [pts[0], pts[1]],
            "color": "#2563eb",
        })
        lines.append({
            "name": eq_matches[1],
            "points": [pts[0], pts[1]],
            "color": "#dc2626",
            "coincident": True,
        })
    elif len(eq_matches) >= 1 and len(pts) >= 2:
        lines.append({
            "name": eq_matches[0],
            "points": [pts[0], pts[1]],
            "color": "#2563eb",
        })
        if len(eq_matches) >= 2:
            l2_pts = [pts[2], pts[3]] if len(pts) >= 4 else [pts[0], pts[1]]
            lines.append({
                "name": eq_matches[1],
                "points": l2_pts,
                "color": "#dc2626",
                "coincident": nature == "coincident",
            })

    if not lines and eq_matches:
        for idx, eq in enumerate(eq_matches[:2]):
            line_pts = _get_line_points_from_equation(eq)
            if len(line_pts) >= 2:
                lines.append({
                    "name": eq,
                    "points": [{"x": p[0], "y": p[1]} for p in line_pts],
                    "color": "#2563eb" if idx == 0 else "#dc2626",
                    "coincident": nature == "coincident",
                })
                for p in line_pts:
                    if p not in pts:
                        pts.append(p)

    unique_pts = []
    seen = set()
    for p in pts:
        if p not in seen:
            seen.add(p)
            unique_pts.append({"x": p[0], "y": p[1], "label": f"({p[0]:g}, {p[1]:g})"})

    shaded_region = None
    if any(k in lowered for k in ("shade", "shaded", "triangular region", "triangle formed")):
        if len(unique_pts) >= 3:
            shaded_region = [{"x": p["x"], "y": p["y"]} for p in unique_pts[:3]]

    result_spec = {
        "title": f"Graphical Representation of {eq_matches[0]} & {eq_matches[1]}" if len(eq_matches) >= 2 else "Graphical Representation",
        "xLabel": "x",
        "yLabel": "y",
        "lines": lines,
        "points": unique_pts,
        "nature": nature,
    }
    if shaded_region:
        result_spec["shaded_region"] = shaded_region
    return result_spec


def process_message_graphs(text: str) -> str:
    """Detects any graph/geometry code blocks or implicit graph requests in assistant text and embeds SVG."""
    if not text:
        return text

    def replace_block(match: re.Match) -> str:
        spec_content = match.group(1).strip()
        if not (spec_content.startswith("{") and spec_content.endswith("}")):
            return match.group(0)
        try:
            cleaned = re.sub(r",\s*([\]}])", r"\1", spec_content)
            spec = json.loads(cleaned)
            if spec.get("type") == "geometry" or isinstance(spec.get("elements"), list):
                svg = render_geometry_diagram(spec)
                return f"\n\n{svg}\n\n"
            if spec.get("title") or spec.get("lines") or spec.get("points") or spec.get("nature") or spec.get("xLabel"):
                svg = render_cartesian_graph(spec)
                return f"\n\n{svg}\n\n"
        except Exception:
            pass
        return match.group(0)

    pattern = re.compile(
        r"```(?:graph|plot|chart|geometry|diagram|json)?\s*([\s\S]*?)\s*```",
        re.DOTALL,
    )
    processed = pattern.sub(replace_block, text)

    # Fallback: if no SVG was embedded but the text describes a graphical solution with coordinates
    if "<svg" not in processed:
        synth = synthesize_graph_from_text(processed)
        if synth:
            try:
                svg = render_cartesian_graph(synth)
                processed = f"{processed}\n\n{svg}\n\n"
            except Exception:
                pass

    return processed
