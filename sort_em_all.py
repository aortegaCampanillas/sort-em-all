#!/usr/bin/env python3
"""Resuelve puzles de "Sort 'em All" (ordenar colores en botellas) a partir de una captura.

Uso:
    python sort_em_all.py captura.png [--capacity N] [--debug salida.png]

Las botellas se numeran de izquierda a derecha y de arriba a abajo, empezando en 1.
"""
import argparse
import heapq
import sys
from collections import deque

import numpy as np
from PIL import Image, ImageDraw

# --------------------------------------------------------------------------- #
# Detección
# --------------------------------------------------------------------------- #

DOWNSAMPLE = 4          # factor de reducción para buscar manchas de líquido
COLOR_MERGE_DIST = 45   # distancia RGB máxima para considerar dos capas del mismo color

NAMED_COLORS = {
    "rojo": (205, 61, 45),
    "naranja": (233, 130, 54),
    "amarillo": (251, 235, 116),
    "verde claro": (110, 205, 110),
    "verde oscuro": (57, 129, 35),
    "azul": (79, 116, 223),
    "celeste": (82, 181, 233),
    "morado": (150, 90, 200),
    "rosa": (228, 140, 233),
    "beige": (214, 192, 166),
    "gris": (128, 128, 128),
    "marrón": (120, 70, 40),
    "blanco": (240, 240, 240),
    "negro": (30, 30, 30),
    "turquesa": (60, 200, 190),
    "fucsia": (230, 50, 140),
}


def liquid_mask(rgb):
    """True donde el píxel parece líquido (ni madera, ni cristal, ni contorno blanco)."""
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = arr.max(axis=-1)
    mn = arr.min(axis=-1)
    v = mx
    s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    delta = np.maximum(mx - mn, 1e-6)
    h = np.where(mx == r, ((g - b) / delta) % 6,
                 np.where(mx == g, (b - r) / delta + 2, (r - g) / delta + 4)) * 60
    woodish = (h >= 12) & (h <= 40) & (v < 0.8) & (s > 0.2)
    dark = v < 0.35
    whiteish = (s < 0.1) & (v > 0.85)
    greyish = s < 0.12
    return ~(woodish | dark | whiteish | greyish)


def is_liquid_color(c):
    return bool(liquid_mask(np.array([[c]], dtype=np.uint8))[0, 0])


def label_blobs(mask):
    """Componentes conexas (4-vecindad) de una máscara booleana. Devuelve bounding boxes."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    blobs = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        q = deque([(y0, x0)])
        n, ymin, ymax, xmin, xmax = 0, y0, y0, x0, x0
        while q:
            y, x = q.popleft()
            n += 1
            ymin, ymax = min(ymin, y), max(ymax, y)
            xmin, xmax = min(xmin, x), max(xmax, x)
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        blobs.append((xmin, ymin, xmax + 1, ymax + 1, n))
    return blobs


def find_liquid_columns(rgb):
    """Localiza las botellas con líquido: manchas altas y casi rectangulares."""
    small = rgb[::DOWNSAMPLE, ::DOWNSAMPLE]
    mask = liquid_mask(small)
    img_w = small.shape[1]
    cands = []
    for x0, y0, x1, y1, n in label_blobs(mask):
        w, h = x1 - x0, y1 - y0
        if w < img_w * 0.04 or w > img_w * 0.25 or h < w * 0.6:
            continue
        if n / (w * h) < 0.75:   # hojas, iconos y demás formas irregulares
            continue
        cands.append([v * DOWNSAMPLE for v in (x0, y0, x1, y1)])
    if not cands:
        return []
    # Descarta manchas de anchura muy distinta a la mediana (botones, anuncios...)
    med_w = np.median([c[2] - c[0] for c in cands])
    return [c for c in cands if abs((c[2] - c[0]) - med_w) < med_w * 0.3]


def group_rows(boxes):
    """Agrupa las cajas en filas según la parte inferior del líquido."""
    boxes = sorted(boxes, key=lambda b: b[3])
    rows = []
    for b in boxes:
        if rows and abs(rows[-1][0][3] - b[3]) < (b[2] - b[0]) * 0.5:
            rows[-1].append(b)
        else:
            rows.append([b])
    return [sorted(r, key=lambda b: b[0]) for r in rows]


def has_bottle_outline(rgb, x, y_top, y_bottom):
    """Comprueba si hay una botella (vacía) en la columna x buscando el contorno blanco."""
    col = rgb[max(0, y_top):min(rgb.shape[0], y_bottom), max(0, x - 3):x + 4].astype(int)
    white = (col.min(axis=-1) > 215) & (col.max(axis=-1) - col.min(axis=-1) < 30)
    return white.any(axis=1).sum() >= 3


def column_profile(rgb, x0, x1, y0, y1):
    """Color medio por fila en la franja central de una botella."""
    w = x1 - x0
    band = rgb[y0:y1, x0 + int(w * 0.35):x1 - int(w * 0.35)].astype(float)
    return np.median(band, axis=1)


def segment_heights(profile):
    """Alturas de los tramos de color (cortes en cambios bruscos), sin el tramo inferior."""
    diff = np.abs(np.diff(profile, axis=0)).sum(axis=1)
    cuts = [0] + [i + 1 for i in np.nonzero(diff > 40)[0]] + [len(profile)]
    segs = [[a, b] for a, b in zip(cuts, cuts[1:]) if b - a > 8]
    # Une tramos contiguos de color parecido (brillos y degradados de la botella)
    merged = []
    for a, b in segs:
        c = np.median(profile[a:b], axis=0)
        if merged and np.linalg.norm(c - merged[-1][2]) < COLOR_MERGE_DIST:
            merged[-1][1] = b
        else:
            merged.append([a, b, c])
    liquid = [(a, b) for a, b, c in merged if is_liquid_color(tuple(int(v) for v in c))]
    return [b - a for a, b in liquid[:-1]]   # el tramo del fondo se deforma por la curva


def detect(image_path, capacity=None):
    img = Image.open(image_path).convert("RGB")
    rgb = np.asarray(img)
    boxes = find_liquid_columns(rgb)
    if not boxes:
        raise SystemExit("No se han encontrado botellas en la imagen.")

    bottles = []   # (x0, x1, y_top_full, y_bottom)
    for row in group_rows(boxes):
        bottom = int(np.median([b[3] for b in row]))
        top = min(b[1] for b in row)
        width = int(np.median([b[2] - b[0] for b in row]))
        centers = [(b[0] + b[2]) // 2 for b in row]
        # Separación entre botellas: la mínima entre vecinas; se usa para buscar vacías
        gaps = np.diff(centers)
        pitch = int(min(gaps)) if len(gaps) else None
        if pitch is None:
            other = [int(min(np.diff([(b[0] + b[2]) // 2 for b in r])))
                     for r in group_rows(boxes) if len(r) > 1]
            pitch = other[0] if other else width * 3 // 2
        xs = set(centers)
        search_top, search_bottom = top - width, bottom + width // 3
        for direction in (-1, 1):
            x = (min(xs) if direction < 0 else max(xs)) + direction * pitch
            while width // 2 < x < rgb.shape[1] - width // 2:
                if has_bottle_outline(rgb, x, search_top, search_bottom):
                    xs.add(x)
                x += direction * pitch
        # Huecos intermedios (botellas vacías entre otras llenas)
        for a, b in zip(sorted(xs), sorted(xs)[1:]):
            k = round((b - a) / pitch)
            for i in range(1, k):
                x = a + i * (b - a) // k
                if has_bottle_outline(rgb, x, search_top, search_bottom):
                    xs.add(x)
        for cx in sorted(xs):
            bottles.append((cx - width // 2, cx + width // 2, top, bottom))

    # Capacidad: altura de una botella llena / altura de la capa más pequeña
    if capacity is None:
        heights = []
        for x0, x1, top, bottom in bottles:
            heights += segment_heights(column_profile(rgb, x0, x1, top, bottom))
        heights = [h for h in heights if h > 0]
        unit = min(heights) if heights else (bottles[0][3] - bottles[0][2])
        capacity = max(1, round((bottles[0][3] - bottles[0][2]) / unit))

    # Muestreo del color en el centro de cada hueco
    slot_colors = []
    for x0, x1, top, bottom in bottles:
        slot_h = (bottom - top) / capacity
        w = x1 - x0
        colors = []
        for i in range(capacity):                  # de abajo arriba
            cy = int(bottom - (i + 0.5) * slot_h)
            patch = rgb[cy - int(slot_h * 0.15):cy + int(slot_h * 0.15),
                        x0 + int(w * 0.3):x1 - int(w * 0.3)].reshape(-1, 3)
            c = tuple(int(v) for v in np.median(patch, axis=0))
            if not is_liquid_color(c):
                break
            colors.append(c)
        slot_colors.append(colors)

    # Agrupa colores parecidos en una paleta
    palette = []
    state = []
    for colors in slot_colors:
        bottle = []
        for c in colors:
            for idx, p in enumerate(palette):
                if sum((a - b) ** 2 for a, b in zip(c, p)) ** 0.5 < COLOR_MERGE_DIST:
                    break
            else:
                palette.append(c)
                idx = len(palette) - 1
            bottle.append(idx)
        state.append(tuple(bottle))
    return img, bottles, capacity, palette, tuple(state)


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #

def top_run(b):
    c = b[-1]
    n = 1
    while n < len(b) and b[-1 - n] == c:
        n += 1
    return c, n


def is_solved(state, capacity):
    return all(not b or (len(b) == capacity and len(set(b)) == 1) for b in state)


def heuristic(state):
    """Cota inferior de movimientos: cada tramo que no está en el fondo debe moverse al
    menos una vez, y de los colores que están en el fondo de varias botellas, todos
    menos uno también."""
    h = 0
    bottoms = {}
    for b in state:
        if not b:
            continue
        runs = 1 + sum(1 for i in range(1, len(b)) if b[i] != b[i - 1])
        h += runs - 1
        bottoms[b[0]] = bottoms.get(b[0], 0) + 1
    return h + sum(n - 1 for n in bottoms.values())


def moves_from(state, capacity):
    for i, src in enumerate(state):
        if not src:
            continue
        color, n = top_run(src)
        if n == len(src) and len(src) == capacity:
            continue                                  # botella ya completa
        tried_empty = False
        for j, dst in enumerate(state):
            if i == j or len(dst) == capacity:
                continue
            if dst:
                if dst[-1] != color:
                    continue
            else:
                if n == len(src) or tried_empty:      # mover a vacía no aporta nada
                    continue
                tried_empty = True
            k = min(n, capacity - len(dst))
            new = list(state)
            new[i] = src[:-k]
            new[j] = dst + (color,) * k
            yield (i, j, color, k), tuple(new)


def solve(start, capacity, max_nodes=2_000_000):
    key = lambda s: tuple(sorted(s))
    counter = 0
    open_heap = [(heuristic(start), 0, counter, start)]
    parent = {key(start): (None, None)}
    g_cost = {key(start): 0}
    while open_heap:
        _, g, _, state = heapq.heappop(open_heap)
        k = key(state)
        if g > g_cost[k]:
            continue
        if is_solved(state, capacity):
            path = []
            while True:
                prev, move = parent[key(state)]
                if prev is None:
                    return path[::-1]
                path.append(move)
                state = prev
        if len(g_cost) > max_nodes:
            break
        for move, nxt in moves_from(state, capacity):
            nk = key(nxt)
            if nk not in g_cost or g + 1 < g_cost[nk]:
                g_cost[nk] = g + 1
                parent[nk] = (state, move)
                counter += 1
                heapq.heappush(open_heap, (g + 1 + heuristic(nxt), g + 1, counter, nxt))
    return None


# --------------------------------------------------------------------------- #
# Salida
# --------------------------------------------------------------------------- #

def color_name(c):
    return min(NAMED_COLORS, key=lambda n: sum((a - b) ** 2 for a, b in zip(c, NAMED_COLORS[n])))


def swatch(c, text="    "):
    r, g, b = c
    return f"\x1b[48;2;{r};{g};{b}m{text}\x1b[0m"


def print_state(state, palette, capacity):
    print("Estado inicial (fondo → boca):")
    for i, b in enumerate(state, 1):
        cells = "".join(swatch(palette[c], "  ") for c in b) + "··" * (capacity - len(b))
        print(f"  Botella {i:>2}: {cells}")
    print()


def save_debug(img, bottles, capacity, state, palette, path):
    out = img.copy()
    d = ImageDraw.Draw(out)
    for i, ((x0, x1, top, bottom), b) in enumerate(zip(bottles, state), 1):
        d.rectangle([x0, top, x1, bottom], outline=(255, 0, 255), width=3)
        slot_h = (bottom - top) / capacity
        for s, c in enumerate(b):
            cy = bottom - (s + 0.5) * slot_h
            d.rectangle([x1 - 25, cy - 10, x1 - 5, cy + 10], fill=palette[c], outline=(0, 0, 0))
        d.text((x0 + 5, top - 40), str(i), fill=(255, 255, 255))
    out.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("imagen", help="captura del nivel")
    ap.add_argument("--capacity", type=int, help="capas por botella (se detecta sola si se omite)")
    ap.add_argument("--debug", metavar="PNG", help="guarda la imagen con lo detectado dibujado encima")
    args = ap.parse_args()

    img, bottles, capacity, palette, state = detect(args.imagen, args.capacity)
    if args.debug:
        save_debug(img, bottles, capacity, state, palette, args.debug)

    print(f"{len(bottles)} botellas, {len(palette)} colores, capacidad {capacity}\n")
    print_state(state, palette, capacity)

    counts = {}
    for b in state:
        for c in b:
            counts[c] = counts.get(c, 0) + 1
    wrong = {c: n for c, n in counts.items() if n != capacity}
    if wrong:
        print("Aviso: estos colores no tienen exactamente", capacity, "capas; "
              "revisa la detección con --debug:")
        for c, n in wrong.items():
            print(f"  {swatch(palette[c], '  ')} {color_name(palette[c])}: {n}")
        print()

    moves = solve(state, capacity)
    if moves is None:
        print("No se ha encontrado solución.")
        sys.exit(1)

    print(f"Solución en {len(moves)} movimientos:\n")
    print("   #  Color                 Origen  →  Destino")
    for n, (i, j, c, k) in enumerate(moves, 1):
        name = color_name(palette[c]) + (f" x{k}" if k > 1 else "")
        print(f"  {n:>2}  {swatch(palette[c])} {name:<16}  {i + 1:>4}   →  {j + 1:>4}")


if __name__ == "__main__":
    main()
