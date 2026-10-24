
"""Hillshade из LAS — копия функций из solution.py (без изменений логики)."""
import math, struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from affine import Affine

_LAS_CLASS_BYTE = {0: 15, 1: 15, 2: 15, 3: 15, 4: 15, 5: 15,
                   6: 16, 7: 16, 8: 16, 9: 16, 10: 16}
HOT_TO_COLD_VALUE = 209.0 / 255.0
HOT_TO_COLD_PHASE = 0.75
DEFAULT_CYCLES_C = 3.0
HILLSHADE_MODALITIES = ("c", "ch", "g", "i")


def _las_header_info(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        head = handle.read(375)
    if len(head) < 227:
        raise RuntimeError(f"{path.name}: файл короче заголовка LAS")
    version = (head[24], head[25])
    point_offset = struct.unpack_from("<I", head, 96)[0]
    point_format = head[104]
    record_length = struct.unpack_from("<H", head, 105)[0]
    point_count = struct.unpack_from("<I", head, 107)[0]
    if version >= (1, 4) and len(head) >= 255:
        count_64 = struct.unpack_from("<Q", head, 247)[0]
        if count_64:
            point_count = count_64
    max_x, min_x, max_y, min_y, max_z, min_z = struct.unpack_from("<6d", head, 179)
    if not (max_x > min_x and max_y > min_y) or point_count <= 0:
        raise RuntimeError(f"{path.name}: некорректный заголовок LAS")
    return {
        "version": version, "point_offset": point_offset,
        "point_format": point_format, "record_length": record_length,
        "point_count": int(point_count),
        "scale": struct.unpack_from("<3d", head, 131),
        "offset": struct.unpack_from("<3d", head, 155),
        "bounds": (min_x, min_y, max_x, max_y),
        "z_range": (min_z, max_z),
    }


def _iter_las_chunks(path: Path, header: Dict[str, Any], chunk_points: int):
    point_format = header["point_format"]
    record_length = header["record_length"]
    if path.suffix.lower() == ".laz":
        import laspy
        with laspy.open(str(path)) as handle:
            for points in handle.chunk_iterator(chunk_points):
                yield (np.asarray(points.x, dtype=np.float64),
                       np.asarray(points.y, dtype=np.float64),
                       np.asarray(points.z, dtype=np.float64),
                       np.asarray(points.classification))
        return
    if point_format not in _LAS_CLASS_BYTE:
        raise RuntimeError(f"{path.name}: формат точек {point_format} не поддерживается")
    dtype = np.dtype({
        "names": ["x", "y", "z", "cls"],
        "formats": ["<i4", "<i4", "<i4", "u1"],
        "offsets": [0, 4, 8, _LAS_CLASS_BYTE[point_format]],
        "itemsize": record_length,
    })
    scale_x, scale_y, scale_z = header["scale"]
    off_x, off_y, off_z = header["offset"]
    remaining = header["point_count"]
    with path.open("rb") as handle:
        handle.seek(header["point_offset"])
        while remaining > 0:
            count = int(min(chunk_points, remaining))
            buffer = handle.read(count * record_length)
            count = len(buffer) // record_length
            if count <= 0:
                break
            records = np.frombuffer(buffer, dtype=dtype, count=count)
            classification = np.asarray(records["cls"])
            if point_format <= 5:
                classification = classification & 0x1F
            yield (records["x"].astype(np.float64) * scale_x + off_x,
                   records["y"].astype(np.float64) * scale_y + off_y,
                   records["z"].astype(np.float64) * scale_z + off_z,
                   classification)
            remaining -= count


def _fill_nan_pyramid(array, max_levels=16):
    current = np.asarray(array, dtype=np.float32)
    levels = []
    for _ in range(max_levels):
        if np.isfinite(current).all() or min(current.shape) < 2:
            break
        levels.append(current)
        pad_h, pad_w = current.shape[0] % 2, current.shape[1] % 2
        if pad_h or pad_w:
            current = np.pad(current, ((0, pad_h), (0, pad_w)),
                             mode="constant", constant_values=np.nan)
        valid = np.isfinite(current)
        values = np.where(valid, current, 0.0).astype(np.float32)
        counts = valid.reshape(current.shape[0] // 2, 2,
                               current.shape[1] // 2, 2).sum(axis=(1, 3))
        sums = values.reshape(current.shape[0] // 2, 2,
                              current.shape[1] // 2, 2).sum(axis=(1, 3))
        current = np.where(counts > 0, sums / np.maximum(counts, 1),
                           np.nan).astype(np.float32)
    finite = np.isfinite(current)
    if not finite.all():
        mean = float(current[finite].mean()) if finite.any() else 0.0
        current = np.where(finite, current, mean).astype(np.float32)
    for level in reversed(levels):
        upscaled = cv2.resize(current, (level.shape[1], level.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        current = np.where(np.isfinite(level), level, upscaled).astype(np.float32)
    return current


def _convex_hull_mask(valid, max_points=1_500_000):
    height, width = valid.shape
    flat = np.flatnonzero(valid.ravel())
    if flat.size < 3:
        return valid.copy()
    if flat.size > max_points:
        flat = flat[::int(math.ceil(flat.size / max_points))]
    rows, cols = np.divmod(flat, width)
    points = np.column_stack([cols, rows]).astype(np.int32).reshape(-1, 1, 2)
    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 1)
    return mask.astype(bool)


def _fill_by_delaunay(dem, valid, to_fill, max_vertices):
    if int(np.count_nonzero(valid)) > max_vertices:
        return None
    try:
        from scipy.interpolate import LinearNDInterpolator
        from scipy.spatial import Delaunay
    except Exception:
        return None
    try:
        rows, cols = np.nonzero(valid)
        triangulation = Delaunay(np.column_stack([cols, rows]).astype(np.float64))
        interpolator = LinearNDInterpolator(triangulation, dem[rows, cols])
        filled = dem.copy()
        fill_rows, fill_cols = np.nonzero(to_fill)
        chunk = 2_000_000
        for start in range(0, fill_rows.size, chunk):
            window = slice(start, start + chunk)
            values = interpolator(
                np.column_stack([fill_cols[window], fill_rows[window]]).astype(np.float64)
            )
            known = np.isfinite(values)
            filled[fill_rows[window][known], fill_cols[window][known]] = values[known]
        return filled
    except Exception:
        return None


def fill_interior_holes(dem, max_vertices=4_000_000):
    dem = np.asarray(dem)
    if dem.ndim != 2:
        raise ValueError(f"DEM должен быть двумерным, получено {dem.shape}")
    valid = np.isfinite(dem)
    if valid.all():
        return dem
    to_fill = _convex_hull_mask(valid) & ~valid
    if not to_fill.any():
        return dem
    filled = _fill_by_delaunay(dem, valid, to_fill, max_vertices)
    if filled is None:
        filled = np.where(to_fill, _fill_nan_pyramid(dem), dem)
    unresolved = to_fill & ~np.isfinite(filled)
    if unresolved.any():
        filled = np.where(unresolved, _fill_nan_pyramid(filled), filled)
    return np.asarray(filled, dtype=np.float32)


def build_dem_from_las(las_path, res=0.1, interp="mean", ground_class=2,
                       use_all=False, max_pixels=60_000_000, chunk_points=2_000_000):
    from rasterio.transform import from_origin
    if interp not in ("mean", "max"):
        raise ValueError(f"interp={interp!r}")
    header = _las_header_info(las_path)
    min_x, min_y, max_x, max_y = header["bounds"]

    def _extent(step):
        left = math.floor(min_x / step) * step
        bottom = math.floor(min_y / step) * step
        width = int(math.ceil((max_x - left) / step)) + 1
        height = int(math.ceil((max_y - bottom) / step)) + 1
        return left, bottom, width, height

    left, bottom, width, height = _extent(res)
    if max_pixels and width * height > max_pixels:
        factor = int(math.ceil(math.sqrt(width * height / max_pixels)))
        res = res * factor
        left, bottom, width, height = _extent(res)
    transform = from_origin(left, bottom + height * res, res, res)
    top = transform.f
    cells = width * height

    def _accumulate(use_all_points):
        sums = np.zeros(cells, dtype=np.float64) if interp == "mean" else None
        peaks = np.full(cells, -np.inf, dtype=np.float32) if interp == "max" else None
        counts = np.zeros(cells, dtype=np.int32)
        classes = {}
        selected = 0
        for x, y, z, classification in _iter_las_chunks(las_path, header, chunk_points):
            unique, freq = np.unique(classification, return_counts=True)
            for value, amount in zip(unique, freq):
                classes[int(value)] = classes.get(int(value), 0) + int(amount)
            mask = np.ones(classification.shape, dtype=bool) if use_all_points \
                else (classification == ground_class)
            if not mask.any():
                continue
            x, y, z = x[mask], y[mask], z[mask]
            selected += int(x.size)
            col = np.floor((x - left) / res).astype(np.int64, copy=False)
            row = np.floor((top - y) / res).astype(np.int64, copy=False)
            inside = (col >= 0) & (col < width) & (row >= 0) & (row < height)
            if not inside.any():
                continue
            flat = row[inside] * width + col[inside]
            values = z[inside]
            order = np.argsort(flat, kind="stable")
            flat, values = flat[order], values[order]
            starts = np.flatnonzero(np.concatenate(([True], flat[1:] != flat[:-1])))
            uniq = flat[starts]
            if interp == "max":
                peaks[uniq] = np.maximum(peaks[uniq], np.maximum.reduceat(values, starts))
            else:
                sums[uniq] += np.add.reduceat(values, starts)
            counts[uniq] += np.diff(np.concatenate((starts, [flat.size])))
        return sums, counts, peaks, classes, selected

    sums, counts, peaks, classes, selected = _accumulate(use_all)
    if selected == 0 and not use_all:
        del sums, counts, peaks
        sums, counts, peaks, classes, selected = _accumulate(True)
    if selected == 0:
        raise RuntimeError(f"{las_path.name}: не удалось выбрать ни одной точки")

    occupied = counts > 0
    if interp == "max":
        dem = np.where(occupied, peaks, np.nan)
        del peaks
    else:
        dem = np.where(occupied, sums / np.maximum(counts, 1), np.nan)
        del sums
    del counts
    dem = dem.reshape(height, width).astype(np.float32)
    holes = int(np.count_nonzero(~occupied))
    dem = fill_interior_holes(dem)
    stats = {"res": res, "points": selected, "classes": classes,
             "holes": holes, "transform": transform}
    return dem, transform, stats


def hillshade_esri(dem, res_x, res_y, azimuth=315.0, altitude=45.0, z_factor=1.0):
    gy_north = -np.gradient(dem, res_y, axis=0)
    gx_east = np.gradient(dem, res_x, axis=1)
    slope = np.arctan(np.hypot(gx_east, gy_north) * z_factor)
    aspect = np.arctan2(-gx_east, -gy_north)
    zenith = math.radians(90.0 - altitude)
    azimuth_rad = math.radians(azimuth)
    shade = 255.0 * (math.cos(zenith) * np.cos(slope)
                     + math.sin(zenith) * np.sin(slope) * np.cos(azimuth_rad - aspect))
    return np.clip(shade, 0.0, 255.0)


def hsv_to_rgb_u8(hue, sat, val):
    h = (np.asarray(hue) % 1.0) * 6.0
    i = np.floor(h).astype(np.int32) % 6
    f = h - np.floor(h)
    p = val * (1.0 - sat)
    q = val * (1.0 - sat * f)
    t = val * (1.0 - sat * (1.0 - f))
    conds = [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5]
    r = np.select(conds, [val, q, p, p, t, val])
    g = np.select(conds, [t, val, val, q, p, p])
    b = np.select(conds, [p, p, t, val, val, q])
    rgb = np.stack([r, g, b]).astype(np.float32)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def _open_writer(path, height, width, count, transform, crs):
    import rasterio
    profile = {"driver": "GTiff", "height": height, "width": width,
               "count": count, "dtype": "uint8", "transform": transform, "crs": crs}
    return rasterio.open(path, "w", **profile)


def _write_gray_pair(out_dir, name, filled, valid, step, transform, crs, tags, block_rows=4096):
    tags = [tag for tag in ("g", "i") if tag in set(tags)]
    if not tags:
        return []
    height, width = filled.shape
    paths = {tag: out_dir / f"{name}_{tag}.tif" for tag in tags}
    writers = {tag: _open_writer(paths[tag], height, width, 1, transform, crs) for tag in tags}
    try:
        for top in range(0, height, block_rows):
            bottom = min(top + block_rows, height)
            lo, hi = max(0, top - 1), min(height, bottom + 1)
            shade = np.rint(hillshade_esri(filled[lo:hi], step, step)).astype(np.uint8)
            shade = shade[top - lo:top - lo + (bottom - top)]
            mask = ~valid[top:bottom]
            window = ((top, bottom), (0, width))
            if "g" in writers:
                gray = shade.copy()
                gray[mask] = 255
                writers["g"].write(gray, 1, window=window)
            if "i" in writers:
                inverted = (255 - shade.astype(np.int16)).astype(np.uint8)
                inverted[mask] = 255
                writers["i"].write(inverted, 1, window=window)
    finally:
        for writer in writers.values():
            writer.close()
    return [paths[tag] for tag in tags]


def _write_palette(out_dir, name, tag, filled, valid, z_min, z_range, cycles,
                   transform, crs, block_rows=4096):
    height, width = filled.shape
    path = out_dir / f"{name}_{tag}.tif"
    with _open_writer(path, height, width, 3, transform, crs) as writer:
        for top in range(0, height, block_rows):
            bottom = min(top + block_rows, height)
            block = filled[top:bottom]
            t = (block - z_min) / z_range
            hue = (HOT_TO_COLD_PHASE - cycles * t) % 1.0
            value = np.full(t.shape, HOT_TO_COLD_VALUE, dtype=np.float32)
            rgb = hsv_to_rgb_u8(hue, np.ones_like(t), value)
            rgb[:, ~valid[top:bottom]] = 0
            writer.write(rgb, window=((top, bottom), (0, width)))
    return path


def make_hillshade_set(las_path, out_dir, res=0.1, modalities=HILLSHADE_MODALITIES,
                       interp="mean", ground_class=2, use_all=False,
                       max_pixels=60_000_000, region_crs=None,
                       cycles_c=DEFAULT_CYCLES_C, cycles_ch=None):
    modalities = [tag for tag in HILLSHADE_MODALITIES if tag in set(modalities)]
    if not modalities:
        raise ValueError("не выбрано ни одной модальности хиллшейдов")
    out_dir.mkdir(parents=True, exist_ok=True)
    dem, transform, stats = build_dem_from_las(
        las_path, res=res, interp=interp, ground_class=ground_class,
        use_all=use_all, max_pixels=max_pixels,
    )
    valid = np.isfinite(dem)
    if not valid.any():
        raise RuntimeError(f"{las_path.name}: DEM пуст")
    z_min = float(np.nanmin(dem))
    z_max = float(np.nanmax(dem))
    z_range = max(z_max - z_min, 1e-9)
    step = float(transform.a)
    filled = np.where(valid, dem, z_min)
    name = las_path.stem.replace("_La_точки", "").replace("_La", "").replace("_ground", "")
    written = []
    written.extend(_write_gray_pair(out_dir, name, filled, valid, step, transform,
                                    region_crs, modalities))
    cycles = {
        "c": float(cycles_c),
        "ch": float(cycles_ch) if cycles_ch else float(max(1, round(2.0 * z_range))),
    }
    for tag in ("c", "ch"):
        if tag not in modalities:
            continue
        written.append(_write_palette(out_dir, name, tag, filled, valid, z_min, z_range,
                                      cycles[tag], transform, region_crs))
    written.sort(key=lambda path: HILLSHADE_MODALITIES.index(path.stem.rsplit("_", 1)[-1]))
    return written
