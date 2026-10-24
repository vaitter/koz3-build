#!/usr/bin/env python3
"""
solution.py — инференс для КОЗ-3 (E1: рельеф).
- читает LAS из input_dir
- строит g/i хиллшейды (res=0.20) из LAS (модель обучена на них)
- YOLO-seg (best.pt) тайлами 1024×1024, overlap 30%
- маски → полигоны в EPSG:3857
- properties: class_name, region_name, markup_type="li", crs, confidence, fid
- дедуп (IoU 0.95) + фильтр площади (1 м²)
"""
import argparse, json, os, sys, time, struct, math, warnings
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import cv2
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin
from shapely.geometry import Polygon, MultiPolygon, mapping, shape
from shapely.ops import transform as shp_transform, unary_union
from shapely.validation import make_valid
from pyproj import Transformer
from ultralytics import YOLO

# ===== Константы =====
CLASS_NAMES = ["kurgany_tselye", "kurgany_povrezhdennye", "fortifikatsii",
               "gorodishcha", "arkhitektury"]
E2_CLASS_NAMES = ["finds_points", "object_poly"]
_E2_MODEL = None  # global E2 model
TILE = 1024
OVERLAP = 0.3
STRIDE = int(TILE * (1 - OVERLAP))
CONF = 0.15
IOU = 0.5
MIN_AREA_M2 = 1.0
DEDUP_IOU = 0.95
HS_RES = 0.10  # ЖЁСТКО
MASK_THRESHOLD = 0.5
MAX_CONTOURS_PER_MASK = 100
CONTOUR_EPSILON_PX = 0.5
SIMPLIFY_TOLERANCE_M = 2.0

_LAS_CLASS_BYTE = {0: 15, 1: 15, 2: 15, 3: 15, 4: 15, 5: 15,
                   6: 16, 7: 16, 8: 16, 9: 16, 10: 16}

# ===== LAS =====
def _las_header_info(path):
    with path.open("rb") as h:
        head = h.read(375)
    if len(head) < 227:
        raise RuntimeError(f"{path.name}: файл короче LAS")
    version = (head[24], head[25])
    point_offset = struct.unpack_from("<I", head, 96)[0]
    point_format = head[104]
    record_length = struct.unpack_from("<H", head, 105)[0]
    point_count = struct.unpack_from("<I", head, 107)[0]
    if version >= (1, 4) and len(head) >= 255:
        c64 = struct.unpack_from("<Q", head, 247)[0]
        if c64: point_count = c64
    max_x, min_x, max_y, min_y, max_z, min_z = struct.unpack_from("<6d", head, 179)
    if not (max_x > min_x and max_y > min_y) or point_count <= 0:
        raise RuntimeError(f"{path.name}: некорректный LAS")
    return {"version": version, "point_offset": point_offset,
            "point_format": point_format, "record_length": record_length,
            "point_count": int(point_count),
            "scale": struct.unpack_from("<3d", head, 131),
            "offset": struct.unpack_from("<3d", head, 155),
            "bounds": (min_x, min_y, max_x, max_y)}

def _iter_las_chunks(path, header, chunk_points):
    pf = header["point_format"]; rl = header["record_length"]
    if path.suffix.lower() == ".laz":
        import laspy
        with laspy.open(str(path)) as h:
            for pts in h.chunk_iterator(chunk_points):
                yield (np.asarray(pts.x, np.float64), np.asarray(pts.y, np.float64),
                       np.asarray(pts.z, np.float64), np.asarray(pts.classification))
        return
    if pf not in _LAS_CLASS_BYTE:
        raise RuntimeError(f"{path.name}: формат {pf} не поддерживается")
    dtype = np.dtype({"names": ["x","y","z","cls"],
                      "formats": ["<i4","<i4","<i4","u1"],
                      "offsets": [0,4,8,_LAS_CLASS_BYTE[pf]],
                      "itemsize": rl})
    sx, sy, sz = header["scale"]; ox, oy, oz = header["offset"]
    rem = header["point_count"]
    with path.open("rb") as h:
        h.seek(header["point_offset"])
        while rem > 0:
            n = int(min(chunk_points, rem))
            buf = h.read(n * rl)
            n = len(buf) // rl
            if n <= 0: break
            rec = np.frombuffer(buf, dtype=dtype, count=n)
            cls = np.asarray(rec["cls"])
            if pf <= 5: cls = cls & 0x1F
            yield (rec["x"].astype(np.float64)*sx+ox,
                   rec["y"].astype(np.float64)*sy+oy,
                   rec["z"].astype(np.float64)*sz+oz, cls)
            rem -= n

def _fill_nan_pyramid(arr, max_levels=16):
    cur = np.asarray(arr, np.float32); levels = []
    for _ in range(max_levels):
        if np.isfinite(cur).all() or min(cur.shape) < 2: break
        levels.append(cur)
        ph, pw = cur.shape[0] % 2, cur.shape[1] % 2
        if ph or pw: cur = np.pad(cur, ((0,ph),(0,pw)), constant_values=np.nan)
        v = np.isfinite(cur); vals = np.where(v, cur, 0.0).astype(np.float32)
        c = v.reshape(cur.shape[0]//2,2,cur.shape[1]//2,2).sum((1,3))
        s = vals.reshape(cur.shape[0]//2,2,cur.shape[1]//2,2).sum((1,3))
        cur = np.where(c > 0, s/np.maximum(c,1), np.nan).astype(np.float32)
    f = np.isfinite(cur)
    if not f.all():
        m = float(cur[f].mean()) if f.any() else 0.0
        cur = np.where(f, cur, m).astype(np.float32)
    for lv in reversed(levels):
        up = cv2.resize(cur, (lv.shape[1], lv.shape[0]), interpolation=cv2.INTER_LINEAR)
        cur = np.where(np.isfinite(lv), lv, up).astype(np.float32)
    return cur

def _convex_hull_mask(valid, max_points=1_500_000):
    h, w = valid.shape
    flat = np.flatnonzero(valid.ravel())
    if flat.size < 3: return valid.copy()
    if flat.size > max_points: flat = flat[::int(math.ceil(flat.size/max_points))]
    r, c = np.divmod(flat, w)
    pts = np.column_stack([c, r]).astype(np.int32).reshape(-1,1,2)
    m = np.zeros((h,w), np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(pts), 1)
    return m.astype(bool)

def _fill_by_delaunay(dem, valid, to_fill, max_v):
    if int(np.count_nonzero(valid)) > max_v: return None
    try:
        from scipy.interpolate import LinearNDInterpolator
        from scipy.spatial import Delaunay
    except Exception: return None
    try:
        r, c = np.nonzero(valid)
        tri = Delaunay(np.column_stack([c, r]).astype(np.float64))
        interp = LinearNDInterpolator(tri, dem[r, c])
        filled = dem.copy()
        fr, fc = np.nonzero(to_fill)
        for s in range(0, fr.size, 2_000_000):
            sl = slice(s, s+2_000_000)
            vals = interp(np.column_stack([fc[sl], fr[sl]]).astype(np.float64))
            k = np.isfinite(vals)
            filled[fr[sl][k], fc[sl][k]] = vals[k]
        return filled
    except Exception: return None

def fill_interior_holes(dem, max_v=4_000_000):
    dem = np.asarray(dem)
    valid = np.isfinite(dem)
    if valid.all(): return dem
    to_fill = _convex_hull_mask(valid) & ~valid
    if not to_fill.any(): return dem
    f = _fill_by_delaunay(dem, valid, to_fill, max_v)
    if f is None: f = np.where(to_fill, _fill_nan_pyramid(dem), dem)
    u = to_fill & ~np.isfinite(f)
    if u.any(): f = np.where(u, _fill_nan_pyramid(f), f)
    return np.asarray(f, np.float32)

def build_dem_from_las(las_path, res=0.2, ground_class=2, max_pixels=60_000_000):
    from rasterio.transform import from_origin
    header = _las_header_info(las_path)
    min_x, min_y, max_x, max_y = header["bounds"]
    def _extent(step):
        l = math.floor(min_x/step)*step; b = math.floor(min_y/step)*step
        w = int(math.ceil((max_x-l)/step))+1; h = int(math.ceil((max_y-b)/step))+1
        return l, b, w, h
    left, bottom, width, height = _extent(res)
    if max_pixels and width*height > max_pixels:
        factor = int(math.ceil(math.sqrt(width*height/max_pixels)))
        res *= factor; left, bottom, width, height = _extent(res)
    transform = from_origin(left, bottom + height*res, res, res)
    top = transform.f; cells = width*height
    sums = np.zeros(cells, np.float64); counts = np.zeros(cells, np.int32)
    selected = 0
    for x, y, z, cls in _iter_las_chunks(las_path, header, 2_000_000):
        mask = (cls == ground_class)
        if not mask.any(): continue
        x, y, z = x[mask], y[mask], z[mask]
        selected += int(x.size)
        col = np.floor((x-left)/res).astype(np.int64, copy=False)
        row = np.floor((top-y)/res).astype(np.int64, copy=False)
        inside = (col>=0)&(col<width)&(row>=0)&(row<height)
        if not inside.any(): continue
        flat = row[inside]*width + col[inside]
        vals = z[inside]
        order = np.argsort(flat, kind="stable")
        flat, vals = flat[order], vals[order]
        starts = np.flatnonzero(np.concatenate(([True], flat[1:] != flat[:-1])))
        uniq = flat[starts]
        sums[uniq] += np.add.reduceat(vals, starts)
        counts[uniq] += np.diff(np.concatenate((starts, [flat.size])))
    if selected == 0:
        raise RuntimeError(f"{las_path.name}: нет точек класса {ground_class}")
    occ = counts > 0
    dem = np.where(occ, sums/np.maximum(counts,1), np.nan)
    dem = dem.reshape(height, width).astype(np.float32)
    dem = fill_interior_holes(dem)
    return dem, transform

def hillshade_esri(dem, res_x, res_y, az=315.0, alt=45.0, zf=1.0):
    gy = -np.gradient(dem, res_y, axis=0); gx = np.gradient(dem, res_x, axis=1)
    slope = np.arctan(np.hypot(gx, gy)*zf)
    aspect = np.arctan2(-gx, -gy)
    zen = math.radians(90.0-alt); azr = math.radians(az)
    shade = 255.0*(math.cos(zen)*np.cos(slope)+math.sin(zen)*np.sin(slope)*np.cos(azr-aspect))
    return np.clip(shade, 0.0, 255.0)

def build_g_from_las(las_path, out_dir, res=0.20):
    """Строит g через hillshade_lib.make_hillshade_set (проверено: corr 0.88)."""
    import sys
    sys.path.insert(0, "/content")
    from hillshade_lib import make_hillshade_set
    out_dir.mkdir(parents=True, exist_ok=True)
    written = make_hillshade_set(
        las_path, out_dir, res=res,
        modalities=("g",), interp="mean", ground_class=2,
    )
    g_path = [p for p in written if p.name.endswith("_g.tif")][0]
    with rasterio.open(g_path) as s:
        tf = s.transform
    return g_path, tf

# ===== Маски → полигоны =====
def mask_to_polygons(mask, max_polys=MAX_CONTOURS_PER_MASK):
    m = (mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n = 0
    for c in cnts:
        if n >= max_polys: break
        s = cv2.approxPolyDP(c, CONTOUR_EPSILON_PX, True)
        if len(s) < 3: continue
        coords = [(float(p[0][0]), float(p[0][1])) for p in s]
        try:
            poly = make_valid(Polygon(coords))
            if poly.is_empty or poly.area < 0.1: continue
            if isinstance(poly, MultiPolygon):
                for sub in poly.geoms:
                    if not sub.is_empty:
                        yield sub; n += 1
            else:
                yield poly; n += 1
        except Exception: continue

# ===== Основной инференс =====
def find_las(region_dir):
    files = [p for p in region_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".las",".laz")]
    if not files: return None
    files.sort(key=lambda p: (("ground" not in p.name.lower()), len(p.name)))
    return files[0]

def load_utm_epsg(region_dir):
    p = region_dir / "UTM.json"
    if not p.exists(): return 32636
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        crs = d.get("crs","")
        if "EPSG::" in crs: return int(crs.split("::")[-1])
        utm = d.get("utm","36N")
        digits = "".join(ch for ch in utm if ch.isdigit())
        return (32600 if not utm.upper().endswith("S") else 32700) + int(digits)
    except Exception: return 32636

def process_region(model, region_dir, tmp_root, region_name):
    las = find_las(region_dir)
    features = []
    n_tiles = 0

    # ===== E1 (LAS) =====
    if las is not None:
        epsg = load_utm_epsg(region_dir)
        t_utm_3857 = Transformer.from_crs(epsg, 3857, always_xy=True)
        print(f"  [{region_name}] LAS={las.name}, EPSG={epsg}")

        t0 = time.time()
        g_tif, g_tf = build_g_from_las(las, tmp_root, res=HS_RES)
        print(f"  [{region_name}] g построен за {time.time()-t0:.1f}с")

        with rasterio.open(g_tif) as src_g:
            data = src_g.read(1)
            H, W = data.shape

        img = np.repeat(data[..., None], 3, axis=-1)
        tf = g_tf

        for y in range(0, max(1, H - TILE + 1), STRIDE):
            for x in range(0, max(1, W - TILE + 1), STRIDE):
                y_end = min(y+TILE, H); x_end = min(x+TILE, W)
                tile = img[y:y_end, x:x_end]
                if tile.size == 0: continue
                if tile.shape[0] < TILE or tile.shape[1] < TILE:
                    pad = np.full((TILE, TILE, 3), 255, np.uint8)
                    pad[:tile.shape[0], :tile.shape[1]] = tile
                    tile = pad
                tile_bgr = np.ascontiguousarray(cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
                try:
                    results = model.predict(source=tile_bgr, conf=CONF, iou=IOU, verbose=False)
                except Exception as e:
                    print(f"    predict err: {e}"); continue
                if not results: continue
                res = results[0]
                if res.masks is None or res.boxes is None: continue
                mask_data = res.masks.data.cpu().numpy()
                oh, ow = res.masks.orig_shape
                cls_arr = res.boxes.cls.cpu().numpy()
                conf_arr = res.boxes.conf.cpu().numpy()
                tile_tf = rasterio.windows.transform(Window(x, y, min(TILE, x_end-x), min(TILE, y_end-y)), tf)
                n_tiles += 1
                for idx, raw in enumerate(mask_data):
                    m = raw
                    if m.shape != (oh, ow):
                        m = cv2.resize(m, (ow, oh), interpolation=cv2.INTER_NEAREST)
                    binm = m > MASK_THRESHOLD
                    hl = min(oh, TILE); wl = min(ow, TILE)
                    if not binm[:hl, :wl].any(): continue
                    full = np.zeros((TILE, TILE), np.uint8)
                    full[:hl, :wl] = binm[:hl, :wl].astype(np.uint8)
                    cid = int(cls_arr[idx]) if idx < len(cls_arr) else 0
                    if cid >= len(CLASS_NAMES): continue
                    cname = CLASS_NAMES[cid]
                    cconf = float(conf_arr[idx]) if idx < len(conf_arr) else 0.5
                    for poly in mask_to_polygons(full):
                        try:
                            g_utm = shp_transform(lambda px, py, z=None: (tile_tf.c + px*tile_tf.a + py*tile_tf.b,
                                                                          tile_tf.f + px*tile_tf.d + py*tile_tf.e), poly)
                            g_3857 = shp_transform(t_utm_3857.transform, g_utm)
                            g_3857 = make_valid(g_3857)
                            if g_3857.is_empty: continue
                            if g_3857.geom_type == "GeometryCollection":
                                parts = [gg for gg in g_3857.geoms if gg.geom_type in ("Polygon","MultiPolygon")]
                                if not parts: continue
                                g_3857 = unary_union(parts)
                            g_3857 = g_3857.simplify(SIMPLIFY_TOLERANCE_M, preserve_topology=True)
                            if g_3857.is_empty: continue
                            area_utm = abs(g_utm.area) if not g_utm.is_empty else 0.0
                            if area_utm < MIN_AREA_M2: continue
                            geom_json = mapping(g_3857)
                            if not geom_json.get("coordinates"): continue
                            features.append({
                                "type": "Feature",
                                "properties": {
                                    "class_name": cname,
                                    "region_name": region_name,
                                    "markup_type": "li",
                                    "crs": "urn:ogc:def:crs:EPSG::3857",
                                    "original_crs": f"urn:ogc:def:crs:EPSG::{epsg}",
                                    "confidence": round(cconf, 3),
                                    "fid": 0,
                                },
                                "geometry": geom_json,
                            })
                        except Exception:
                            continue
        print(f"  [{region_name}] E1: {n_tiles} тайлов, {len(features)} фич")

    # ===== E2 (геофизика) =====
    has_geo_gt = False
    for pat in ("*MagGr*.geojson", "*EtGr*.geojson", "*TerFl*.geojson",
                "*maggr*.geojson", "*etgr*.geojson", "*terfl*.geojson"):
        if list(region_dir.rglob(pat)):
            has_geo_gt = True
            break
    geo_tifs = []
    if has_geo_gt or (las is None and any(region_dir.rglob("*.tif"))):
        for tif in region_dir.rglob("*.tif"):
            low = str(tif).lower()
            if any(m in low for m in ("lidar", "li_карт", "hillshade", "отмывк")):
                continue
            geo_tifs.append(tif)
    for pat in ("*maggr*.tif", "*etgr*.tif", "*terfl*.tif",
                "*MagGr*.tif", "*EtGr*.tif", "*TerFl*.tif",
                "*магнито*.tif", "*электро*.tif", "*термо*.tif"):
        geo_tifs.extend(region_dir.rglob(pat))
    geo_tifs = list(dict.fromkeys(geo_tifs))
    print(f"  [DEBUG] geo_tifs={len(geo_tifs)}, _E2_MODEL={_E2_MODEL is not None}")
    if geo_tifs and _E2_MODEL is not None:
        e2 = _E2_MODEL
        print(f"  [{region_name}] E2: {len(geo_tifs)} .tif")
        for tif in geo_tifs[:1]:
            try:
                with rasterio.open(tif) as src_e2:
                    W2, H2, tf2 = src_e2.width, src_e2.height, src_e2.transform
                    crs2 = src_e2.crs
                    # LOCAL_CS / None → fallback на UTM.json региона
                    if crs2 is None or "LOCAL_CS" in str(crs2).upper():
                        crs2 = f"EPSG:{load_utm_epsg(region_dir)}"
                        print(f"    [E2] crs2={src_e2.crs} → fallback {crs2}")
                    if W2 < 512 or H2 < 512:
                        print(f"    [E2] {tif.name}: {W2}x{H2} < 512, skip")
                        continue
                    data2 = src_e2.read()
                    if data2.shape[0] >= 4:
                        data2 = data2[:3]
                    elif data2.shape[0] == 1:
                        data2 = np.repeat(data2, 3, axis=0)
                    img2 = data2[:3].transpose(1, 2, 0)
                    t_utm_3857_e2 = Transformer.from_crs(crs2, 3857, always_xy=True)
                    for y2 in range(0, max(1, H2 - 512 + 1), 358):
                        for x2 in range(0, max(1, W2 - 512 + 1), 358):
                            tile2 = img2[y2:y2+512, x2:x2+512]
                            if tile2.shape[0] < 512 or tile2.shape[1] < 512: continue
                            tile2_bgr = np.ascontiguousarray(cv2.cvtColor(tile2, cv2.COLOR_RGB2BGR))
                            r2 = e2.predict(source=tile2_bgr, conf=0.4, iou=0.5, max_det=100, verbose=False)[0]
                            if r2.masks is None or r2.boxes is None: continue
                            m2_data = r2.masks.data.cpu().numpy()
                            oh2, ow2 = r2.masks.orig_shape
                            c2 = r2.boxes.cls.cpu().numpy()
                            cf2 = r2.boxes.conf.cpu().numpy()
                            tf2_tile = rasterio.windows.transform(Window(x2, y2, min(512, W2-x2), min(512, H2-y2)), tf2)
                            for idx2, raw2 in enumerate(m2_data):
                                m2 = raw2
                                if m2.shape != (oh2, ow2):
                                    m2 = cv2.resize(m2, (ow2, oh2), interpolation=cv2.INTER_NEAREST)
                                bin2 = m2 > MASK_THRESHOLD
                                hl2, wl2 = min(oh2, 512), min(ow2, 512)
                                if not bin2[:hl2, :wl2].any(): continue
                                full2 = np.zeros((512, 512), np.uint8)
                                full2[:hl2, :wl2] = bin2[:hl2, :wl2].astype(np.uint8)
                                cid2 = int(c2[idx2]) if idx2 < len(c2) else 0
                                if cid2 >= len(E2_CLASS_NAMES): continue
                                cname2 = E2_CLASS_NAMES[cid2]
                                cconf2 = float(cf2[idx2]) if idx2 < len(cf2) else 0.5
                                for poly2 in mask_to_polygons(full2):
                                    try:
                                        g_utm2 = shp_transform(lambda px, py, z=None: (
                                            tf2_tile.c + px*tf2_tile.a + py*tf2_tile.b,
                                            tf2_tile.f + px*tf2_tile.d + py*tf2_tile.e), poly2)
                                        g_3857_2 = shp_transform(t_utm_3857_e2.transform, g_utm2)
                                        g_3857_2 = make_valid(g_3857_2)
                                        if g_3857_2.is_empty: continue
                                        if g_3857_2.geom_type not in ("Polygon", "MultiPolygon"):
                                            parts2 = [gg for gg in g_3857_2.geoms if gg.geom_type in ("Polygon", "MultiPolygon")]
                                            if not parts2: continue
                                            g_3857_2 = unary_union(parts2)
                                        g_3857_2 = g_3857_2.simplify(SIMPLIFY_TOLERANCE_M, preserve_topology=True)
                                        if g_3857_2.is_empty: continue
                                        features.append({
                                            "type": "Feature",
                                            "properties": {
                                                "class_name": cname2,
                                                "region_name": region_name,
                                                "markup_type": "maggr",
                                                "crs": "urn:ogc:def:crs:EPSG::3857",
                                                "original_crs": crs2,
                                                "confidence": round(cconf2, 3),
                                                "fid": 0,
                                            },
                                            "geometry": mapping(g_3857_2),
                                        })
                                    except Exception:
                                        continue
            except Exception as e:
                print(f"  [{region_name}] E2 {tif.name}: {e}")

    print(f"  [{region_name}] тайлов E1: {n_tiles}, фич всего: {len(features)}")
    return features

def dedup(features, thr=DEDUP_IOU):
    groups = defaultdict(list)
    for f in features:
        groups[f["properties"]["class_name"]].append(f)
    out = []
    for items in groups.values():
        uniq = []
        for feat in items:
            try: poly = make_valid(shape(feat["geometry"]))
            except Exception: continue
            if poly.is_empty: continue
            dup = False
            for sp, _ in uniq:
                try:
                    inter = poly.intersection(sp).area
                    union = poly.union(sp).area
                    if union and inter/union > thr:
                        dup = True; break
                except Exception: pass
            if not dup: uniq.append((poly, feat))
        out.extend(f for _, f in uniq)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--model", default=None)
    ap.add_argument("--tmp", default="/tmp/hs")
    args = ap.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model) if args.model else Path(__file__).parent / "best.pt"
    if not model_path.exists():
        print(f"Нет весов: {model_path}"); sys.exit(1)
    tmp_root = Path(args.tmp); tmp_root.mkdir(parents=True, exist_ok=True)

    print(f"Загрузка модели {model_path}...")
    model = YOLO(str(model_path))
    print(f"Классы модели: {model.names}")
    global _E2_MODEL
    e2_path = Path(__file__).parent / "best_e2.pt"
    if e2_path.exists():
        print(f"Загрузка E2: {e2_path}")
        try:
            _E2_MODEL = YOLO(str(e2_path))
            print(f"E2 classes: {_E2_MODEL.names}")
        except Exception as e:
            print(f"E2 не загружен: {e}")
    else:
        print(f"⚠️ E2 не найден: {e2_path}")


    regions = sorted(d for d in input_path.iterdir() if d.is_dir() and not d.name.startswith("."))
    if not regions:
        regions = [input_path]
    print(f"Регионов: {len(regions)}")

    all_features = []
    for reg in regions:
        try:
            feats = process_region(model, reg, tmp_root, reg.name)
            all_features.extend(feats)
        except Exception as e:
            print(f"  [{reg.name}] ОШИБКА: {e}")
            import traceback; traceback.print_exc()

    print(f"\nВсего фич: {len(all_features)}")
    features = dedup(all_features)
    print(f"После дедупа: {len(features)}")

    for i, f in enumerate(features):
        f["properties"]["fid"] = i

    out_file = output_path / "result.geojson"
    with out_file.open("w", encoding="utf-8") as h:
        json.dump({"type": "FeatureCollection", "features": features}, h, ensure_ascii=False)
    print(f"\n✓ Сохранено: {out_file} ({len(features)} фич)")

if __name__ == "__main__":
    main()
