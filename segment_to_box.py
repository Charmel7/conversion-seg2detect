#!/usr/bin/env python3
"""
segment_to_box.py

Transformer un fichier YOLOv8 segmentation (.txt) en annotation de détection
(bounding box) au format YOLOv8: "class_id x_center y_center width height".

Concrètement:
- Chaque ligne d'entrée: "class_id x1 y1 x2 y2 ... xn yn"
- Le nombre de points par ligne peut varier.
- Le script:
  1) lit le fichier ligne par ligne,
  2) détermine le nombre maximal de points,
  3) pad les lignes plus courtes avec NaN pour uniformiser la taille,
  4) utilise une logique équivalente à segment2box d’Ultralytics pour extraire les coordonnées,
  5) calcule la bounding box minimale (x_min, y_min, x_max, y_max),
  6) convertit en YOLOv8 detection: class_id x_center y_center width height
  7) écrit le résultat dans un nouveau fichier .txt (même nom, répertoire ciblé)
  
Important:
- La normalisation (x_center, y_center, width, height) est effectuée uniquement si une taille d’image est fournie
  via l’option --image-size WIDTH HEIGHT.
- Si l’image size n’est pas fournie, le script lève une erreur explicite et ne produit pas d’annotations non valides.

Usage rapide:
  python segment_to_box.py -i chemin/vers/annotations_seg.txt -o chemin/vers/annotations_det.txt --image-size 640 480

À propos: L’algorithme est robuste face aux lignes incomplètes et corrompues grâce au padding NaN et à la vérification des points valides.

Auteur: Ask AI, Codeway
Licence: MIT
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import List, Optional, Tuple

import numpy as np

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def _parse_line_to_points(line: str) -> Optional[Tuple[int, np.ndarray]]:
    """
    Parse a single line of the input file.

    Returns:
        (class_id, points) where points is an (N, 2) numpy array of coordinates
        or None if the line is invalid.

    Validation:
    - First token doit être l’identifiant de classe (int)
    - Le reste doivent être des nombres et former des paires (x, y)
    - Si le nombre de valeurs de coordonnées est impair, la dernière coordonnée est ignorée.
    """
    line = line.strip()
    if not line:
        return None

    tokens = line.split()
    if len(tokens) < 3:
        logger.warning(f"Ignoring line (too short): '{line}'")
        return None

    # Parse class_id
    try:
        class_id = int(tokens[0])
    except ValueError:
        logger.warning(f"Ignoring line (invalid class_id): '{line}'")
        return None

    # Parse coordinates (rest of tokens)
    coords_flat = []
    for t in tokens[1:]:
        try:
            coords_flat.append(float(t))
        except ValueError:
            logger.warning(f"Ignoring line (non-numeric coordinate): '{line}'")
            return None

    if len(coords_flat) < 2:
        logger.warning(f"Ignoring line (no coordinates): '{line}'")
        return None

    # If odd number of coordinates, drop the last one
    if len(coords_flat) % 2 != 0:
        logger.warning(f"Line has odd number of coordinate values; dropping the last one: '{line}'")
        coords_flat = coords_flat[:-1]

    pts = np.array(coords_flat, dtype=float).reshape(-1, 2)
    if pts.size == 0:
        return None

    return class_id, pts


def _compute_bbox_from_points(pts: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """
    Compute the axis-aligned bounding box (xmin, ymin, xmax, ymax) from a set of points.

    Args:
        pts: shape (N, 2)

    Returns:
        (xmin, ymin, xmax, ymax) or None if there are no valid points.
    """
    if pts.size == 0 or pts.shape[0] == 0:
        return None

    xmin = float(np.min(pts[:, 0]))
    ymin = float(np.min(pts[:, 1]))
    xmax = float(np.max(pts[:, 0]))
    ymax = float(np.max(pts[:, 1]))

    if xmax < xmin or ymax < ymin:
        return None

    return xmin, ymin, xmax, ymax


def _to_yolo_normalized_bbox(class_id: int,
                           bbox: Tuple[float, float, float, float],
                           image_size: Tuple[int, int]) -> Tuple[str, float, float, float, float]:
    """
    Convert a bbox in pixel coordinates to YOLOv8 detection format [class_id x_center y_center w h],
    normalised by the given image size.

    Args:
        class_id: int
        bbox: (xmin, ymin, xmax, ymax) in pixels
        image_size: (width, height)

    Returns:
        (line_str, x_center_norm, y_center_norm, w_norm, h_norm)
        line_str is the final formatted line as "class_id x_center y_center w h"
    """
    xmin, ymin, xmax, ymax = bbox
    width_img, height_img = image_size

    if width_img <= 0 or height_img <= 0:
        raise ValueError("Image size must be positive integers")

    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin

    x_center_norm = x_center / float(width_img)
    y_center_norm = y_center / float(height_img)
    w_norm = w / float(width_img)
    h_norm = h / float(height_img)

    # Clamp to [0, 1] to avoid extreme values due to numerical issues
    x_center_norm = max(0.0, min(1.0, x_center_norm))
    y_center_norm = max(0.0, min(1.0, y_center_norm))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))

    line_str = f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
    return line_str, x_center_norm, y_center_norm, w_norm, h_norm


def convert_seg_file_to_yolo(input_path: str,
                             output_path: str,
                             image_size: Optional[Tuple[int, int]] = None) -> int:
    """
    Convert a segmentation annotation file to a detection annotation file.

    Args:
        input_path: Path to the input .txt with segmentation points
        output_path: Path where to write the output .txt with detection boxes
        image_size: (width, height) to normalize coordinates; if None, raises ValueError

    Returns:
        Number of successfully converted lines (detections written)

    Behavior:
        - Reads input line by line
        - Pads each line to the maximum number of points with NaN
        - Extracts only valid (finite) coordinates
        - Computes the minimal bounding box for valid points
        - Converts to YOLOv8 detection format (normalized if image_size is provided)
    """
    if input_path is None or output_path is None:
        raise ValueError("input_path and output_path must be provided")

    if image_size is None:
        raise ValueError("image_size (width, height) must be provided to normalise to YOLO format")

    # Read all lines
    with open(input_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    parsed_entries: List[Tuple[int, np.ndarray]] = []
    max_points: int = 0

    for idx, line in enumerate(raw_lines, start=1):
        parsed = _parse_line_to_points(line)
        if parsed is None:
            # _parse_line_to_points already logs a warning
            continue
        class_id, pts = parsed
        parsed_entries.append((class_id, pts))
        max_points = max(max_points, pts.shape[0])

    if len(parsed_entries) == 0:
        logger.info("No valid lines found for conversion. Output file will be empty.")
        # Still create an empty output file to reflect the operation
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as of:
            pass
        return 0

    # Prepare output
    out_lines: List[str] = []
    converted_count = 0
    skipped_count = 0

    for class_id, pts in parsed_entries:
        # Pad to max_points with NaN
        coords_pad = np.full((max_points, 2), np.nan, dtype=float)
        coords_pad[:pts.shape[0], :] = pts

        # Valid points: both coordinates finite
        valid_mask = np.isfinite(coords_pad[:, 0]) & np.isfinite(coords_pad[:, 1])
        valid_pts = coords_pad[valid_mask]

        if valid_pts.shape[0] == 0:
            logger.warning(f"Line with class {class_id} has no valid points after padding. Skipping.")
            skipped_count += 1
            continue

        bbox = _compute_bbox_from_points(valid_pts)
        if bbox is None:
            logger.warning(f"Could not compute bbox for class {class_id}. Skipping line.")
            skipped_count += 1
            continue

        # Compute normalized YOLO bbox
        try:
            line_out, _, _, _, _ = _to_yolo_normalized_bbox(class_id, bbox, image_size)
        except Exception as exc:
            logger.warning(f"Failed to convert bbox for class {class_id}: {exc}. Skipping line.")
            skipped_count += 1
            continue

        out_lines.append(line_out)
        converted_count += 1

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as of:
        for line in out_lines:
            of.write(line + "\n")

    logger.info(f"Conversion completed: {converted_count} lines written, {skipped_count} lines skipped.")
    return converted_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLOv8 segmentation annotations to detection (bounding box) annotations."
    )
    parser.add_argument("-i", "--input", dest="input_path", required=True,
                        help="Input segmentation annotation file (.txt) with lines: 'class_id x1 y1 x2 y2 ... xn yn'")
    parser.add_argument("-o", "--output", dest="output_path", required=True,
                        help="Output detection annotation file (.txt) with lines: 'class_id x_center y_center width height' (normalized)")
    parser.add_argument("--image-size", dest="image_size", nargs=2, type=int, required=True,
                        metavar=("WIDTH", "HEIGHT"),
                        help="Image size for normalization: width height (e.g., 640 480)")
    parser.add_argument("--log-level", dest="log_level", default="INFO",
                        help="Logging level (DEBUG, INFO, WARNING, ERROR)")

    args = parser.parse_args()

    # Adjust logger level if requested
    log_level = getattr(logging, args.log_level.upper(), None)
    if isinstance(log_level, int):
        logger.setLevel(log_level)

    image_size = (int(args.image_size[0]), int(args.image_size[1]))
    try:
        converted = convert_seg_file_to_yolo(args.input_path, args.output_path, image_size=image_size)
        # Report if nothing converted but file exists
        if converted is None:
            logger.error("Conversion did not produce any output.")
    except Exception as exc:
        logger.error(f"Unexpected error during conversion: {exc}")
        raise


if __name__ == "__main__":
    main()
