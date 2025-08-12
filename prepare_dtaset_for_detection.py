#!/usr/bin/env python3
"""
prepare_dataset_for_detection.py

Parcourt un dossier contenant des annotations YOLOv8 segmentation (*.txt),
et produit un nouveau dossier avec les annotations converties en détection
(au format YOLOv8 : class_id x_center y_center width height).

Fonctionnalités:
- Parcourt récursivement tous les fichiers .txt dans le dossier d'entrée
- Appelle la logique de conversion du script segment_to_box.py pour chaque fichier
- Crée une structure miroir dans le dossier de sortie avec les mêmes noms de fichier
- Optionnel: copie les images associées (mêmes noms de base) si disponibles
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple, List

# Import de la fonction de conversion du premier script
try:
    from segment_to_box import convert_seg_file_to_yolo
except Exception as e:
    raise SystemExit(f"Unable to import convert_seg_file_to_yolo from segment_to_box.py: {e}")


def _find_corresponding_image(input_txt_path: Path, image_extensions: List[str]) -> Optional[Path]:
    """
    Cherche une image associée dans le même répertoire que le fichier .txt.
    Retourne le chemin de l'image si elle existe, sinon None.
    """
    base = input_txt_path.with_suffix("")
    dirpath = input_txt_path.parent
    for ext in image_extensions:
        candidate = dirpath / (input_txt_path.stem + ext)
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Convertir tous les fichiers segmentation .txt d'un dossier en détection .txt, "
                    "en utilisant segment_to_box.convert_seg_file_to_yolo."
    )
    parser.add_argument("--input-dir", "-i", required=True, help="Dossier source contenant les annotations .txt (segmentation)")
    parser.add_argument("--output-dir", "-o", required=True, help="Dossier destination pour les annotations converties de détection")
    parser.add_argument("--image-size", nargs=2, type=int, required=True,
                        metavar=("WIDTH", "HEIGHT"),
                        help="Image size used for normalization: width height (e.g., 640 480)")
    parser.add_argument("--copy-images", action="store_true",
                        help="Optionnel: copier les images associées si trouvées.")
    parser.add_argument("--log-level", default="INFO", help="Niveau de journalisation (DEBUG, INFO, WARNING, ERROR)")

    args = parser.parse_args()

    # Configure logging
    log_level = getattr(logging, args.log_level.upper(), None)
    if isinstance(log_level, int):
        logging.getLogger().setLevel(log_level)

    input_root = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    image_size = (args.image_size[0], args.image_size[1])
    copy_images = bool(args.copy_images)

    if not input_root.exists() or not input_root.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]

    total_files = 0
    converted_files = 0
    skipped_files = 0
    copied_images = 0

    for txt_path in sorted(input_root.rglob("*.txt")):
        total_files += 1
        rel_path = txt_path.relative_to(input_root)
        out_txt_path = output_root / rel_path
        out_txt_path = out_txt_path.with_suffix(".txt")
        out_txt_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            converted = convert_seg_file_to_yolo(str(txt_path), str(out_txt_path), image_size=image_size)
            if converted > 0:
                converted_files += 1
            else:
                logging.warning(f"No valid lines converted for {txt_path}")
                skipped_files += 1
        except Exception as exc:
            logging.error(f"Failed to convert {txt_path}: {exc}")
            skipped_files += 1
            continue

        if copy_images:
            src_image = _find_corresponding_image(txt_path, image_extensions)
            if src_image:
                dest_image = (output_root / rel_path.parent / (txt_path.stem + src_image.suffix)).resolve()
                dest_image.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_image, dest_image)
                copied_images += 1

    logging.info("Dataset preparation completed.")
    logging.info(f"Total .txt files found: {total_files}")
    logging.info(f"Files converted: {converted_files}")
    logging.info(f"Files skipped/with errors: {skipped_files}")
    if copy_images:
        logging.info(f"Images copied: {copied_images}")


if __name__ == "__main__":
    main()
