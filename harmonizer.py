import os
import shutil
import hashlib
from sklearn.model_selection import train_test_split

def merge_yolov8_datasets(
    source_dirs: list,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    img_extensions: tuple = ('.jpg', '.png', '.jpeg'),
    seed: int = 42
):
    """
    Fusionne plusieurs datasets YOLOv8 en un seul dataset organisé
    
    Args:
        source_dirs: Liste des chemins des dossiers sources
        output_dir: Dossier de sortie final
        train_ratio: Proportion pour l'entraînement
        val_ratio: Proportion pour validation
        img_extensions: Extensions d'images reconnues
        seed: Seed aléatoire pour reproductibilité
    """
    # Vérification des ratios
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 0.001, "Les ratios doivent sommer à 1"
    
    # Création de la structure YOLOv8
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

    # Collecte de tous les échantillons valides
    all_samples = []
    duplicate_count = 0

    for source_dir in source_dirs:
        images_dir = os.path.join(source_dir, 'images')
        labels_dir = os.path.join(source_dir, 'labels')
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"⚠️ Structure invalide dans {source_dir} - ignoré")
            continue

        for img_file in os.listdir(images_dir):
            if not img_file.lower().endswith(img_extensions):
                continue

            base_name = os.path.splitext(img_file)[0]
            label_file = f"{base_name}.txt"
            img_path = os.path.join(images_dir, img_file)
            label_path = os.path.join(labels_dir, label_file)

            if not os.path.exists(label_path):
                print(f"⚠️ Label manquant pour {img_path} - ignoré")
                continue

            # Vérification des doublons par hash
            with open(img_path, 'rb') as f:
                img_hash = hashlib.md5(f.read()).hexdigest()
            with open(label_path, 'rb') as f:
                label_hash = hashlib.md5(f.read()).hexdigest()

            all_samples.append({
                'img_path': img_path,
                'label_path': label_path,
                'img_hash': img_hash,
                'label_hash': label_hash,
                'base_name': base_name
            })

    # Détection et suppression des doublons
    unique_samples = {}
    for sample in all_samples:
        key = (sample['img_hash'], sample['label_hash'])
        if key not in unique_samples:
            unique_samples[key] = sample
        else:
            duplicate_count += 1
            print(f"⚠️ Doublon détecté: {sample['img_path']}")

    print(f"🔍 {len(all_samples)} échantillons trouvés, {duplicate_count} doublons supprimés")

    # Conversion en liste et séparation
    unique_samples = list(unique_samples.values())
    train_val, test = train_test_split(unique_samples, test_size=test_ratio, random_state=seed)
    train, val = train_test_split(train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=seed)

    # Copie des fichiers avec nouveaux noms uniques
    def copy_samples(samples, split):
        for i, sample in enumerate(samples):
            new_base = f"{split}_{i:05d}"
            
            # Copie image
            img_ext = os.path.splitext(sample['img_path'])[1]
            new_img = f"{new_base}{img_ext}"
            shutil.copy2(
                sample['img_path'],
                os.path.join(output_dir, split, 'images', new_img)
            )
            
            # Copie label
            shutil.copy2(
                sample['label_path'],
                os.path.join(output_dir, split, 'labels', f"{new_base}.txt")
            )

    copy_samples(train, 'train')
    copy_samples(val, 'val')
    copy_samples(test, 'test')

    # Création du fichier data.yaml automatique
    classes = set()
    for split in splits:
        labels_dir = os.path.join(output_dir, split, 'labels')
        for label_file in os.listdir(labels_dir):
            with open(os.path.join(labels_dir, label_file), 'r') as f:
                for line in f:
                    class_id = line.split()[0]
                    classes.add(class_id)

    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(f"train: {os.path.join(output_dir, 'train', 'images')}\n")
        f.write(f"val: {os.path.join(output_dir, 'val', 'images')}\n")
        f.write(f"test: {os.path.join(output_dir, 'test', 'images')}\n\n")
        f.write(f"nc: {len(classes)}\n")
        f.write(f"names: {list(classes)}\n")

    print(f"✅ Fusion réussie! Dataset final dans {output_dir}")
    print(f"📊 Répartition: {len(train)} train, {len(val)} val, {len(test)} test")
    print(f"🏷️ Classes détectées: {len(classes)}")

