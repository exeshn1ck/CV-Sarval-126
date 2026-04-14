import os
import cv2
import albumentations as A

DATASET_DIR = r"D:\YandexDisk\MAI\2nd term\CV\DZ4\dataset"
TRAIN_IMAGES_DIR = os.path.join(DATASET_DIR, "images", "train")
TRAIN_LABELS_DIR = os.path.join(DATASET_DIR, "labels", "train")

AUGS_PER_IMAGE = 3  # сколько новых вариантов делать из одного кадра

transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.7
        ),
        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.10,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.7
        ),
        A.GaussNoise(p=0.2),
        A.Blur(blur_limit=3, p=0.2),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3
    )
)


def read_yolo_labels(label_path: str):
    bboxes = []
    class_labels = []

    if not os.path.exists(label_path):
        return bboxes, class_labels

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls_id = int(float(parts[0]))
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        bboxes.append([x_center, y_center, width, height])
        class_labels.append(cls_id)

    return bboxes, class_labels


def write_yolo_labels(label_path: str, bboxes, class_labels):
    with open(label_path, "w", encoding="utf-8") as f:
        for bbox, cls_id in zip(bboxes, class_labels):
            x_center, y_center, width, height = bbox
            f.write(
                f"{cls_id} "
                f"{x_center:.6f} {y_center:.6f} "
                f"{width:.6f} {height:.6f}\n"
            )


def main():
    image_files = [
        f for f in os.listdir(TRAIN_IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    image_files.sort()

    created = 0

    for image_name in image_files:
        image_path = os.path.join(TRAIN_IMAGES_DIR, image_name)
        base_name = os.path.splitext(image_name)[0]
        label_path = os.path.join(TRAIN_LABELS_DIR, base_name + ".txt")

        image = cv2.imread(image_path)
        if image is None:
            print(f"Не удалось прочитать: {image_path}")
            continue

        bboxes, class_labels = read_yolo_labels(label_path)

        # если на картинке нет объекта, можно либо пропустить, либо аугментировать как фон
        for aug_idx in range(AUGS_PER_IMAGE):
            try:
                transformed = transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
            except Exception as e:
                print(f"Ошибка при аугментации {image_name}: {e}")
                continue

            aug_image = transformed["image"]
            aug_bboxes = transformed["bboxes"]
            aug_labels = transformed["class_labels"]

            new_image_name = f"{base_name}_aug_{aug_idx}.jpg"
            new_label_name = f"{base_name}_aug_{aug_idx}.txt"

            new_image_path = os.path.join(TRAIN_IMAGES_DIR, new_image_name)
            new_label_path = os.path.join(TRAIN_LABELS_DIR, new_label_name)

            cv2.imwrite(new_image_path, aug_image)
            write_yolo_labels(new_label_path, aug_bboxes, aug_labels)

            created += 1

    print(f"Готово. Создано новых изображений: {created}")


if __name__ == "__main__":
    main()