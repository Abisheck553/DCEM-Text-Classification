"""
Downloads images for all 10 classes automatically using Bing Image Search.
Run this once to build your dataset.
"""

import os
from icrawler.builtin import BingImageCrawler

# Search keywords for each class (matches the paper's classes)
CLASS_QUERIES = {
    "yoga":         "yoga exercise pose",
    "cooking":      "cooking food kitchen",
    "craft":        "arts and crafts making",
    "concert":      "music concert live show",
    "teleshopping": "teleshopping tv shopping channel",
    "animal":       "animals wildlife nature",
    "outlet":       "shopping outlet store",
    "technology":   "technology gadgets electronics",
    "elearning":    "elearning online education class",
    "animation":    "cartoon animation movie scene",
}

TRAIN_COUNT = 80   # images per class for training
TEST_COUNT  = 20   # images per class for testing


def download_class(class_name, query, save_dir, count):
    os.makedirs(save_dir, exist_ok=True)
    print(f"  Downloading '{class_name}' → {save_dir}  ({count} images)")
    crawler = BingImageCrawler(
        storage={"root_dir": save_dir},
        feeder_threads=1,
        parser_threads=1,
        downloader_threads=4,
    )
    crawler.crawl(keyword=query, max_num=count, file_idx_offset=0)


def main():
    print("=" * 55)
    print("  DCEM Dataset Downloader")
    print("=" * 55)
    print(f"Classes   : {len(CLASS_QUERIES)}")
    print(f"Train/class: {TRAIN_COUNT} images")
    print(f"Test/class : {TEST_COUNT} images")
    print(f"Total      : ~{len(CLASS_QUERIES) * (TRAIN_COUNT + TEST_COUNT)} images")
    print("=" * 55)

    for class_name, query in CLASS_QUERIES.items():
        print(f"\n[{class_name.upper()}]")
        download_class(
            class_name, query,
            save_dir=f"data/train/{class_name}",
            count=TRAIN_COUNT,
        )
        download_class(
            class_name, query + " hd",
            save_dir=f"data/test/{class_name}",
            count=TEST_COUNT,
        )

    print("\n" + "=" * 55)
    print("  Download complete!")
    print("  Training images : data/train/")
    print("  Testing images  : data/test/")
    print("=" * 55)
    print("\nNext step — train the model:")
    print("  python main.py --mode train --data_dir data/train --backbone vgg16 --num_classes 10 --epochs 30")


if __name__ == "__main__":
    main()
