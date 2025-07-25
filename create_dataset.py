import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import torch
from datasets import (ClassLabel, Dataset, DatasetDict, Features,
                      Image as HfImage, Sequence, Value)
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import AutoProcessor

# --- Configuration ---
RAW_DATA_DIR = './raw_dataset' 
IMAGE_DIR = os.path.join(RAW_DATA_DIR, 'images')
ANNOT_DIR = os.path.join(RAW_DATA_DIR, 'labeled_annotations')

# This is where the final, processed dataset will be saved.
OUTPUT_DIR = "./processed_doclaynet"
LABELS_FILE_PATH = os.path.join(OUTPUT_DIR, "labels.json")

# --- Logging Setup ---
# This will create a log file to tell you which files were skipped and why.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_creation_final.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def validate_and_process_file(file_info):
    """
    Definitively validates a single data file, including checking
    bounding boxes against actual image dimensions.
    """
    idx, basename = file_info
    json_path = os.path.join(ANNOT_DIR, f"{basename}.json")

    try:
        # --- Find and Validate Image ---
        png_path = os.path.join(IMAGE_DIR, f"{basename}.png")
        jpg_path = os.path.join(IMAGE_DIR, f"{basename}.jpg")
        
        image_path = None
        if os.path.exists(png_path):
            image_path = png_path
        elif os.path.exists(jpg_path):
            image_path = jpg_path
        else:
            return None # Should not happen due to initial filtering

        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except UnidentifiedImageError:
            logger.warning(f"SKIPPING {basename}: Corrupted or unidentified image file.")
            return None

        # --- Validate JSON and Bounding Boxes ---
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        form_data = data.get('form', [])
        if not form_data:
            logger.warning(f"SKIPPING {basename}: 'form' data is empty or missing.")
            return None

        words, boxes, labels = [], [], []
        for i, item in enumerate(form_data):
            box = item.get('box')
            if not (isinstance(box, list) and len(box) == 4):
                logger.warning(f"SKIPPING {basename}: Invalid box format at item {i}: {box}")
                return None
            
            x0, y0, w, h = box
            x1, y1 = x0 + w, y0 + h

            # --- THE CRUCIAL VALIDATION STEP ---
            # Check if the bounding box is within the image dimensions.
            # Using a small tolerance for floating point issues.
            if not (0 <= x0 and 0 <= y0 and x1 <= img_width + 0.01 and y1 <= img_height + 0.01):
                logger.error(
                    f"SKIPPING {basename}: Bounding box [{x0}, {y0}, {x1}, {y1}] "
                    f"is outside image dimensions [{img_width}, {img_height}]."
                )
                return None

            words.append(item.get('text', ''))
            boxes.append([int(x0), int(y0), int(x1), int(y1)])
            labels.append(item.get('level'))

        return {'id': str(idx), 'tokens': words, 'bboxes': boxes, 'ner_tags_str': labels, 'image_path': image_path, 'original_basename': basename}

    except Exception as e:
        logger.error(f"SKIPPING {basename} due to critical error: {e}", exc_info=True)
        return None

def sanity_check_processed_dataset(dataset, label_list):
    """
    Iterates through the processed dataset to ensure all values are valid
    before saving to disk. This is the final check to prevent CUDA errors.
    """
    logger.info("Running final sanity check on processed data...")
    num_labels = len(label_list)
    failed_samples = []

    for split in dataset.keys():
        for i in tqdm(range(len(dataset[split])), desc=f"Checking '{split}' split"):
            sample = dataset[split][i]
            basename = sample['original_basename']
            
            # Check bounding boxes
            bboxes = torch.tensor(sample['bbox'])
            if torch.any(bboxes < 0) or torch.any(bboxes > 1000):
                logger.error(f"Validation FAILED for {basename}: Bbox out of [0, 1000] range.")
                failed_samples.append(basename)
                continue

            # Check labels
            labels = torch.tensor(sample['labels'])
            if torch.any((labels < -100) | (labels >= num_labels)):
                logger.error(f"Validation FAILED for {basename}: Label out of [0, {num_labels-1}] range.")
                failed_samples.append(basename)
                continue
    
    if failed_samples:
        logger.error("Sanity check failed. The following files contain corrupted data after processing:")
        for fname in set(failed_samples):
            logger.error(f" - {fname}.json")
        return False
    
    logger.info("✅ Sanity check passed. All processed data is valid.")
    return True


def main():
    logger.info("🚀 Starting DEFINITIVE dataset creation pipeline...")
    
    image_basenames = {os.path.splitext(f)[0] for f in os.listdir(IMAGE_DIR)}
    json_basenames = {os.path.splitext(f)[0] for f in os.listdir(ANNOT_DIR)}
    matching_basenames = sorted(list(image_basenames.intersection(json_basenames)))
    
    logger.info(f"Found {len(matching_basenames)} matching image-annotation pairs.")
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(
            executor.map(validate_and_process_file, enumerate(matching_basenames)), 
            total=len(matching_basenames), 
            desc="Validating raw files against image dimensions"
        ))
    
    data_list = [d for d in results if d is not None]
    logger.info(f"Validation complete. Found {len(data_list)} fully valid files.")
    
    if not data_list:
        raise ValueError("No valid data could be loaded. Check data_creation_final.log for details.")

    all_labels = sorted(list(set(label for sample in data_list for label in sample['ner_tags_str'])))
    label2id = {label: i for i, label in enumerate(all_labels)}
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(LABELS_FILE_PATH, 'w') as f:
        json.dump(all_labels, f, indent=2)
    logger.info(f"✅ Label list with {len(all_labels)} labels saved to {LABELS_FILE_PATH}")

    for sample in data_list:
        sample['ner_tags'] = [label2id[l] for l in sample['ner_tags_str']]
        sample['image'] = sample.pop('image_path')
        del sample['ner_tags_str']

    features = Features({
        'id': Value('string'),
        'tokens': Sequence(Value('string')),
        'bboxes': Sequence(Sequence(Value('int64'))),
        'ner_tags': Sequence(ClassLabel(names=all_labels)),
        'image': HfImage(decode=True),
        'original_basename': Value('string'),
    })

    dataset = Dataset.from_list(data_list, features=features)
    dataset = dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)

    logger.info("Initializing LayoutLMv3Processor and applying it to the dataset...")
    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

    def prepare_examples(examples):
        encoding = processor(
            examples["image"],
            examples["tokens"],
            boxes=examples["bboxes"],
            word_labels=examples["ner_tags"],
            truncation=True,
            padding="max_length"
        )
        # Keep original_basename for debugging
        encoding['original_basename'] = examples['original_basename']
        return encoding

    num_procs = max(1, os.cpu_count() - 2)
    logger.info(f"Using {num_procs} cores for processing...")

    processed_dataset = dataset.map(
        prepare_examples,
        batched=True,
        num_proc=num_procs,
        remove_columns=dataset["train"].column_names,
        desc="Running LayoutLMv3 processor"
    )
    
    # Run the final check before saving
    if not sanity_check_processed_dataset(processed_dataset, all_labels):
        logger.error("Halting due to validation errors. Please check the log and fix the corrupted files.")
        return

    # Remove the temporary column before saving
    processed_dataset = processed_dataset.remove_columns(['original_basename'])
    
    logger.info(f"Saving final processed dataset to {OUTPUT_DIR}...")
    processed_dataset.save_to_disk(OUTPUT_DIR)
    
    logger.info("="*50)
    logger.info(f"✅✅✅ All done! Your training-ready dataset is saved in '{OUTPUT_DIR}'.")
    logger.info("="*50)

if __name__ == "__main__":
    main()
