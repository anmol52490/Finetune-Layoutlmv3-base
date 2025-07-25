import os
import json
from concurrent.futures import ThreadPoolExecutor
from datasets import Dataset, Features, Sequence, ClassLabel, Value, DatasetDict
from datasets import Image as HfImage
import logging

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- IMPORTANT: Set your local paths here ---
# This should be the path to your folder with 'images' and 'labeled_annotations'
base_dir = './raw_dataset' # e.g., 'C:/Users/admin/Desktop/DocLayNet_raw'
image_dir = os.path.join(base_dir, 'images')
annot_dir = os.path.join(base_dir, 'labeled_annotations')

# --- Path for final output ---
output_dir = "./processed_doclaynet"
labels_file_path = os.path.join(output_dir, "labels.json")

def process_file(idx_filename):
    # (This function is the same as your original one)
    idx, filename = idx_filename
    try:
        json_path = os.path.join(annot_dir, filename)
        image_path = os.path.join(image_dir, filename.replace('.json', '.png'))
        if not os.path.exists(image_path):
            image_path = os.path.join(image_dir, filename.replace('.json', '.jpg'))
            if not os.path.exists(image_path): return None
        with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
        words, boxes, labels = [], [], []
        for item in data['form']:
            words.append(item['text'])
            x0, y0, w, h = item['box']
            boxes.append([x0, y0, x0 + w, y0 + h])
            labels.append(item['level'])
        return {'id': str(idx), 'tokens': words, 'bboxes': boxes, 'ner_tags_str': labels, 'image': image_path}
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        return None

def main():
    logger.info("Starting dataset creation process...")
    json_files = [f for f in os.listdir(annot_dir) if f.endswith('.json')]
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        data_list = list(executor.map(process_file, enumerate(json_files)))
    data_list = [d for d in data_list if d is not None]

    # --- Create and SAVE the label list ---
    all_labels = set()
    for sample in data_list:
        all_labels.update(sample['ner_tags_str'])
    label_list = sorted(list(all_labels))
    
    os.makedirs(output_dir, exist_ok=True)
    with open(labels_file_path, 'w') as f:
        json.dump(label_list, f)
    logger.info(f"✅ Label list saved to {labels_file_path}")

    # --- Create the initial dataset ---
    label2id = {label: i for i, label in enumerate(label_list)}
    for sample in data_list:
        sample['ner_tags'] = [label2id[l] for l in sample['ner_tags_str']]
        del sample['ner_tags_str']

    features = Features({
        'id': Value('string'),
        'tokens': Sequence(Value('string')),
        'bboxes': Sequence(Sequence(Value('int64'))),
        'ner_tags': Sequence(ClassLabel(names=label_list)),
        'image': HfImage(decode=True)
    })
    dataset = Dataset.from_list(data_list, features=features)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # --- Process and save the final dataset ---
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

    def prepare_examples(examples):
        images = examples["image"]
        words = examples["tokens"]
        boxes = examples["bboxes"]
        word_labels = examples["ner_tags"]
        encoding = processor(images, words, boxes=boxes, word_labels=word_labels, truncation=True, padding="max_length")
        return encoding

    logger.info("Applying processor to the dataset...")
    processed_dataset = dataset.map(
        prepare_examples,
        batched=True,
        num_proc=os.cpu_count(), # Use multiple cores for processing
        remove_columns=dataset["train"].column_names
    )
    
    logger.info(f"Saving processed dataset to {output_dir}...")
    processed_dataset.save_to_disk(output_dir)
    logger.info("✅ Dataset creation and processing complete!")

if __name__ == "__main__":
    main()