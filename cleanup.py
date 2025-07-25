import os
import re
import logging

# --- Configuration ---
LOG_FILE = "data_creation_final.log"
ANNOT_DIR = "raw_dataset/labeled_annotations"
IMAGE_DIR = "images"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cleanup.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_corrupted_basenames(log_file_path):
    """Parses the log file to extract the basenames of corrupted files."""
    if not os.path.exists(log_file_path):
        logger.error(f"Log file not found at: {log_file_path}")
        return set()

    # Regex to find lines with "SKIPPING" or "FAILED" and extract the basename
    # It looks for the word SKIPPING or FAILED, then captures the word-like sequence after it.
    pattern = re.compile(r"(?:SKIPPING|FAILED for)\s+([\w\d\.-]+):")
    
    corrupted_files = set()
    with open(log_file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                # The basename is the first captured group, without the .json extension
                basename = os.path.splitext(match.group(1))[0]
                corrupted_files.add(basename)
                
    return corrupted_files

def main():
    logger.info("--- Starting Corrupted Data Cleanup Script ---")

    corrupted_basenames = get_corrupted_basenames(LOG_FILE)

    if not corrupted_basenames:
        logger.info("No corrupted files found in the log. Nothing to delete.")
        return

    logger.info(f"Found {len(corrupted_basenames)} unique corrupted files to delete.")

    deleted_count = 0
    for basename in corrupted_basenames:
        # --- Delete JSON file ---
        json_file = os.path.join(ANNOT_DIR, f"{basename}.json")
        if os.path.exists(json_file):
            try:
                os.remove(json_file)
                logger.info(f"Deleted annotation: {json_file}")
                deleted_count += 1
            except OSError as e:
                logger.error(f"Error deleting {json_file}: {e}")

        # --- Delete corresponding image file (check for both .jpg and .png) ---
        jpg_file = os.path.join(IMAGE_DIR, f"{basename}.jpg")
        png_file = os.path.join(IMAGE_DIR, f"{basename}.png")

        if os.path.exists(jpg_file):
            try:
                os.remove(jpg_file)
                logger.info(f"Deleted image: {jpg_file}")
            except OSError as e:
                logger.error(f"Error deleting {jpg_file}: {e}")
        elif os.path.exists(png_file):
            try:
                os.remove(png_file)
                logger.info(f"Deleted image: {png_file}")
            except OSError as e:
                logger.error(f"Error deleting {png_file}: {e}")
        else:
            logger.warning(f"No corresponding image found for {basename}.json")

    logger.info(f"--- Cleanup complete. Deleted {deleted_count} annotation files (and their images). ---")
    logger.info("You can now safely re-run the 'create_dataset.py' script.")

if __name__ == "__main__":
    main()
