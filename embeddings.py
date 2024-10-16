import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import json
import os
import numpy as np
from tqdm import tqdm
import multiprocessing
from functools import partial

class OptimizedCLIPEmbeddingGenerator:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate_embeddings(self, texts, images):
        try:
            inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                text_embeddings = outputs.text_embeds.cpu().numpy()
                image_embeddings = outputs.image_embeds.cpu().numpy()

            return text_embeddings, image_embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return None, None

def load_image(path):
    try:
        return Image.open(path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

def load_images(image_paths):
    with multiprocessing.Pool() as pool:
        return pool.map(load_image, image_paths)

def process_llava_instruct_dataset(llava_json_path, image_folder_path, output_file, batch_size=32, checkpoint_interval=1000):
    # Load the LLaVA dataset
    try:
        with open(llava_json_path, "r") as file:
            llava_data = json.load(file)
    except Exception as e:
        print(f"Error loading dataset from {llava_json_path}: {e}")
        return

    clip_generator = OptimizedCLIPEmbeddingGenerator()
    
    # Initialize lists to store processed data
    ids = []
    image_filenames = []
    image_embeddings = []

    # Load processed data checkpoint if it exists
    start_index = 0
    if os.path.exists(f"{output_file}_checkpoint.npz"):
        try:
            checkpoint = np.load(f"{output_file}_checkpoint.npz")
            ids = list(checkpoint['ids'])
            image_filenames = list(checkpoint['image_filenames'])
            image_embeddings = list(checkpoint['image_embeddings'])
            start_index = len(ids)
            print(f"Resuming from checkpoint at index {start_index}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    for i in tqdm(range(start_index, len(llava_data), batch_size), desc="Processing dataset"):
        batch_entries = llava_data[i:i+batch_size]

        batch_image_paths = []
        batch_ids = []

        for entry in batch_entries:
            image_filename = entry['image']
            image_path = os.path.join(image_folder_path, image_filename)
            batch_image_paths.append(image_path)
            batch_ids.append(image_filename.replace('COCO_train2014_', ''))

        batch_images = load_images(batch_image_paths)

        # Generate embeddings for valid images
        valid_indices = [index for index, img in enumerate(batch_images) if img is not None]
        valid_images = [batch_images[idx] for idx in valid_indices]
        valid_ids = [batch_ids[idx] for idx in valid_indices]

        if valid_images:
            _, batch_image_embeddings = clip_generator.generate_embeddings([""] * len(valid_images), valid_images)

            if batch_image_embeddings is not None:
                ids.extend(valid_ids)
                image_filenames.extend([os.path.basename(batch_image_paths[idx]) for idx in valid_indices])
                image_embeddings.extend(batch_image_embeddings)

        # Save checkpoint every checkpoint_interval steps
        if (i + len(batch_entries)) % checkpoint_interval == 0:
            try:
                np.savez(f"{output_file}_checkpoint.npz",
                         ids=np.array(ids),
                         image_filenames=np.array(image_filenames),
                         image_embeddings=np.array(image_embeddings))
                print(f"Checkpoint saved at {i + len(batch_entries)} entries.")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")

    # Final save after all processing
    try:
        np.savez(output_file,
                 ids=np.array(ids),
                 image_filenames=np.array(image_filenames),
                 image_embeddings=np.array(image_embeddings))
        print(f"Processed data saved to {output_file}")
    except Exception as e:
        print(f"Error saving processed data to {output_file}: {e}")

# Utility function to load processed data for training
def load_processed_data(file_path):
    try:
        data = np.load(file_path)
        return {
            'ids': data['ids'],
            'image_filenames': data['image_filenames'],
            'image_embeddings': data['image_embeddings']
        }
    except Exception as e:
        print(f"Error loading processed data from {file_path}: {e}")
        return None

# Main execution
if __name__ == "__main__":
    print("Processing LLaVA Instruct dataset...")
    process_llava_instruct_dataset('llava_instruct_150k.json', '/content/train2014', '/content/drive/MyDrive/processed_llava_instruct_data.npz')
