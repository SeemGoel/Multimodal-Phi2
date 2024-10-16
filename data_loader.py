import torch
from torch.utils.data import IterableDataset
import json
import numpy as np
from transformers import AutoTokenizer
from collections import Counter
from tqdm import tqdm


class MultimodalDataset(IterableDataset):
    def __init__(self, json_file, embedding_file, tokenizer_name, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, pad_token="[PAD]"
        )
        self.max_length = max_length
        self.json_file = json_file
        self.embedding_file = embedding_file

        # Load CLIP embeddings
        embeddings_data = np.load(embedding_file)
        print(f"Keys in the embedding file: {embeddings_data.files}")

        # Try to load embeddings with different possible key names
        embedding_key = next(
            (key for key in embeddings_data.files if "embedding" in key.lower()), None
        )
        if embedding_key:
            self.image_embeddings = embeddings_data[embedding_key]
        else:
            raise KeyError("No embedding key found in the NPZ file")

        # Try to load ids with different possible key names
        id_key = next(
            (key for key in embeddings_data.files if "id" in key.lower()), None
        )
        if id_key:
            self.image_ids = embeddings_data[id_key]
        else:
            raise KeyError("No id key found in the NPZ file")

        print(f"Loaded embeddings shape: {self.image_embeddings.shape}")
        print(f"Loaded ids shape: {self.image_ids.shape}")

        # Create a mapping from image_id to embedding index
        self.image_id_to_index = {id: idx for idx, id in enumerate(self.image_ids)}

        self.missing_embeddings = Counter()

        # Load the JSON data to get the total number of items
        with open(self.json_file, 'r') as f:
            self.data = json.load(f)
        
        # Calculate the total number of conversation pairs
        self.total_items = sum(len(item['conversations']) // 2 for item in self.data)
        
        print(f"Total number of conversation pairs: {self.total_items}")

    def __iter__(self):
        with open(self.json_file, "r") as f:
            data = json.load(f)
            for item in data:
                try:
                    image_id = (
                        item["image"].replace(".jpg", "").replace("COCO_train2014_", "")
                    )
                    for i in range(0, len(item["conversations"]), 2):
                        human_conv = item["conversations"][i]
                        gpt_conv = (
                            item["conversations"][i + 1]
                            if i + 1 < len(item["conversations"])
                            else None
                        )

                        if (
                            human_conv["from"] == "human"
                            and gpt_conv
                            and gpt_conv["from"] == "gpt"
                        ):
                            question = (
                                human_conv["value"].replace("<image>\n", "").strip()
                            )
                            answer = gpt_conv["value"]

                            # Tokenize input
                            input_text = f"Human: {question}\nAssistant: {answer}"
                            tokenized = self.tokenizer(
                                input_text,
                                truncation=True,
                                max_length=self.max_length,
                                padding="max_length",
                            )

                            # Get image embedding
                            image_embedding = self.get_image_embedding(image_id)

                            yield {
                                "input_ids": torch.tensor(tokenized["input_ids"]),
                                "attention_mask": torch.tensor(
                                    tokenized["attention_mask"]
                                ),
                                "labels": torch.tensor(tokenized["input_ids"]),
                                "image_embeddings": torch.tensor(
                                    image_embedding
                                ).float(),
                            }

                except Exception as e:
                    print(f"Error processing item: {str(e)}")
                    print(f"Problematic item: {item}")

    def get_image_embedding(self, image_id):
        if image_id in self.image_id_to_index:
            return self.image_embeddings[self.image_id_to_index[image_id]]
        else:
            self.missing_embeddings[image_id] += 1
            return np.zeros(self.image_embeddings.shape[1])

    def print_missing_embeddings_summary(self):
        print("\nMissing Embeddings Summary:")
        print(f"Total unique missing embeddings: {len(self.missing_embeddings)}")
        print(
            f"Total missing embedding occurrences: {sum(self.missing_embeddings.values())}"
        )
        print("\nTop 10 missing embeddings:")
        for image_id, count in self.missing_embeddings.most_common(10):
            print(f"Image ID: {image_id}, Count: {count}")

    def save_to_json(self, output_file, limit=None):
        print(f"Saving dataset to JSON: {output_file}")
        dataset = []
        for item in tqdm(self.data[:limit] if limit else self.data, desc="Processing items"):
            try:
                image_id = item['image'].replace('.jpg', '').replace('COCO_train2014_', '')
                image_embedding = self.get_image_embedding(image_id).tolist()
                
                for i in range(0, len(item['conversations']), 2):
                    human_conv = item['conversations'][i]
                    gpt_conv = item['conversations'][i+1] if i+1 < len(item['conversations']) else None

                    if human_conv['from'] == 'human' and gpt_conv and gpt_conv['from'] == 'gpt':
                        question = human_conv['value'].replace("<image>\n", "").strip()
                        answer = gpt_conv['value']

                        dataset_item = {
                            "image_id": image_id,
                            "question": question,
                            "answer": answer,
                            "image_embedding": image_embedding
                        }
                        dataset.append(dataset_item)
            
            except Exception as e:
                print(f"Error processing item: {str(e)}")
                print(f"Problematic item: {item}")

        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Dataset saved to {output_file}")
        print(f"Total items saved: {len(dataset)}")


import torch
from torch.utils.data import DataLoader

# Main execution
if __name__ == "__main__":
    json_file = '/content/llava_instruct_150k.json'
    tokenizer_name = 'microsoft/phi-2'
    embedding_file = '/content/drive/MyDrive/llm/clip_embeddings_instruct150k_15Oct_2024.npz'

    print(f"Initializing dataset with:")
    print(f"JSON file: {json_file}")
    print(f"Embedding file: {embedding_file}")
    print(f"Tokenizer: {tokenizer_name}")

    dataset = MultimodalDataset(json_file, embedding_file, tokenizer_name)

    # Create a DataLoader
    batch_size = 32
    print(f"Creating DataLoader with batch size: {batch_size}")
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Calculate the number of batches
    num_batches = (dataset.total_items + batch_size - 1) // batch_size
    print(f"Total number of batches: {num_batches}")

    # Iterate through the data
    print("Starting data iteration...")
    try:
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches", total=num_batches)):
            # Unpack the batch
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            image_embeddings = batch["image_embeddings"]

            # Here you would typically process your batch, e.g., pass it through your model
            # For demonstration, we'll just print the shapes of each tensor
            print(f"\nBatch {batch_idx + 1}")
            print(f"Input IDs shape: {input_ids.shape}")
            print(f"Attention mask shape: {attention_mask.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Image embeddings shape: {image_embeddings.shape}")

            # Break after first batch for demonstration purposes
            # Remove this line to process all batches
            if batch_idx == 0:
                break

        print("Data iteration complete!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    # Print missing embeddings summary
    print("\nPrinting missing embeddings summary:")
    dataset.print_missing_embeddings_summary()

    # Save dataset to JSON
    json_output_file = '/content/drive/MyDrive/processed_dataset.json'
    print("\nSaving dataset to JSON...")
    dataset.save_to_json(json_output_file, limit=1000)  # Set limit=None to process all items
