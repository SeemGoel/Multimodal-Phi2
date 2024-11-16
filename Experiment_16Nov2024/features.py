# -*- coding: utf-8 -*-
"""prepare_dataset_tokenise.py - Optimized for Multimodal Fine-tuning"""

import os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, WhisperProcessor, WhisperForConditionalGeneration, PreTrainedModel,BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import json
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import gc
from transformers import EarlyStoppingCallback
from torch.utils.checkpoint import checkpoint_sequential

# Initialize Whisper components for audio transcription
whisper_model_name = "openai/whisper-small"
whisper_processor = WhisperProcessor.from_pretrained(whisper_model_name)
whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)

# Load embeddings with error handling
def load_embeddings(file_path):
    try:
        data = np.load(file_path)
        if 'image_ids' in data and 'embeddings' in data:
            return {'ids': data['image_ids'], 'embeddings': data['embeddings']}
        else:
            raise ValueError(f"Unexpected structure in {file_path}.")
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None

# Process audio files
def transcribe_speech(audiopath):
    try:
        speech, rate = librosa.load(audiopath, sr=16000)
        audio_input = whisper_processor(speech, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            generated_ids = whisper_model.generate(audio_input["input_features"])
        return whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None



@dataclass
class MultimodalDataCollator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {"input_ids": self.tokenizer.pad({"input_ids": [f["input_ids"] for f in features]}, padding=True, return_tensors="pt")["input_ids"]}
        batch["attention_mask"] = torch.ones_like(batch["input_ids"])
        batch["labels"] = batch["input_ids"].clone()
        if "image_embeddings" in features[0]:
            batch["image_embeddings"] = torch.stack([f["image_embeddings"] for f in features])
        if "audio_embeddings" in features[0]:
            batch["audio_embeddings"] = torch.stack([f["audio_embeddings"] for f in features])
        return batch

# Dataset preparation with better error handling and modularization
def prepare_dataset(image_embeddings_path, dataset_path, cache_dir=None):
    image_embeddings = load_embeddings(image_embeddings_path)
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    processed_data = [{"conversation": item["conversations"], "image_embedding": image_embeddings['embeddings'][np.where(image_embeddings['ids'] == item['image'])[0][0]] if image_embeddings and "image" in item else None, "audio_path": item.get("audio")} for item in data]
    dataset = Dataset.from_dict({"conversation": [item["conversation"] for item in processed_data], "image_embedding": [item.get("image_embedding") for item in processed_data], "audio_path": [item.get("audio_path") for item in processed_data]})
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # tokenizer.chat_template = """
    # {% for message in messages %}
    # {% if message.role == 'system' %}<|system|>{{message.content}}<|endoftext|>{% elif message.role == 'user' %}<|user|>{{message.content}}<|endoftext|>{% elif message.role == 'assistant' %}<|assistant|>{{message.content}}<|endoftext|>{% endif %}{% endfor %}
    # """
    tokenizer.chat_template = """
    {% for message in messages %}
    {% if message.role == 'system' %}<|system|>{{message.content}}<|endofsystem|>{% elif message.role == 'user' %}<|user|>{{message.content}}<|endoftext|>{% elif message.role == 'assistant' %}<|assistant|>{{message.content}}<|endoftext|>{% endif %}{% endfor %}
    """
    prepared_dataset = dataset.map(lambda examples: prepare_example(examples, tokenizer), batched=True, remove_columns=dataset.column_names, batch_size=1).with_format("torch")
    # dataset_dict = DatasetDict({"train": prepared_dataset.train_test_split(test_size=0.1)["train"], "test": prepared_dataset.train_test_split(test_size=0.1)["test"]})
    dataset_dict = prepared_dataset.train_test_split(test_size=0.2) # Split into train and a combined validation/test set
    dataset_dict["validation"] = dataset_dict["test"].train_test_split(test_size=0.5)["train"] # Split the combined set in half
    dataset_dict["test"] = dataset_dict["test"].train_test_split(test_size=0.5)["test"] # Split the combined set in half

    # Assuming you have your dataset in 'dataset_dict'
    drive_path = "/content/drive/MyDrive/Cap_dataset" # Replace with your desired path in Google Drive
    dataset_dict.save_to_disk(drive_path)


    # if cache_dir:
    #     os.makedirs(cache_dir, exist_ok=True)
    #     dataset_dict.save_to_disk(cache_dir)
    return dataset_dict, tokenizer

# Example preparation for dataset rows
def prepare_example(examples, tokenizer):
    image_embeddings, audio_embeddings, tokenized_inputs = [], [], []
    for idx, conv in enumerate(examples["conversation"]):
        image_embedding = torch.tensor(examples["image_embedding"][idx]) if examples["image_embedding"][idx] is not None else None
        transcription = transcribe_speech(examples["audio_path"][idx]) if "audio_path" in examples and examples["audio_path"][idx] else None
        for i in range(0, len(conv), 2):
            if i + 1 < len(conv):
                human_msg = conv[i]["value"].replace("<image>", "").replace("<audio>", "").strip()
                if transcription:
                    human_msg += f"\nAudio Transcription: {transcription}"
                gpt_msg = conv[i + 1]["value"]
                tokenized_input = tokenizer.apply_chat_template([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": f"{human_msg}"}, {"role": "assistant", "content": gpt_msg}], return_tensors="pt", padding=True)
                tokenized_inputs.append(tokenized_input.squeeze(0))
                if image_embedding is not None:
                    image_embeddings.append(image_embedding)
    max_length = max(input.shape[0] for input in tokenized_inputs)
    padded_inputs = [torch.nn.functional.pad(input, (0, max_length - input.shape[0])) for input in tokenized_inputs]
    result = {"input_ids": torch.stack(padded_inputs), "attention_mask": torch.stack(padded_inputs).ne(tokenizer.pad_token_id).long(), "labels": torch.stack(padded_inputs).clone()}
    if image_embeddings:
        result["image_embeddings"] = torch.stack(image_embeddings)
    if audio_embeddings:
        result["audio_embeddings"] = torch.stack(audio_embeddings)
    return result
