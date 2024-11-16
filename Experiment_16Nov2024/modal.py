import os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, WhisperProcessor, WhisperForConditionalGeneration, PreTrainedModel,BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import json
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import gc


# Define multimodal projector class
class ProjectionBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.pre_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Sequential(nn.Linear(input_dim, output_dim), nn.GELU(), nn.Linear(output_dim, output_dim))

    def forward(self, x):
        return self.proj(self.pre_norm(x))

class MultimodalProjector(nn.Module):
    def __init__(self, image_input_dim, audio_input_dim, output_dim):
        super().__init__()
        self.image_proj = ProjectionBlock(image_input_dim, output_dim)
        self.audio_proj = ProjectionBlock(audio_input_dim, output_dim)

    def forward(self, image_embedding=None, audio_embedding=None):
        projected_image = self.image_proj(image_embedding) if image_embedding is not None else None
        projected_audio = self.audio_proj(audio_embedding) if audio_embedding is not None else None
        return projected_image, projected_audio



class Phi3WithProjector(PreTrainedModel):
    def __init__(self, config, phi3_model, projector):
        super().__init__(config)
        self.phi3_model = phi3_model
        self.projector = projector
        self.supports_gradient_checkpointing = True 


    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, image_embeddings=None, audio_embeddings=None, labels=None, **kwargs):
        # Use get_input_embeddings() to retrieve the embeddings layer
        if inputs_embeds is None:
            inputs_embeds = self.phi3_model.get_input_embeddings()(input_ids)

        # Project both image and audio embeddings to the appropriate dimension
        projected_image, projected_audio = self.projector(image_embeddings, audio_embeddings)

        # Concatenate the embeddings
        embeddings_to_concat = [inputs_embeds]
        if projected_image is not None:
            embeddings_to_concat.append(projected_image.unsqueeze(1))
        if projected_audio is not None:
            embeddings_to_concat.append(projected_audio.unsqueeze(1))

        combined_embeddings = torch.cat(embeddings_to_concat, dim=1)

        # Modify how the attention mask is extended
        extended_attention_mask = attention_mask.clone()  # Start with a copy

        # Extend for image and audio, if present
        if projected_image is not None:
            extended_attention_mask = torch.cat([extended_attention_mask, torch.ones_like(extended_attention_mask[:, :1])], dim=1)
        if projected_audio is not None:
            extended_attention_mask = torch.cat([extended_attention_mask, torch.ones_like(extended_attention_mask[:, :1])], dim=1)

        # Adjust labels to match the extended input sequence length
        if labels is not None:
            # Pad labels with -100 to ignore the added tokens in the loss calculation
            num_added_tokens = sum(1 for emb in [projected_image, projected_audio] if emb is not None)
            labels = torch.cat([labels, torch.full((labels.shape[0], num_added_tokens), -100, dtype=labels.dtype, device=labels.device)], dim=1)

        return self.phi3_model(
            inputs_embeds=combined_embeddings,
            attention_mask=extended_attention_mask,
            labels=labels,
            **kwargs
        )

    def get_input_embeddings(self):
        """Returns the model's input embeddings."""
        return self.phi3_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Sets the model's input embeddings."""
        self.phi3_model.set_input_embeddings(value)
    
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (nn.Linear,)):
            module.gradient_checkpointing = value

    def enable_gradient_checkpointing(self):
      """Enable gradient checkpointing for the model."""
      if hasattr(self.base_model, 'gradient_checkpointing_enable'):
          self.base_model.gradient_checkpointing_enable()
      elif hasattr(self.base_model, 'gradient_checkpointing'):
          self.base_model.gradient_checkpointing = True
