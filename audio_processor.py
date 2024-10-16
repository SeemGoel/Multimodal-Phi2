import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class AudioProcessor:
    def __init__(self, model_name="openai/whisper-base"):
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def transcribe(self, audio_file):
        # Load and preprocess the audio file
        audio_input = self.processor(audio_file, return_tensors="pt").input_features.to(self.device)

        # Generate the transcription
        with torch.no_grad():
            generated_ids = self.model.generate(audio_input)
        
        # Decode the generated ids to text
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription

    def get_audio_embeddings(self, audio_file):
        # Load and preprocess the audio file
        audio_input = self.processor(audio_file, return_tensors="pt").input_features.to(self.device)

        # Get the audio embeddings
        with torch.no_grad():
            outputs = self.model(audio_input, output_hidden_states=True)
        
        # Use the last hidden state as the audio embedding
        audio_embeddings = outputs.last_hidden_state.mean(dim=1)
        return audio_embeddings.cpu().numpy()
