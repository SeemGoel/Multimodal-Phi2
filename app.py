from flask import Flask, request, jsonify, render_template
import torch
from model import MultimodalPhi2
from audio_processor import AudioProcessor
from PIL import Image
import io

app = Flask(__name__)

# Load your fine-tuned model
model = MultimodalPhi2("path/to/your/fine-tuned/model", clip_embedding_dim=512, whisper_embedding_dim=768, phi2_hidden_dim=2560)
model.eval()

# Initialize audio processor
audio_processor = AudioProcessor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'text' in request.form:
        # Process text input
        text = request.form['text']
        response = process_text(text)
    elif 'image' in request.files:
        # Process image input
        image = request.files['image']
        response = process_image(image)
    elif 'audio' in request.files:
        # Process audio input
        audio = request.files['audio']
        response = process_audio(audio)
    else:
        return jsonify({'error': 'No input provided'}), 400

    return jsonify({'response': response})

def process_text(text):
    # Tokenize and process text
    inputs = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    response = model.tokenizer.decode(outputs.logits.argmax(dim=-1)[0], skip_special_tokens=True)
    return response

def process_image(image):
    # Process image and get CLIP embeddings
    img = Image.open(io.BytesIO(image.read()))
    # Get CLIP embeddings (you'll need to implement this part)
    clip_embeddings = get_clip_embeddings(img)
    
    # Process with the model
    inputs = model.tokenizer("Describe this image:", return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs, image_embeddings=clip_embeddings)
    response = model.tokenizer.decode(outputs.logits.argmax(dim=-1)[0], skip_special_tokens=True)
    return response

def process_audio(audio):
    # Process audio with Whisper
    audio_file = io.BytesIO(audio.read())
    transcription = audio_processor.transcribe(audio_file)
    audio_embeddings = audio_processor.get_audio_embeddings(audio_file)
    
    # Process with the model
    inputs = model.tokenizer(f"Transcription: {transcription}\nRespond to this:", return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs, audio_embeddings=audio_embeddings)
    response = model.tokenizer.decode(outputs.logits.argmax(dim=-1)[0], skip_special_tokens=True)
    return response

if __name__ == '__main__':
    app.run(debug=True)
