from flask import Flask, request, abort, send_file, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import pickle
import io
from dotenv import load_dotenv
import wikipedia
import time
import json
import mapping_animal as MA
from groq import Groq
from typing import List, Optional
from pydantic import BaseModel

app = Flask(__name__)
load_dotenv()

# Initialize Groq client
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

model = ['deepseek-r1-distill-llama-70b','groq-llama3-70b']

# Define Pydantic models for structured response
class FoodRecommendation(BaseModel):
    jenis: str
    frekuensi: str

class HealthRecommendation(BaseModel):
    vaksin: List[str]
    checkup: str

class GroomingRecommendation(BaseModel):
    mandi: str
    sisir: str

class PetCare(BaseModel):
    makanan: FoodRecommendation
    kesehatan: HealthRecommendation
    grooming: GroomingRecommendation
    aktivitas: str
    peringatan: List[str]

class PetCareRecommendation(BaseModel):
    ras: str
    perawatan: PetCare
    tips_khusus: str
    sumber: str

def get_model(model_path):
    print(f"Trying to load model from: {os.path.abspath(model_path)}")
    if os.path.exists(model_path):
        try:
            loaded_model = load_model(model_path)
            print(f"Successfully loaded model from {model_path}")
            return loaded_model
        except Exception as e:
            print(f"Error memuat model dari {model_path}: {e}")
    else:
        print(f"Error: File model tidak ditemukan di {model_path}")
    return None

def get_mapping(mapping_path):
    print(f"Trying to load mapping from: {os.path.abspath(mapping_path)}")
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'rb') as f:
                loaded_index_to_breed_map = pickle.load(f)
            print(f"Successfully loaded mapping from {mapping_path}")
            return loaded_index_to_breed_map
        except Exception as e:
            print(f"Error memuat mapping dari {mapping_path}: {e}")
    else:
        print(f"Error: File mapping tidak ditemukan di {mapping_path}")
    return None

def get_konteks_wiki(breed, sentences=5):
    try:
        # Tambahkan 'cat' atau 'dog' berdasarkan ras yang dikenal
        hewan = MA.get_mapping(breed)
        return wikipedia.summary(f"{breed} {hewan}", sentences=sentences)
    except wikipedia.exceptions.DisambiguationError as e:
        return wikipedia.summary(e.options[0], sentences=sentences)
    except Exception as e:
        return f"Informasi spesifik untuk {breed} tidak ditemukan. {str(e)}"

def get_care_recommendations(breed, context, max_retries=3):
    """Get care recommendations using Groq API with schema validation"""
    try:
        # Create system prompt with specific instructions
        system_prompt = f"""You are a professional pet care specialist with extensive experience in breed-specific care. 
        Please provide detailed recommendations for {breed} based on the following information:

        {context[:2000]}...

        You must output a valid JSON object that follows the provided schema exactly."""

        # Get the schema from our Pydantic model
        schema = PetCareRecommendation.model_json_schema()
        
        # Make API call to Groq
        retry_count = 0
        backoff_time = 1
        
        while retry_count < max_retries:
            try:
                chat_completion = groq.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": f"Provide care recommendations for {breed} breed"
                        }
                    ],
                    model = model[retry_count],  # You can also use other models like "mixtral-8x7b-32768" or "gemma-7b-it"
                    temperature=0.3,
                    response_format={"type": "json_object"},
                    stream=False
                )
                
                # Extract and validate response
                response_content = chat_completion.choices[0].message.content
                
                # Validate response against our schema
                validated_data = PetCareRecommendation.model_validate_json(response_content)
                
                # Return the validated JSON string and model name
                return response_content, model[retry_count]
                
            except Exception as e:
                print(f"Error with Groq API, attempt {retry_count+1}: {str(e)}")
                retry_count += 1
                
                if retry_count < max_retries:
                    print(f"Retrying in {backoff_time} seconds...")
                    time.sleep(backoff_time)
                    backoff_time *= 2  # Exponential backoff
                else:
                    print(f"All retries failed for Groq API")
                    break
    
    except Exception as e:
        print(f"Unexpected error in get_care_recommendations: {str(e)}")
    
    # If everything fails, return a generic recommendation
    generic_response = {
        "ras": breed,
        "perawatan": {
            "makanan": {"jenis": "High-quality pet food appropriate for breed", "frekuensi": "Follow veterinarian recommendations"},
            "kesehatan": {"vaksin": ["Core vaccines as recommended by vet"], "checkup": "Yearly"},
            "grooming": {"mandi": "As needed", "sisir": "Regular brushing"},
            "aktivitas": "Regular exercise appropriate for breed",
            "peringatan": ["Consult with veterinarian for breed-specific concerns"]
        },
        "tips_khusus": "Consult with a professional veterinarian for personalized advice",
        "sumber": "Generic recommendations"
    }
    
    return json.dumps(generic_response), "generic_fallback"

@app.route("/")
def index():
    return send_file('src/index.html')

@app.route('/predict', methods=['POST'])
def predict_breed():
    print("Starting prediction...")
    if 'image_upload' not in request.files:
        abort(400, description="Tidak ada file yang dikirim.")

    file = request.files['image_upload']
    
    # Debug current working directory
    print(f"Current directory: {os.getcwd()}")
    
    # Load the first model
    model_path = 'model/final_pet_classifier_model_v3.keras'
    mapping_path = 'mapping/index_to_breed_map.pkl'
    
    model = get_model(model_path)
    index_to_breed_map = get_mapping(mapping_path)
    
    if model is None:
        abort(500, description=f"Gagal memuat model dari {model_path}")
    if index_to_breed_map is None:
        abort(500, description=f"Gagal memuat mapping dari {mapping_path}")
    
    # Prepare and process image for the first model
    file.seek(0)
    img_bytes = io.BytesIO(file.read())
    img_bytes.seek(0)
    
    target_size = (224, 224)
    img = image.load_img(img_bytes, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    # Make prediction with the first model
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = int(np.argmax(predictions[0]))
    predicted_class_prob = float(predictions[0][predicted_class_idx])
    
    predict_breed = index_to_breed_map.get(predicted_class_idx)
    if predict_breed is None:
        predict_breed = "Tidak dapat mendeteksi ras kucing atau anjing."
    
    used_model = 'basic_model'
    
    # Now load the efficient model and try second prediction
    try:
        efficient_model_path = 'model/final_EfficientNetB3_model.keras'
        efficient_mapping_path = 'mapping/index_to_breed_mapefficient.pkl'
        
        efficient_model = get_model(efficient_model_path)
        index_to_breed_map_efficient = get_mapping(efficient_mapping_path)
        
        if efficient_model is not None and index_to_breed_map_efficient is not None:
            # Process image for the efficient model
            img_bytes.seek(0)
            target_size = (300, 300)
            img = image.load_img(img_bytes, target_size=target_size, interpolation='nearest')
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            
            # Make prediction with the efficient model
            predictions_efficient = efficient_model.predict(img_array, verbose=0)
            prediction_efficient_index = np.argmax(predictions_efficient[0])
            confidence = float(predictions_efficient[0][prediction_efficient_index])
            
            predicted_breed_efficient = index_to_breed_map_efficient.get(prediction_efficient_index, 
                                                            f"ERROR: Index {prediction_efficient_index} not in map!")
            
            # Select the best prediction
            if predicted_breed_efficient == predict_breed:
                if confidence > predicted_class_prob:
                    predicted_class_prob = confidence
                    used_model = 'EfficientNetB3'
            elif confidence > predicted_class_prob:
                predict_breed = predicted_breed_efficient
                predicted_class_prob = confidence
                used_model = 'EfficientNetB3'
        else:
            print("Skipping efficient model prediction due to loading failure")
    except Exception as e:
        print(f"Error during efficient model prediction: {e}")
        # Continue with the basic model prediction only
    
    # Get Wikipedia context
    konteks = get_konteks_wiki("purebred" + predict_breed)
    
    # Get care recommendations with Groq instead of Hugging Face
    care_recommendations, model_used = get_care_recommendations(predict_breed, konteks)

    # Return JSON response
    return jsonify({
        "predicted_breed": predict_breed,
        "confidence": round(predicted_class_prob * 100, 2),
        "model_used": used_model,
        "breed_info": konteks,
        "care_recommendations": care_recommendations,
        "recommendations_source": model_used
    })

def main():
    port = int(os.environ.get('PORT', 80))
    app.run(host='0.0.0.0', port=port, debug=True)

if __name__ == "__main__":
    main()