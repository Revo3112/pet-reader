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

class ActivityRecommendation(BaseModel):
    aktivitas: str

class PetCare(BaseModel):
    makanan: FoodRecommendation
    kesehatan: HealthRecommendation
    grooming: GroomingRecommendation
    aktivitas: str
    peringatan: Optional[List[str]] = []

class PetCareRecommendation(BaseModel):
    ras: str
    perawatan: PetCare
    tips_khusus: Optional[str] = None
    sumber: Optional[str] = None

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

# Penyempurnaan pada fungsi get_care_recommendations untuk memastikan format JSON yang konsisten
def get_care_recommendations(breed, context, max_retries=3):
    """Dapatkan rekomendasi perawatan untuk ras hewan tertentu."""
    retry_count = 0

    system_prompt = f"""You are a professional pet care specialist with extensive experience in breed-specific care.

    Provide detailed and specific care recommendations for {breed} based on the following information:
    {context[:2000]}...

    RESPOND IN INDONESIAN LANGUAGE with valid JSON matching EXACTLY this structure:
    {{
      "ras": "{breed}",
      "perawatan": {{
        "makanan": {{
          "jenis": "SPECIFIC food types",
          "frekuensi": "SPECIFIC feeding frequency"
        }},
        "kesehatan": {{
          "vaksin": ["SPECIFIC vaccine 1", "SPECIFIC vaccine 2"],
          "checkup": "SPECIFIC checkup frequency"
        }},
        "grooming": {{
          "mandi": "SPECIFIC bathing frequency",
          "sisir": "SPECIFIC brushing frequency"
        }},
        "aktivitas": "SPECIFIC activity recommendations",
        "peringatan": ["SPECIFIC warning 1", "SPECIFIC warning 2"]
      }},
      "tips_khusus": "SPECIFIC special tips",
      "sumber": "source of information"
    }}

    DO NOT include any extra text, explanations, or markdown. ONLY valid JSON is acceptable.
    """

    while retry_count < max_retries:
        try:
            response = groq.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Berikan rekomendasi perawatan untuk {breed} dalam format JSON sesuai petunjuk."}
                ],
                model=model[0],
                temperature=0,  # Lower temperature for more consistent formatting
                stream = False,
                max_tokens=1200,
                response_format={"type": "json_object"},
            )

            response_text = response.choices[0].message.content

            # Ambil hanya bagian JSON dalam respons
            import re
            import json

            # Coba ekstrak JSON jika dikelilingi oleh backticks
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                response_text = json_match.group(1)
            else:
                # Coba cari JSON object dengan regex
                json_match = re.search(r'({[\s\S]*})', response_text)
                if json_match:
                    response_text = json_match.group(1)

            # Parse JSON
            data = json.loads(response_text)

            # Validasi menggunakan model Pydantic
            validated_data = PetCareRecommendation(**data)

            # Debug - tambahkan log untuk melihat data yang berhasil divalidasi
            print(f"Validation successful: {validated_data.dict()}")

            return validated_data.dict(), model[0]

        except Exception as e:
            retry_count += 1
            print(f"Error getting recommendations (attempt {retry_count}/{max_retries}): {str(e)}")

            # Jika JSON parsing yang gagal, coba perbaiki
            if "JSONDecodeError" in str(e):
                print(f"JSON decode error. Raw response: {response_text}")
                # Coba "bersihkan" respons jika berisi karakter tidak valid
                cleaned_response = re.sub(r'[^\x00-\x7F]+', '', response_text)
                try:
                    data = json.loads(cleaned_response)
                    print("Berhasil parse JSON setelah dibersihkan")
                    validated_data = PetCareRecommendation(**data)
                    return validated_data.dict(), model[0]
                except Exception as inner_e:
                    print(f"Masih gagal parse JSON setelah dibersihkan: {inner_e}")

            # Jika gagal validasi Pydantic, lihat errornya
            if "ValidationError" in str(e):
                print(f"Validation error details: {str(e)}")

    # Fallback yang lebih spesifik jika semua upaya gagal
    generic_response = {
        "ras": breed,
        "perawatan": {
            "makanan": {
                "jenis": f"Royal Canin/Purina khusus {breed} atau makanan tinggi protein dengan bahan alami",
                "frekuensi": "2-3 kali sehari dalam porsi terkontrol sesuai berat badan"
            },
            "kesehatan": {
                "vaksin": ["Rabies", "Distemper", "Parvovirus", "Adenovirus", "Leptospirosis"],
                "checkup": "Setiap 6 bulan untuk pemeriksaan gigi dan kesehatan umum"
            },
            "grooming": {
                "mandi": f"Sesuai karakter bulu {breed}, umumnya 2-4 minggu sekali",
                "sisir": "2-3 kali seminggu untuk mencegah bulu kusut dan rontok"
            },
            "aktivitas": f"Minimal 30 menit aktivitas fisik setiap hari, disesuaikan dengan energi {breed}",
            "peringatan": [
                f"Perhatikan masalah genetik umum pada {breed}",
                "Jaga berat badan ideal untuk mencegah masalah sendi",
                "Perhatikan tanda-tanda alergi atau masalah kulit"
            ]
        },
        "tips_khusus": f"Ajak {breed} bersosialisasi sejak dini dan berikan stimulasi mental dengan mainan interaktif",
        "sumber": "Pedoman perawatan hewan peliharaan"
    }

    # Validasi fallback response untuk memastikan sesuai dengan model
    try:
        validated_fallback = PetCareRecommendation(**generic_response)
        return validated_fallback.dict(), 'fallback model'
    except Exception as e:
        print(f"Even fallback validation failed: {str(e)}")
        # Jika bahkan fallback gagal, kembalikan minimum data
        return {
            "ras": breed,
            "perawatan": {
                "makanan": {"jenis": "Konsultasikan dengan dokter hewan", "frekuensi": "Sesuai kebutuhan"},
                "kesehatan": {"vaksin": ["Konsultasikan dengan dokter hewan"], "checkup": "Rutin"},
                "grooming": {"mandi": "Sesuai kebutuhan", "sisir": "Rutin"},
                "aktivitas": "Aktivitas teratur"
            }
        }, "minimal fallback"

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
    konteks = get_konteks_wiki(predict_breed)

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
