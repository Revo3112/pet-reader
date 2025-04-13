from flask import Flask, request, abort, send_file, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import pickle
import io  # Tambahkan import io

app = Flask(__name__)

def get_model(model_path):
    if os.path.exists(model_path):
        try:
            loaded_model = load_model(model_path)
            return loaded_model
        except Exception as e:
            print(f"Error memuat model dari {model_path}: {e}")
    else:
        print(f"Error: File model tidak ditemukan di {model_path}")
    return None

def get_mapping(mapping_path):
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'rb') as f:
                loaded_index_to_breed_map = pickle.load(f)
            return loaded_index_to_breed_map
        except Exception as e:
            print(f"Error memuat mapping dari {mapping_path}: {e}")
    else:
        print(f"Error: File mapping tidak ditemukan di {mapping_path}")
    return None

@app.route("/")
def index():
    return send_file('src/index.html')

@app.route('/predict', methods=['POST'])
def predict_breed():
    more_confidence = False
    if 'image_upload' not in request.files:
        abort(400, description="Tidak ada file yang dikirim.")

    file = request.files['image_upload']  # <--- Mengambil file sesuai nama 'image_upload'
    model = get_model('model/final_pet_classifier_model_v3.keras')
    index_to_breed_map = get_mapping('mapping/index_to_breed_map.pkl')
    
    if model is None or index_to_breed_map is None:
        abort(500, description="Gagal memuat model atau mapping.")
    
    # Pindahkan pointer ke awal file sebelum membaca (untuk jaga-jaga)
    file.seek(0)

    # Bungkus file ke BytesIO
    img_bytes = io.BytesIO(file.read())
    # Pastikan pointer di awal
    img_bytes.seek(0)

    # Load gambar dari BytesIO
    target_size = (224, 224)
    img = image.load_img(img_bytes, target_size=target_size)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    while(more_confidence != True):
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = int(np.argmax(predictions[0]))
        predicted_class_prob = float(predictions[0][predicted_class_idx])
        
        predict_breed = index_to_breed_map.get(predicted_class_idx)
        if predict_breed is None:
            predict_breed = "Tidak dapat mendeteksi ras kucing atau anjing."
            more_confidence = True
        if predicted_class_prob > 0.1:
            more_confidence = True
    
    return jsonify({
        "predicted_bread": predict_breed,
        "predicted_class_prob": round(predicted_class_prob, 4)
    })

def main():
    app.run(port=int(os.environ.get('PORT', 80)))

if __name__ == "__main__":
    main()
