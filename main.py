import os
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask import Flask, send_file

app = Flask(__name__)
model = None
index_to_breed_map = None

@app.route("/")
def index():
    return send_file('src/index.html')

def get_model(model_path, model):
    if os.path.exists(model_path):
        try :
            model = load_model(model_path)
            return model
        except Exception as e :
            print(f"Error memuat model dari {model_path}: {e}")
    else:
        print(f"Error: File model tidak ditemukan di {model_path}")
    return None

def get_mapping(mapping_path, index_to_breed_map):
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'rb') as f:
                index_to_breed_map = pickle.load(f)
            return index_to_breed_map
        except Exception as e:
            print(f"Error memuat mapping dari {mapping_path}: {e}")
    else:
        print(f"Error: File mapping tidak ditemukan di {mapping_path}")
    return None


def main():
    app.run(port=int(os.environ.get('PORT', 80)))

if __name__ == "__main__":
    main()
