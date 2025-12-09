from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os
from PIL import Image
from datetime import datetime
from flask_cors import CORS
import traceback

app = Flask(__name__)
CORS(app)

# =========================
# LOAD MODEL
# =========================
modelnasnet = load_model("NASNetMobile.h5", compile=False)
modelvgg = load_model("VGG16.h5", compile=False)
modelxception = load_model("Xception.h5", compile=False)
modelcnn = load_model("scratchCNN.h5", compile=False)

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tiff', 'webp', 'jfif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# =========================
# ROUTES
# =========================
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("cnn.html")

@app.route("/classification", methods=['GET', 'POST'])
def classification():
    return render_template("classifications.html")

@app.route('/submit', methods=['POST'])
def predict():
    try:
        # cek input file
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No image in the request'
            }), 400

        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No selected file'
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'message': f'File type {file.filename.split(".")[-1]} is not allowed'
            }), 400

        # Simpan file dengan timestamp untuk menghindari conflict
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        
        # Buat folder jika belum ada
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Simpan file
        file.save(file_path)

        # =========================
        # CONVERT IMAGE TO RGB & SAVE AS NEW FILE
        # =========================
        img = Image.open(file_path).convert('RGB')
        
        # Buat nama file untuk preview
        preview_filename = f"preview_{timestamp}.png"
        preview_path = os.path.join(app.config['UPLOAD_FOLDER'], preview_filename)
        img.save(preview_path, format="png")
        img.close()

        # =========================
        # PREPROCESS IMAGE
        # =========================
        img = load_img(preview_path, target_size=(128, 128))
        x = img_to_array(img)
        x = x / 127.5 - 1
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])

        # =========================
        # PREDICT
        # =========================
        class_names = ['With Pest', 'Without Pest']
        
        predictions = {}
        
        # Cek model mana yang dipilih
        use_xception = request.form.get('xception', 'true').lower() == 'true'
        use_vgg = request.form.get('vgg16', 'true').lower() == 'true'
        use_nasnet = request.form.get('nasnet', 'true').lower() == 'true'
        use_cnn = request.form.get('cnn', 'true').lower() == 'true'

        if use_xception:
            prediction_array_xception = modelxception.predict(images, verbose=0)
            predictions['xception'] = {
                'prediction': class_names[np.argmax(prediction_array_xception)],
                'confidence': float(np.max(prediction_array_xception)) * 100
            }

        if use_vgg:
            prediction_array_vgg = modelvgg.predict(images, verbose=0)
            predictions['vgg'] = {
                'prediction': class_names[np.argmax(prediction_array_vgg)],
                'confidence': float(np.max(prediction_array_vgg)) * 100
            }

        if use_nasnet:
            prediction_array_nasnet = modelnasnet.predict(images, verbose=0)
            predictions['nasnet'] = {
                'prediction': class_names[np.argmax(prediction_array_nasnet)],
                'confidence': float(np.max(prediction_array_nasnet)) * 100
            }

        if use_cnn:
            prediction_array_cnn = modelcnn.predict(images, verbose=0)
            predictions['cnn'] = {
                'prediction': class_names[np.argmax(prediction_array_cnn)],
                'confidence': float(np.max(prediction_array_cnn)) * 100
            }

        return jsonify({
            'success': True,
            'image_url': f"/{preview_path}",
            'predictions': predictions
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'An error occurred: {str(e)}'
        }), 500

# =========================
# RUN APP
# =========================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)