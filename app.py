from flask import Flask, request, render_template, send_from_directory
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load custom YOLOv8 model using SAHI
model_path = 'models/fold-2.pt'
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=model_path,
    confidence_threshold=0.2,
    device='cpu',  # Use GPU if available
    config_path=None  # Adjust if necessary
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Process image with SAHI and YOLOv8
        sahi_result = get_sliced_prediction(
            file_path,
            detection_model,
            slice_height=900,
            slice_width=900,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )

        # Check input image format
        with Image.open(file_path) as img:
            image_format = img.format

        # Save SAHI result with the same format as input image
        sahi_result_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'sahi_result')
        sahi_result.export_visuals(export_dir=app.config['UPLOAD_FOLDER'], file_name='sahi_result')

        # Count objects detected
        object_counts = {}
        for prediction in sahi_result.object_prediction_list:
            label = prediction.category.name
            if label in object_counts:
                object_counts[label] += 1
            else:
                object_counts[label] = 1

         # Hapus file yang diupload setelah selesai digunakan
        os.remove(file_path)

        return render_template('results.html', sahi_image='sahi_result.png', object_counts=object_counts)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)