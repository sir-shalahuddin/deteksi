from flask import Flask, request, render_template, send_from_directory
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os
from PIL import Image

app = Flask(__name__, template_folder='../templates', static_folder='../static')
# app.config['UPLOAD_FOLDER'] = os.path.join('..', 'static', 'uploads')

# Load custom YOLOv8 model using SAHI
model_path = os.path.join('..', 'models', 'fold-2.pt')
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=model_path,
    confidence_threshold=0.2,
    device='cpu'  # Use 'cuda' if deploying with GPU support
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
        file_path = os.path.join("static/uploads", file.filename)
        os.makedirs("static/uploads/", exist_ok=True)  # Ensure the upload folder exists
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

        # Save SAHI result image
        sahi_result_img_path = os.path.join("static/uploads" ,'sahi_result.png')
        sahi_result.export_visuals("static/uploads", file_name='sahi_result')

        # Count objects detected
        object_counts = {}
        for prediction in sahi_result.object_prediction_list:
            label = prediction.category.name
            object_counts[label] = object_counts.get(label, 0) + 1

        # Remove the uploaded file after processing
        os.remove(file_path)

        return render_template('results.html', sahi_image='sahi_result.png', object_counts=object_counts)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory("../static/uploads", filename)

if __name__ == '__main__':
    app.run(debug=True)
