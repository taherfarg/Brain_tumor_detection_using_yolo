from flask import Flask, render_template, request, jsonify, send_file
from model import predict_tumor, get_model_info
import os
from io import BytesIO
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    model_info = get_model_info()
    return render_template('index.html', model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        # Generate a unique filename
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            prediction, confidence, image_with_box, segmentation_mask = predict_tumor(filepath)
            
            # Save the image with bounding box and segmentation
            output_filename = 'output_' + filename
            output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            image_with_box.save(output_filepath)
            
            # Save segmentation mask
            mask_filename = 'mask_' + filename
            mask_filepath = os.path.join(app.config['UPLOAD_FOLDER'], mask_filename)
            segmentation_mask.save(mask_filepath)
            
            return jsonify({
                'prediction': prediction, 
                'confidence': f"{confidence:.2%}",
                'image_path': '/get_image/' + output_filename,
                'mask_path': '/get_image/' + mask_filename,
                'original_path': '/get_image/' + filename
            })
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'File type not allowed'})

@app.route('/get_image/<filename>')
def get_image(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), mimetype='image/jpeg')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

if __name__ == '__main__':
    app.run(debug=True)