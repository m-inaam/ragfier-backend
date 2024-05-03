from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from bidi.algorithm import get_display

app = Flask(__name__)

# Load ARSL model
arsl_model = tf.keras.models.load_model('models/arsl_model.h5', compile=False)
arsl_categories = [
    ["ain",'ع'], ["al","ال"], ["aleff",'أ'], ["bb",'ب'], ["dal",'د'], ["dha",'ط'], ["dhad","ض"], ["fa","ف"],
    ["gaaf",'جف'], ["ghain",'غ'], ["ha",'ه'], ["haa",'ه'], ["jeem",'ج'], ["kaaf",'ك'], ["la",'لا'], ["laam",'ل'],
    ["meem",'م'], ["nun","ن"], ["ra",'ر'], ["saad",'ص'], ["seen",'س'], ["sheen","ش"], ["ta",'ت'], ["taa",'ط'],
    ["thaa","ث"], ["thal","ذ"], ["toot",' ت'], ["waw",'و'], ["ya","ى"], ["yaa","ي"], ["zay",'ز']
]

def process_arsl_image(img):
    img = cv2.resize(img, (64, 64))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, 64 , 64 , 3))
    img = img.astype('float32') / 255.
    return img

@app.route('/arsl_predict', methods=['POST'])
def arsl_predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in request'}), 400
    
    image_file = request.files['image']
    image_np = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    
    try:
        proba = arsl_model.predict(process_arsl_image(img))[0]
        mx = np.argmax(proba)
        score = proba[mx] * 100
        res = arsl_categories[mx][0]
        sequence = arsl_categories[mx][1]
                
        return jsonify({'result': res, 'score': score}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Load ASL model
asl_model = tf.keras.models.load_model("models/asl_Model.h5", compile=False)
asl_class_names = open("models/labels.txt", "r").readlines()

def process_asl_image(img):
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
    img = (img / 127.5) - 1
    return img

@app.route('/asl_predict', methods=['POST'])
def asl_predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in request'}), 400
    
    image_file = request.files['image']
    image_np = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    
    try:
        img_processed = process_asl_image(img)
        prediction = asl_model.predict(img_processed)
        index = np.argmax(prediction)
        class_name = asl_class_names[index]
        confidence_score = prediction[0][index]
        
        return jsonify({'result': class_name[2:], 'confidence': str(np.round(confidence_score * 100))[:-2]}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
