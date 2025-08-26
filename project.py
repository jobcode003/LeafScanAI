import numpy as np
from keras.models import load_model
from keras.preprocessing import image
model=load_model('best_model.keras')

class_names = ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight'
    , 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Tomato__Target_Spot',
               'Tomato__Tomato_mosaic_virus','Tomato__Tomato_YellowLeaf__Curl_Virus',
               'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_healthy','Tomato_Late_blight',
               'Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite']
def process(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize just like Rescaling(1./255)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence
image__path='C:\\Users\\PC\\Desktop\\plantdata\\Dataset\\Corn_(maize)___Common_rust_\\RS_Rust 1565.JPG'
predicted_class, confidence = process(image__path)
print(f"Predicted: {predicted_class} ({confidence}% confidence)")
