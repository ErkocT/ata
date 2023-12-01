from flask import Flask, render_template, request
from  tensorflow.keras.preprocessing.image import load_img
from  tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
#from keras.applications.vgg19 import decode_predictions
from  tensorflow.keras.models import  load_model, Model
import numpy as np
import pickle
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from tensorflow.keras.optimizers import Adam


#model_path = 'models/model_1pickle.pkl'

#model = load_model(model_path, compile=True)
# Load the model using pickle
#with open(model_path, 'rb') as file:
    #model = pickle.load(file)

app = Flask(__name__)

model = load_model("model_deployment_test1.keras")
"""  
# Load the VGG19
base_model = VGG19(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
    )

# custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu')(x)

predictions = Dense(3, activation='softmax')(x)

# final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the VGG19 model
for layer in base_model.layers:
    layer.trainable = False
# Adam
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
 

model.load_weights("model_deployment_test1.weights.h5")
"""
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    
   
    CLASSES = ['COVID-19', 'Non-COVID', 'Normal']
    result_label = CLASSES[predicted_class]

    return render_template('index.html', prediction=result_label)

if __name__ == '__main__':
    app.run(port=3000, debug=True)