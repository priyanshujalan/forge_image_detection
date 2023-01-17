from tensorflow import keras
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import PySimpleGUI as sg
import seaborn as sns

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential

sns.set(style='white', context='notebook', palette='deep')

# Initial Preparation
from PIL import Image
from pylab import *
from PIL import Image, ImageChops, ImageEnhance

#Importing the model
model = keras.models.load_model('./detection_model')

# Creating the GUI
sg.theme('BluePurple')

layout = [
		# [sg.Text('Your typed characters appear here:'),
		# sg.Text(size=(15,1), key='-OUTPUT-')],
		[sg.Input(key="-FILE-", enable_events=True, visible=True)],
		[sg.FilesBrowse("Browse File", enable_events=True, target="-FILE-"), sg.Button('Is it forged?'), sg.Button('Exit')],
        [sg.Text('No Image Detected!', key="result")]]

window = sg.Window('Project', layout)

def convert_to_ela_image(path, quality):
    filename = path
    resaved_filename = filename.split('.')[0] + '.resaved.jpg'
    ELA_filename = filename.split('.')[0] + '.ela.png'
    
    im = Image.open(filename).convert('RGB')
    im.save(resaved_filename, 'JPEG', quality=quality)
    resaved_im = Image.open(resaved_filename)
    
    ela_im = ImageChops.difference(im, resaved_im)
    
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    
    return ela_im

while True:
    event, values = window.read()
    if event in ('Exit'):
        break
    
    if event == 'Is it forged?':
        X = []
        ## preprocess image [ ELA , Normalization, Reshape ]
        X.append(array(convert_to_ela_image(values['-FILE-'], 90).resize((128, 128))).flatten() / 255.0)
        X = np.array(X)
        X = X.reshape(-1, 128, 128, 3)

        ## send preprocessed image to model

        # Predict the values from the validation dataset
        Y_pred = model.predict(X)

        # Convert predictions classes to one hot vectors 
        Y_pred_classes = np.argmax(Y_pred,axis = 1) 
        if  Y_pred_classes[0] == 0:
            window.FindElement('result').Update('Result: Authentic Image!')

        else:
            window.FindElement('result').Update('Result: Forged Image!')



window.close()

## GUI


