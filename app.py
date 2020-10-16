# -*- coding: utf-8 -*-


import numpy as np
import os
#import re
#import glob

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input,decode_predictions
from werkzeug.utils import secure_filename
from tensorflow import keras
import tensorflow as tf
import logging as log


from flask import Flask,request,render_template

app=Flask(__name__)
model_path='model.h5'
model=load_model(model_path)

### Load the model


def model_predict(imgpath,model):
    
    img=image.load_img(imgpath,target_size=(64,64))
    
    #Preprocessing the image
    X=image.img_to_array(img)
    X=np.expand_dims(X,axis=0)
    preds=model.predict(X)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="The Weather is Cloudy"
    elif preds==1:
        preds="The Weather is Rainy"
    elif preds==2:
        preds="The Weather is Shine"
    else:
        preds="The Weather is Sunrise"
    
    
    return preds
        

        
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method=="POST":
        
    ## Get the File from the post
       f=request.files['file']
       basepath=os.path.dirname(__file__)
       file_path=os.path.join(basepath,'uploads',secure_filename(f.filename))
       f.save(file_path)
       ## Here we will make predictions
       pred=model_predict(file_path,model)
       result=pred
       return result
    return None
       

if __name__=="__main__":
    app.run(debug=True)

    
    


