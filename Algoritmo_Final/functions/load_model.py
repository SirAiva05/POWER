import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity
tf.compat.v1.logging.ERROR
warnings.filterwarnings('ignore', module='tensorflow')

from keras.models import model_from_json

def load_model(hours_ph, lb):
    
    if hours_ph == 2:
        
        if lb == 2: 
            
            # load json and create model
            json_file = open('models/SimpleRNN_ph2_lb2_model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            
            loaded_model = model_from_json(loaded_model_json)
            
            # load weights into new model
            loaded_model.load_weights("models/SimpleRNN_ph2_lb2_model.h5")
      
        elif lb == 4: 
            
            # load json and create model
            json_file = open('models/SimpleRNN_ph2_lb4_model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            
            loaded_model = model_from_json(loaded_model_json)
            
            # load weights into new model
            loaded_model.load_weights("models/SimpleRNN_ph2_lb4_model.h5")
            
        elif lb == 6: 
            
            # load json and create model
            json_file = open('models/SimpleRNN_ph2_lb6_model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            
            loaded_model = model_from_json(loaded_model_json)
            
            # load weights into new model
            loaded_model.load_weights("models/SimpleRNN_ph2_lb6_model.h5")

        elif lb == 8: 
            
            # load json and create model
            json_file = open('models/SimpleRNN_ph2_lb8_model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            
            loaded_model = model_from_json(loaded_model_json)
            
            # load weights into new model
            loaded_model.load_weights("models/SimpleRNN_ph2_lb8_model.h5")

        elif lb == 12: 
        
            # load json and create model
            json_file = open('models/SimpleRNN_ph2_lb12_model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            
            loaded_model = model_from_json(loaded_model_json)
            
            # load weights into new model
            loaded_model.load_weights("models/SimpleRNN_ph2_lb12_model.h5")

    if hours_ph == 4: 
        
        if lb == 4: 
        
            # load json and create model
            json_file = open('models/SimpleRNN_ph4_lb4_model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            
            loaded_model = model_from_json(loaded_model_json)
            
            # load weights into new model
            loaded_model.load_weights("models/SimpleRNN_ph4_lb4_model.h5")

        elif lb == 8: 
        
            # load json and create model
            json_file = open('models/SimpleRNN_ph4_lb8_model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            
            loaded_model = model_from_json(loaded_model_json)
            
            # load weights into new model
            loaded_model.load_weights("models/SimpleRNN_ph4_lb8_model.h5")

        elif lb == 12: 
        
            # load json and create model
            json_file = open('models/SimpleRNN_ph4_lb12_model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            
            loaded_model = model_from_json(loaded_model_json)
            
            # load weights into new model
            loaded_model.load_weights("models/SimpleRNN_ph4_lb12_model.h5")

    if hours_ph == 12: 
        
        if lb == 12:
            
            # load json and create model
            json_file = open('models/SimpleRNN_ph12_lb12_model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            
            loaded_model = model_from_json(loaded_model_json)
            
            # load weights into new model
            loaded_model.load_weights("models/SimpleRNN_ph12_lb12_model.h5")

        elif lb == 24:
            
            # load json and create model
            json_file = open('models/SimpleRNN_ph12_lb24_model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            
            loaded_model = model_from_json(loaded_model_json)
            
            # load weights into new model
            loaded_model.load_weights("models/SimpleRNN_ph12_lb24_model.h5")

    return loaded_model