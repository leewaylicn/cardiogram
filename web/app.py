from flask import Flask, render_template, url_for, request, json,jsonify                                                                                         
#from pandas import json_normalize    

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+"/src")
#sys.path.append("../src") 

#from cardiogram import CardiogramDetector  
from cardiogram import CardiogramDetector                                                                                                                      
                                                                                                                                                                 
app=Flask(__name__)                                                                                                                                              
                                                                                                                                                                 
@app.route('/')                                                                                                                                                  
def hello_flask():                                                                                                                                               
    return "<h1>Hello, Flask!</h1>"                                                                                                                              
                                                                                                                                                                 
@app.route('/infer',methods=['GET', 'POST'])                                                                                                                     
def infer():                                                                                                                                                     
    data = json.loads(request.get_data(as_text=True))                                                                                                            
    print("data is ",data)                                                                                                                                       
                                                                                                                                                                 
    #df = json_normalize(data)                                                                                                                                    
    print("path is: ")
    print(data['bucket'])
    bucket = data['bucket']
    file_name = data['file_name']
    result = CardiogramDetector(bucket, file_name).detect()
    print(result)                                                                                                                                             
                                                                                                                                                                 
    ret = {                                                                                                                                                      
        'result':str(result)
    }                                                                                                                                                            
                                                                                                                                                                 
    return json.dumps(ret)                                                                                                                                       
                                                                                                                                                                 
if __name__=='__main__':                                                                                                                                         
    app.debug=True
    app.run(host='0.0.0.0',port=5000)   