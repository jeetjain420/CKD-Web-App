# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:28:15 2019

@author: acer pc
"""

from flask import Flask, render_template, request
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import warnings 
warnings.filterwarnings('ignore')



app = Flask(__name__)

@app.route('/',methods= ['GET'])
def index():
    return render_template('index.html')
    
@app.route('/predict',methods = ['POST'])
def predict():
    data=pd.read_csv(r"C:\Users\acer pc\Desktop\Project\featured.csv")
    y=data["classification"].values
    x_data=data.drop(["classification"],axis=1)
    x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=4,test_size=0.4)
    rf=RandomForestClassifier(n_estimators=100,random_state=1)
    rf.fit(x_train,y_train)
    
    if request.method == 'POST':
        sg= request.form['sg']
        al = request.form['al']
        pc = request.form['pc']
        sc = request.form['sc']
        hemo = request.form['hemo']
        rc = request.form['rc']
        htn = request.form['htn']
        dm = request.form['dm']
        predi = [sg,al,pc,sc,hemo,rc,htn,dm]
        my_prediction = rf.predict(predi)
        
        
    return render_template('result.html', prediction = my_prediction)
       
        
if __name__ == "__main__":
    app.run(port = 5000)
    app.debug = True
    
    