# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 01:43:00 2020

@author: Dell
"""
import pandas as pd
import numpy as np 
import flask
import pickle
from flask import Flask, render_template, request
column_names = ['Beneficiary gender code','Beneficiary Age category code','Base DRG code','ICD9 primary procedure code','Inpatient days code','DRG quintile average payment amount','DRG quintile payment amount code']

app=Flask(__name__, template_folder = 'template')
@app.route("/")
def index():
 return flask.render_template("indexx.html")
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,7)
    to_predict = pd.DataFrame(data = to_predict, columns = column_names)
    loaded_model = pickle.load(open("rf.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]
@app.route("/predict",methods = ["POST"])
def result():
    if request.method == "POST":
        to_predict_list = request.form.to_dict(flat = False)
        to_predict_list=list(to_predict_list.values())
        result = ValuePredictor(to_predict_list)
        prediction = str(result)
        return render_template("predict.html",prediction=prediction)
if __name__ == "__main__":
 app.run(debug=True)