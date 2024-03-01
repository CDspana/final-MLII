# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 00:43:39 2024

@author: user
"""

from flask import Flask, render_template, request
from qa_module import QA  # Import your QA class

app = Flask(__name__)

# Initialize your QA instance
qa_instance = QA(r"C:\Users\user\Documents\MLDS\proyecto final\doc.txt")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_responses', methods=['POST'])
def get_responses():
    question = request.form['question']
    responses = qa_instance.get_responses(question)
    return render_template('responses.html', responses=responses)

if __name__ == '__main__':
    app.run(debug=False)
