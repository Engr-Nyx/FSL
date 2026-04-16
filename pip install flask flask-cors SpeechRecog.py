pip install flask flask-cors SpeechRecognition googletrans==4.0.0-rc1 pyttsx3

from flask import Flask, request, jsonify
from flask_cors import CORS
import speech_recognition as sr
from googletrans import Translator
import pyttsx3

app = Flask(_name_)
CORS(app)

translator = Translator()
engine = pyttsx3.init()

@app.route('/interpret', methods=['POST'])
def interpret():
    data = request.json
    text = data['text']
    target_lang = data['language']

    translated = translator.translate(text, dest=target_lang)

    engine.say(translated.text)
    engine.runAndWait()

    return jsonify({
        "original": text,
        "translated": translated.text
    })

if _name_ == '_main_':
    app.run(debug=True)