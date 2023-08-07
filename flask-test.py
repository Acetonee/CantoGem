from flask import *
from melodygenerator import make_melody_response, MIDI_SAVE_PATH

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/get-lyrics/<lyrics>")
def send_melody(lyrics):
    make_melody_response(lyrics)
    with open(MIDI_SAVE_PATH, "r") as midi_file:
        response = make_response(midi_file)
    response.headers.set('Content-Type', 'audio/midi')
    return response

app.run()