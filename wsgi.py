from flask import *
from melodygenerator import make_melody_response, Voice

from paths import PROGRESS_PATH, MIDI_SAVE_PATH, MP3_PATH

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/get-midi/<lyrics>")
def send_melody(lyrics):
    make_melody_response(lyrics, Voice.TENOR)
    with open(MIDI_SAVE_PATH, "rb") as midi_file:
        response = make_response(midi_file.read())
        response.headers.set('Content-Type', 'audio/midi')
        return response

@app.route("/get-mp3")
def send_mp3():
    with open(MP3_PATH, "rb") as mp3_file:
        response = make_response(mp3_file.read())
        response.headers.set('Content-Type', 'audio/mp3')
        return response


@app.route("/progress")
def get_progress():
    with open(PROGRESS_PATH, "r") as progressbar_file:
        return progressbar_file.read()