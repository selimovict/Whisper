import ffmpeg
from tempfile import NamedTemporaryFile
from pydub import AudioSegment
from pydub.exceptions import PydubException

print('Tarik')

import whisper
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

model = whisper.load_model("medium")

from flask import Flask, abort, request
from flask_cors import CORS
from tempfile import NamedTemporaryFile
import torch

# Check if NVIDIA GPU is available
torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


app = Flask(__name__)
CORS(app)

@app.route("/")
def hello():
    return "Whisper API Project"

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg'}  # Add more extensions as needed

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_mp3(input_path, output_path):
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="mp3")
    except PydubException as convert_error:
        print(f"Error converting to MP3: {str(convert_error)}")
        raise  # Re-raise the exception to propagate it up

@app.route('/whisper', methods=['POST'])
def handler():
    try:
        if not request.files:
            # If the user didn't submit any files, return a 400 (Bad Request) error.
            abort(400)
        results = []
        resultContent=[]
        Language=[]
        # Loop over every file that the user submitted.
        for filename, handle in request.files.items():
            #if not allowed_file(filename):
            #    return {'Message': 'Extension is not allowed'}
            # Create a temporary file.
            # The location of the temporary file is available in `temp.name`.
            temp = NamedTemporaryFile()

            # Write the user's uploaded file to the temporary file.
            # The file will get deleted when it drops out of scope.


            handle.save(temp)

            mp3_temp = NamedTemporaryFile(delete=False, suffix=".mp3")
            convert_to_mp3(temp.name, mp3_temp.name)

            results.append(mp3_temp.name)

            audio = whisper.load_audio(mp3_temp.name)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            _, probs = model.detect_language(mel)
            DetectedLanguge=f"Detected language: {max(probs, key=probs.get)}"
            Language.append(DetectedLanguge)
            options = whisper.DecodingOptions(fp16=False)
            result_Content = whisper.decode(model, mel, options)
            resultContent.append(result_Content.text)


        return {'results': results,'resultContent':resultContent,'language':Language}
    except Exception as e:
        return {'exception':e}

if __name__ == '__main__':


	app.run( port=9000,debug=False)



#
# audio = whisper.load_audio("Audio.mp3.m4a")
# audio = whisper.pad_or_trim(audio)
#
# # make log-Mel spectrogram and move to the same device as the model
# mel = whisper.log_mel_spectrogram(audio).to(model.device)
#
# # detect the spoken language
# _, probs = model.detect_language(mel)
# print(f"Detected language: {max(probs, key=probs.get)}")
#
# # decode the audio
# options = whisper.DecodingOptions(fp16 = False)
# result = whisper.decode(model, mel, options)
#
# # print the recognized text
# print(result.text)
