# generate a summary for a youtube video

# eg. python gs.py --url="http://y2u.be/49gDWE9pB5k"

import os
import argparse
# from mimetypes import guess_extension
# from slugify import slugify
from pytube import YouTube
from pytube.exceptions import VideoUnavailable
from tempfile import TemporaryDirectory
from faster_whisper import WhisperModel

from llms import LLM

parser = argparse.ArgumentParser(description='generate a summary for a youtube video.')

parser.add_argument('-u', '--url', type=str)

args = parser.parse_args()

llm = LLM()

if args.url is None:
    print("URL parameter is required")

def extract_text(path):
    model_size = "medium" # "large-v2"
    # Run on GPU with FP16
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe(path, beam_size=5)
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    result = '.'.join(segment.text for segment in segments)

    return result

def parse_video(url):
    try:
        yt = YouTube(url)
    except VideoUnavailable:
        print(f'Video {url} is unavaialable')
    else:
        print(f'Downloading video: {url} - {yt.title}')

        with TemporaryDirectory(prefix="ai_video_assistant_") as tempdir:
            audio_stream = yt.streams.filter(only_audio=True).first()
            path = audio_stream.download(output_path=tempdir)
            print("The download is done, start to analyze the voice...")
            speech = extract_text(path)
            print("Speech analysis is done, start generating summaries...")
            return llm.generate_summary(speech)


summary = parse_video(args.url)

print(summary)

