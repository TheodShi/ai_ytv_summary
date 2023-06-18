# generate a summary for a youtube video

# eg. python gs.py --url="http://y2u.be/49gDWE9pB5k"

import os
import openai
from dotenv import load_dotenv, find_dotenv
import argparse
# from mimetypes import guess_extension
# from slugify import slugify
from pytube import YouTube
from pytube.exceptions import VideoUnavailable
from tempfile import TemporaryDirectory
from faster_whisper import WhisperModel

parser = argparse.ArgumentParser(description='generate a summary for a youtube video.')

parser.add_argument('-u', '--url', type=str)

args = parser.parse_args()

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
    #print(result)
    return result

def generate_summary(speech):
    _ = load_dotenv(find_dotenv()) # read local .env file
    openai.api_key  = os.environ['OPENAI_API_KEY']

    def get_completion(prompt, model="gpt-3.5-turbo-16k"):
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message["content"]

    delimiter = "####"
    prompt = f"""
        请根据下面的讲话内容生成一个中文的摘要，讲话的内容被包含在{delimiter}分隔符内.
        生成的摘要请遵循下面的条件:
            a. 不要包含打招呼和感谢的内容
            b. 生成的摘要内容要尽可能清晰,简洁
            c. 讲话内容如果包含多个结论,请使用a,b,c这样的方式去分别罗列每个结论
        
        {delimiter}
        {speech}
        {delimiter}
    """
    #print(prompt)
    return get_completion(prompt)

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
            return generate_summary(speech)


summary = parse_video(args.url)

print(summary)

