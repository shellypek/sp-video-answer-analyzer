FROM python:3.9

# Install FFmpeg
RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir  soundfile matplotlib moviepy opencv-python git+https://github.com/openai/whisper.git 
COPY . .
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9004"]