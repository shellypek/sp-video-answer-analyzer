FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD [ "uvicorn", "main:app " , "--host", "localhost", "--port", "9004"]
