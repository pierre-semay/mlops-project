FROM python:3.11-slim

WORKDIR /app

# Installeer systeemafhankelijkheden die librosa vaak nodig heeft voor audio-omzetting
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Kopieer eerst de requirements en installeer de libraries (optimaal voor Docker-cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopieer NU pas de rest van je project. 
# Dit trekt je volledige mappenstructuur (inclusief main.py, index.html én de models/ map) in /app
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]