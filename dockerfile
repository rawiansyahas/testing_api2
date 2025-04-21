# 1. Start from a slim Python base
FROM python:3.11-slim

# 2. Install system deps for Pillow, facenet-pytorch, etc.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      git \
      libjpeg-dev \
      zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

# 3. Upgrade pip
RUN pip install --upgrade pip

# 4. Set your working dir
WORKDIR /app

# 5. Copy & install Python requirements (including gunicorn)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your code
COPY . .

# 7. Tell Docker/ Railway which port to expose
ENV PORT 5000
EXPOSE 5000

# 8. Use Gunicorn to serve your Flask app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT"]
