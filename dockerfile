# Use a slim Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Expose default port
ENV PORT 5000
EXPOSE 5000

# Launch
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT"]
