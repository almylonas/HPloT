# Use Python 3.11 for better compatibility
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies that numba might need
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first
COPY requirements.txt .

# Install project dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of your app
COPY . .

# Run your app
CMD ["python", "main.py"]