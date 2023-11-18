# Use Python 3.11 base image
FROM python:3.11

# Set environment variables for Python and ensure output is sent directly to terminal without buffering
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory within the container
WORKDIR /app

# Install necessary system dependencies for specified Python libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libffi-dev \
        libssl-dev \
        build-essential \
        libopenblas-dev \
        libjpeg-dev \
        zlib1g-dev \
        libpng-dev \
        libfreetype6-dev \
        libwebp-dev \
        libtiff-dev \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file to the container
COPY requirements.txt .

# Install required Python packages
RUN python -m venv env && \
    . env/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# Copy the project files to the container
COPY . .

# Expose the port on which your Flask app runs
EXPOSE 5000

# Command to run the Flask application within the container
CMD ["python", "main.py"]
