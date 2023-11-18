# Use Python 3.11 base image
FROM python:3.11

# Set the working directory within the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN python -m venv env && \
    . env/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# Copy the project files to the container
COPY . .

# Expose the port on which your Flask app runs
EXPOSE 5000

# Command to run the Flask application within the container
CMD ["python", "main.py"]
