# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for eSpeak and other requirements
RUN apt-get update && apt-get install -y \
    git \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Clone the DiffRhythm repository
RUN git clone https://github.com/ASLP-lab/DiffRhythm.git .

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables for eSpeak (if needed)
ENV PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1
ENV PHONEMIZER_ESPEAK_PATH=/usr/bin

# Expose any necessary ports (if applicable)
# EXPOSE 8000

# Create a volume for input/output files
VOLUME ["/app/input", "/app/output"]

# Set the default command to run when starting the container
# You might want to modify this based on specific inference scripts
CMD ["bash"]
