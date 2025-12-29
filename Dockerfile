# Use a lightweight official Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies first (for layer caching optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main application script
# Ensure your 'council.py' is in the same directory as this Dockerfile
COPY council.py .

# Expose an interactive environment (optional, useful if you want to run it live)
# If you decide to run a web service later, you'd expose a port here (e.g., 8080)

# Command to run the application when the container starts
# We use 'python -u' to ensure stdout is unbuffered, which is better for container logs
CMD ["python", "-u", "council.py"]