# Use a smaller base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies first (faster builds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY multi_task_transformer.py .
COPY main.ipynb .

# Run your script (modify as needed)
CMD ["python", "multi_task_transformer.py"]
