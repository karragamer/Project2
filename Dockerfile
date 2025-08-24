# Use official Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into container
COPY . .

# Expose port for dashboard
EXPOSE 8050

# Command to run your app (update later if main file changes)
CMD ["python", "src/main.py"]

