# Base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy app files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit on container start
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
