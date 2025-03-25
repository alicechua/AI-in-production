# Use a Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /frontend

# Copy the requirements.txt into the container
COPY requirements.txt /frontend/

# Install the dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire frontend application code into the container
COPY . /frontend/

# Expose port 7860 for the Gradio interface
EXPOSE 7860

# Command to run the Gradio app
CMD ["python", "gradio_interface.py"]