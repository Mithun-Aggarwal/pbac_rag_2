# Dockerfile

# Stage 1: Use an official Python runtime as a parent image.
# We use 'slim' because it's a smaller, more production-friendly version.
FROM python:3.10-slim

# Stage 2: Set the working directory inside the container to /app
# All subsequent commands will be run from this directory.
WORKDIR /app

# Stage 3: Copy the requirements file into the container.
# We copy this first to take advantage of Docker's layer caching.
# If this file doesn't change, Docker won't re-install the libraries every time.
COPY requirements.txt .

# Stage 4: Install any needed packages specified in requirements.txt
# --no-cache-dir makes the final image smaller.
RUN pip install --no-cache-dir -r requirements.txt

# Stage 5: Copy the rest of your application's code into the container.
# The '.' means copy everything from the current directory on your host machine
# into the /app directory inside the container. It will obey the .dockerignore file.
COPY . .

# This copies your pre-processed data directly into the image.
COPY ./output_folder /home/mit/Learning_and_growing/AI_DATA_EXTRACTION_AND_SEARCH_V1/Curated_information/documents/output_folder

# Stage 6: Expose the port that Streamlit runs on.
# This tells Docker that the container will listen on port 8501.
EXPOSE 8501

# Stage 7: Define the command to run your application.
# This command is executed when the Docker container starts.
# We add flags to ensure Streamlit is accessible from outside the container.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]