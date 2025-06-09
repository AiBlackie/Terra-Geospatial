# Step 1: Start with an official lightweight Python base image.
# This is like getting a clean, empty computer with Python 3.9 pre-installed.
FROM python:3.9-slim

# Step 2: Set the working directory inside the container to /app.
# This is like creating a project folder inside the clean computer.
WORKDIR /app

# Step 3: Copy your requirements file into the container's /app folder.
# We do this first to take advantage of Docker's caching.
COPY requirements.txt ./

# Step 4: Install all the Python libraries from your requirements file.
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of your application code (app.py, images, etc.) into the container.
COPY . .

# Step 6: Tell Docker what port the container will listen on. Cloud Run uses this.
EXPOSE 8080

# Step 7: The command to actually run your Streamlit application when the container starts.
# The extra flags are important for making it work correctly with Cloud Run's networking.
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
