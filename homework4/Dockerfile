FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim
WORKDIR /app

# 1. Update apt and install essential system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libgfortran5 && \
    rm -rf /var/lib/apt/lists/*

# 2. Install pipenv and upgrade pip
RUN pip install --upgrade pip && \
    pip install pipenv

# 3. Copy dependency files
COPY Pipfile Pipfile.lock ./

# 4. (NEW) Remove pywinpty from Pipfile.lock before installing dependencies
#    This line uses 'sed' to delete any lines containing "pywinpty" from Pipfile.lock.
#    It's a common workaround for cross-platform Pipfile.lock issues.
RUN sed -i '/"pywinpty"/d' Pipfile.lock

# 5. Install Python dependencies from Pipfile.lock
RUN pipenv install --system --deploy

# 6. Copy application code
COPY starter.py ./

# 7. Define the command to run the application
CMD "python" "starter.py"