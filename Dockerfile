FROM python:3.11-slim

# System dependencies needed by opencv-python-headless and video I/O.
# libgl1 / libglib2.0-0 are included defensively since some opencv builds
# still dlopen libGL even in "headless" mode.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the package (core deps + streamlit) before copying app code so
# dependency layers are cached independently of app-code changes.
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir ".[app]"

COPY app/ app/

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
