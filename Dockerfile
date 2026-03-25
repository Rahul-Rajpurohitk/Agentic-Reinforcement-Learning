FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package itself
RUN pip install --no-cache-dir -e .

EXPOSE 8000

# Match their workflow: uvicorn serving the app
CMD ["uvicorn", "src.agentic_rl.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
