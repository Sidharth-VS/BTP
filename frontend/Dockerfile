FROM python:3.10-slim
WORKDIR /app
# Added pandas for telemetry data manipulation
RUN pip install --no-cache-dir streamlit requests pandas
# Copy the entire directory to capture app.py AND the pages/ directory
COPY . /app/
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]