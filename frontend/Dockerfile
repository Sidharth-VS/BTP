FROM python:3.10-slim
WORKDIR /app
# Install Streamlit and Requests
RUN pip install --no-cache-dir streamlit requests
# Copy the app
COPY app.py /app/app.py
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]