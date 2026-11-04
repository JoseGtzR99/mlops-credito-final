FROM python:3.11-slim
WORKDIR /srv
RUN pip install --no-cache-dir streamlit==1.66.1 requests==2.34.2
COPY streamlit_app.py .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
