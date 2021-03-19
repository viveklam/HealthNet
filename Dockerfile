FROM python:3.7

## App engine stuff
# Expose port you want your app on
EXPOSE 8080

# Upgrade pip 
RUN pip3 install -U pip

COPY requirements.txt app/requirements.txt
RUN pip3 install -r app/requirements.txt

# Create a new directory for app (keep it in its own directory)
COPY . /app
WORKDIR app

# Run
#ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
CMD streamlit run --server.port 8080 --server.enableCORS false app.py