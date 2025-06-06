# FROM python:3.12.1-slim

# RUN pip install -U pip
# RUN pip install pipenv 

# WORKDIR /app
# COPY ["Pipfile","Pipfile.lock", "./"]

# RUN pipenv install --system --deploy

# COPY ["predict.py", "lin_reg.bin","./"]

# EXPOSE 9696

# ENTRYPOINT ["gunicorn", "--bind 0.0.0.0:9696", "predict:app"]
FROM python:3.12.1-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

# COPY Pipfile and Pipfile.lock from the parent directory (relative to the build context)
COPY "Pipfile" "Pipfile.lock" "/app/" 

RUN pipenv install --system --deploy

# COPY predict.py and lin_reg.bin from the current directory (relative to the build context)
COPY "web_service/predict.py" "web_service/lin_reg.bin" "/app/"

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]