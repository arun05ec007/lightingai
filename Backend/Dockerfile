# 
FROM python:3.9

ENV PROJECT_DIR /usr/local/src/api
# 
WORKDIR ${PROJECT_DIR}

RUN pip install --upgrade pip 
# 
COPY ./requirements.in ${PROJECT_DIR}/requirements.in

RUN pip install pip-tools

RUN pip-compile

# 
RUN pip install --no-cache-dir -r ${PROJECT_DIR}/requirements.txt

# 
COPY ./ ${PROJECT_DIR}