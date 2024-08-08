# 
FROM python:latest

ENV PATH="/usr/local/bin:${PATH}" 
ENV PROJECT_DIR /usr/local/bin
# 
WORKDIR ${PROJECT_DIR}

RUN pip install --upgrade pip 
# 
COPY ./requirements.in ${PROJECT_DIR}/requirements.in

RUN pip install pip-tools

RUN pip-compile

# 
RUN pip install --no-cache-dir -r ${PROJECT_DIR}/requirements.txt

# Download spacy model
RUN spacy download en_core_web_sm


RUN echo $PATH

# 
COPY ./ ${PROJECT_DIR}