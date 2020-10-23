FROM pytorch/pytorch:latest
WORKDIR /app

# Copy dependencies list to container
COPY ./requirements.txt .

RUN apt-get update -y && \
    apt-get install -y nocache nano && \
    pip install -r requirements.txt

COPY . .

# # Active shell without gunicorn
# CMD ["/bin/bash"]

# For listening gunicorn service
EXPOSE 8000
ENTRYPOINT [ "sh", "docker-entrypoint.sh" ]