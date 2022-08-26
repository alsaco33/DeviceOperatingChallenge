FROM tensorflow/tensorflow:2.9.1-gpu

RUN apt-get update
RUN python3 -m pip install --upgrade pip

# Install requried python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Create a /work directory within the container, copy everything from the
# build directory and switch there.
RUN mkdir /work
COPY . /work
WORKDIR /work

# Make sure your scripts should be executable within the container.
RUN chmod +x test.sh
RUN chmod +x train.sh