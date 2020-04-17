FROM python:3.7
ENV OPENCV_VERSION=4.1.1
# general
RUN apt-get update
RUN apt-get install -y build-essential cmake \
    wget git unzip

# Install all dependencies for OpenCV
RUN apt-get -y update && \
    apt-get -y install \
        tesseract-ocr \
        python3-dev \
        git \
        wget \
        unzip \
        cmake \
        build-essential \
        pkg-config \
        libatlas-base-dev \
        gfortran \
        libgtk2.0-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libv4l-dev

# numpy etc
RUN pip install wheel && \
    pip install numpy && \
    pip install pandas && \
    pip install scipy && \
    pip install scikit-learn && \
    pip install python-magic

# lib for tesseract
RUN apt-get update && \
    apt-get -y install \
        g++ \
        autoconf \
        automake \
        libtool \
        autoconf-archive \
        zlib1g-dev \
        libicu-dev \
        libpango1.0-dev \
        libcairo2-dev

# tesseract 4
RUN apt-get update && apt-get install -y libleptonica-dev \
    libtesseract4 \
    libtesseract-dev \
    tesseract-ocr

# Get language data.
RUN apt-get install -y \
    tesseract-ocr-eng
    # add more if needed

RUN apt-get -y clean all && \
    rm -rf /var/lib/apt/lists/* && \

    # Install OpenCV
    wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip -O opencv3.zip && \
    unzip -q opencv3.zip && \
    mv /opencv-$OPENCV_VERSION /opencv && \
    rm opencv3.zip && \
    wget https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip -O opencv_contrib3.zip && \
    unzip -q opencv_contrib3.zip && \
    mv /opencv_contrib-$OPENCV_VERSION /opencv_contrib && \
    rm opencv_contrib3.zip \
    && \

    # Clean
    apt-get -y remove \
        python3-dev \
        libatlas-base-dev \
        gfortran \
        libgtk2.0-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libv4l-dev \
        && \
    apt-get clean && \
    rm -rf /opencv /opencv_contrib /var/lib/apt/lists/*



COPY ./ ./app
WORKDIR ./app

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3"]
CMD ["run_sd.py", "/data", "/data/output_slates.csv"]