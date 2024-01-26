#!/bin/bash

./configure --cc="gcc -m64" --enable-asm --disable-doc --disable-ffplay --disable-ffprobe --enable-ffmpeg --enable-shared --disable-static --disable-bzlib --disable-libopenjpeg --disable-iconv --disable-avdevice  --disable-swscale --disable-postproc --disable-avfilter  --prefix=bin --arch=--arch=x86_64

make
make install