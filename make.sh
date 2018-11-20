#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Author: Donny You(donnyyou@163.com)

pip3 install -r requirements.txt

python setup.py develop

cd extensions/apis/cocoapi/PythonAPI
python setup.py install