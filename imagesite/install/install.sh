#!/bin/bash
set -e

#sudo pip install -r  requirements.txt

sudo apt-get install nginx

sudo cp imagesite.conf /etc/nginx/conf.d/

sudo nginx -s reload
