#!/bin/bash
set -e

echo "runing site ..."

sudo service nginx restart

gunicorn -w4 -b0.0.0.0:8000 image_site:app

