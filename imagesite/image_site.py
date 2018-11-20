#!/usr/bin python
# -*- coding:utf-8 -*-


from flask import Flask, request, jsonify
from src import utils


app = Flask(__name__)


@app.route('/path')
def list_desdir():
    dir_path = request.args.get('dir').rstrip('/')
    sub_parts = dir_path.split('/')
    if sub_parts[0] == 'dataset' or sub_parts[0] == 'project':
        return jsonify(utils.list_dir(dir_path))
    else:
        return 'Dir Path: {} is not valid.'.format(dir_path)


@app.route('/console')
def list_params():
    return jsonify(utils.list_jsons())


@app.route('/command')
def generate_command():
    dir_path = request.args.get('dir').rstrip('/')
    sub_parts = dir_path.split('/')
    if sub_parts[0] == 'dataset' or sub_parts[0] == 'project':
        return jsonify(utils.list_dir(dir_path))
    ### run command that submits to me.
    else:
        return 'Dir Path: {} is not valid.'.format(dir_path)


if __name__ == '__main__':
    app.run()

