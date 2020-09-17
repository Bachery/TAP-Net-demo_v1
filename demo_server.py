from flask import Flask, send_file, send_from_directory, request, json

import numpy as np
import time
import datetime
import os

import test

app = Flask(__name__)


@app.route('/demo/js/<path:path>')
def send_js(path):
    return send_from_directory('demo/js', path)

@app.route('/demo/css/<path:path>')
def send_css(path):
    return send_from_directory('demo/css', path)
    
@app.route('/data/<path:path>')
def send_data(path):
    return send_from_directory('data', path)

@app.route('/img/<path:path>')
def send_img(path):
    return send_from_directory('img', path)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('./demo', 'favicon.ico')

@app.route('/demo/<path:path>')
def send_demo(path):
	return send_from_directory('./demo', path)

@app.route('/test_result/<path:path>')
def send_result(path):
	return send_from_directory('test_result', path)

@app.route('/demo')
def index():
	print('index')
	return send_file('demo/index.html')

@app.route('/startsign', methods=['GET', 'POST'])
def start():
	if request.method == 'POST':
		data = request.get_data()
		data = json.loads(data)
		save_dir = data['save_dir']
		test.test_pack(save_dir)
		return '666'
		
		# save_dir = test.test_pack()
		# print(save_dir)
		# msg = {}
		# msg['save_dir'] = save_dir
		# return json.dumps(msg)


if __name__=="__main__":
	app.run(
		host='0.0.0.0',
		port= 5000,
		debug=False
	)