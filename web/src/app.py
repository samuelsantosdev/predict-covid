import datetime
from flask import Flask, render_template, url_for, make_response
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    r = make_response(render_template('index.html', date='?q='+str(datetime.datetime.now().timestamp()) ))
    r.headers.set("Pragma-directive", "no-cache")
    r.headers.set("Cache-directivev", "no-cache")
    r.headers.set("Cache-control", "no-cache")
    r.headers.set("Pragma", "no-cache")
    r.headers.set("Expires", "0")
    return r
