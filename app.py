from flask import Flask, render_template, url_for, redirect, make_response, jsonify
from datetime import datetime
from flask import request


app = Flask(__name__)


if __name__ == "__main__":
    app.run(debug = True)