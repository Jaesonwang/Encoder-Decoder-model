import time
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, World!'

if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(host='0.0.0.0', port=5000)
    while True:
        time.sleep(10)

    print("Flask app stopped.")