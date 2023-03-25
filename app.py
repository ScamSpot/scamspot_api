from flask import Flask, request, make_response

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World! New version'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')