import random
from flask import Flask, make_response
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
import json
import requests


app = Flask(__name__)
CORS(app)

api = Api(app)

parser = reqparse.RequestParser()

#@app.after_request
#def set_csp_header(response):
#    response.headers['Content-Security-Policy'] = "connect-src http://127.0.0.1:5000 *.facebook.com facebook.com *.fbcdn.net *.facebook.net wss://*.facebook.com:* ws://localhost:* blob: *.instagram.com *.cdninstagram.com wss://*.instagram.com:* 'self' *.teststagram.com wss://edge-chat.instagram.com connect.facebook.net"
#    return response


@app.route('/')
def hello_world():
    return 'Hello, World! Scam Checker API'



class ScamChecker(Resource):
    def get(self):
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = 'https://www.instagram.com'

        return {"message": "Hello, World!"}, 200

    def post(self):
        parser.add_argument("comment_id", type=str, required=True, help="comment_id is required")
        parser.add_argument("comment_text", type=str, required=True, help="comment_text is required")
        args = parser.parse_args()

        comment_id = args["comment_id"]
        comment_text = args["comment_text"]

        # Testing only
        # score = random number between 10 and 100        
        # score = random.randint(10, 100)
        
        API_TOKEN = "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"
        MODEL = "svalabs/twitter-xlm-roberta-crypto-spam"
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
        API_URL = "https://api-inference.huggingface.co/models/" + MODEL
        payload = {"inputs": comment_text}
        data = json.dumps(payload)
        response = requests.request("POST", API_URL, headers=headers, data=data)
        data = json.loads(response.content.decode("utf-8"))

        positive_score = None
        negative_score = None
        for d in data[0]:
            if d['label'] == 'HAM':
                positive_score = d['score']
            elif d['label'] == 'SPAM':
                negative_score = d['score']

        score = 100 - (negative_score * 100)
        # round score to 0 decimal places down
        score = int(score)

        # Access-Control-Allow-Origin
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = 'https://www.instagram.com'


        return {"comment_id": comment_id, "score": score}, 201


api.add_resource(ScamChecker, '/scam/')


if __name__ == "__main__":
  app.run(debug=True, host='0.0.0.0')



