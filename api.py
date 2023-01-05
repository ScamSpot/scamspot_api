import random
from flask import Flask
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()

class ScamChecker(Resource):
    def get(self):        
        return "get"
    
    def post(self):
        parser.add_argument("comment_id", type=str, required=True, help="comment_id is required")
        parser.add_argument("comment_text", type=str, required=True, help="comment_text is required")
        args = parser.parse_args()

        comment_id = args["comment_id"]
        comment_text = args["comment_text"]
        
        # score = random number between 10 and 100
        score = random.randint(10, 100)
        
        return {"comment_id": comment_id, "score": score}, 201

        
api.add_resource(ScamChecker, '/scam/')


if __name__ == "__main__":
  app.run(debug=True)
  


