from flask import Blueprint, request
from flask_restful import Resource, Api
from sqlalchemy import exc

from project import db
from project.api.models import Run

runs_blueprint = Blueprint('runs', __name__)
api = Api(runs_blueprint)


class RunsPing(Resource):
    def get(self):
        return {
        'status': 'success',
        'message': 'pong!'
    }


class RunsList(Resource):
    def post(self):
        post_data = request.get_json()
        response_object = {
            'status': 'fail',
            'message': 'Invalid payload.'
        }
        if not post_data:
            return response_object, 400

        try:
            run = Run(**post_data)
        except TypeError:
            return response_object, 400

        try:
            db.session.add(run)
            db.session.commit()
            response_object['status'] = 'success'
            response_object['message'] = 'New run was added!'
            return response_object, 201
        except exc.IntegrityError:
            db.session.rollblock()
            return response_object, 400

    def get(self):
        """Get all runs"""
        response_object = {
            'status': 'success',
            'data': {
                'runs': [run.to_json() for run in Run.query.all()]
            }
        }
        return response_object, 200


class Runs(Resource):
    def get(self, run_id):
        """Get single run details."""
        response_object = {
            'status': 'fail',
            'message': 'Run does not exist'
        }
        try:
            run = Run.query.filter_by(id=run_id).first()
            if not run:
                return response_object, 404
            else:
                response_object = {
                    'status': 'success',
                    'data': run.to_json()
                }
                return response_object, 200
        except ValueError:
            return response_object, 404
        except exc.DataError:
            return response_object, 404


api.add_resource(RunsPing, '/runs/ping')
api.add_resource(RunsList, '/runs')
api.add_resource(Runs, '/runs/<run_id>')
