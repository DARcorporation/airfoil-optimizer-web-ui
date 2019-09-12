from flask import Blueprint, request
from flask_restful import Resource, Api
from sqlalchemy import exc

from project import db
from project.api.models import Run

runs_blueprint = Blueprint("runs", __name__)
api = Api(runs_blueprint)


class RunsPing(Resource):
    def get(self):
        return {"status": "success", "message": "pong!"}


class RunsList(Resource):
    def post(self):
        post_data = request.get_json()
        response_object = {"status": "fail", "message": "Invalid payload."}
        if not post_data:
            return response_object, 400

        try:
            run = Run(**post_data)
        except TypeError:
            return response_object, 400

        try:
            db.session.add(run)
            db.session.commit()
            response_object["status"] = "success"
            response_object["message"] = "New run was added!"
            return response_object, 201
        except exc.IntegrityError:
            db.session.rollblock()
            return response_object, 400

    def get(self):
        """Get all runs"""
        response_object = {
            "status": "success",
            "data": {"runs": [run.to_json() for run in Run.query.all()]},
        }
        return response_object, 200


class Runs(Resource):
    def get(self, run_id):
        """Get single run details."""
        response_object = {"status": "fail", "message": "Run does not exist"}
        try:
            run = Run.query.filter_by(id=run_id).first()
            if not run:
                return response_object, 404
            else:
                response_object = {"status": "success", "data": run.to_json()}
                return response_object, 200
        except ValueError:
            return response_object, 404
        except exc.DataError:
            return response_object, 404


class RunsAccept(Resource):
    def get(self):
        """Return the run with the lowest id and status 0 and flip its status to 1."""
        run = Run.query.filter_by(status=0).first()
        if not run:
            response_object = {"status": "success", "message": "No runs in queue"}
            return response_object, 204
        else:
            run.status = 1
            response_object = {"status": "success", "data": run.to_json()}
            db.session.commit()
            return response_object, 200


class RunsComplete(Resource):
    def post(self):
        """Mark a run as complete."""
        post_data = request.get_json()
        response_object = {"status": "fail", "message": "Run does not exist."}
        if not post_data:
            return response_object, 400

        try:
            run = Run.query.filter_by(id=post_data["id"]).first()
            if not run:
                return response_object, 404
            elif run.status != 1:
                response_object[
                    "message"
                ] = "Cannot complete run which has not been accepted"
                return response_object, 400
            else:
                success = post_data["success"]
                run.status = 2 if success else 3
                response_object = {
                    "status": "success",
                    "message": f"Run marked as {'complete' if success else 'failed'}",
                }
                db.session.commit()
                return response_object, 200
        except exc.DataError:
            return response_object, 404
        except KeyError:
            response_object["message"] = "Invalid payload"
            return response_object, 400


api.add_resource(RunsPing, "/runs/ping")
api.add_resource(RunsList, "/runs")
api.add_resource(Runs, "/runs/<run_id>")
api.add_resource(RunsAccept, "/runs/accept")
api.add_resource(RunsComplete, "/runs/complete")
