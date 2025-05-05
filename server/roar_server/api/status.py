from http import HTTPStatus

from flask import Blueprint

status_bp = Blueprint("status", __name__, url_prefix="/status")


@status_bp.route("", methods=["GET"])
def get_status():
    return "OK", HTTPStatus.OK
