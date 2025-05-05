import socket
from http import HTTPStatus

from flask import Blueprint

from environment.settings import CLIENT_IP
from environment.state_handling import set_rw_done

rw_bp = Blueprint("ransomware", __name__, url_prefix="/rw")


@rw_bp.route("/done", methods=["PUT"])
def mark_done():
    set_rw_done()
    return "", HTTPStatus.NO_CONTENT


def send_reset_corpus():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((CLIENT_IP, 42667))
        sock.send(bytes("reset", encoding="utf-8"))
        print("Sent reset to client.")


def send_terminate():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((CLIENT_IP, 42667))
        sock.send(bytes("terminate", encoding="utf-8"))
        print("Sent terminate to client.")
