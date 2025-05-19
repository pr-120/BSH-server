import os
import socket
from http import HTTPStatus

from dotenv import load_dotenv
from flask import Blueprint
import subprocess

from environment.settings import CLIENT_IP
from environment.state_handling import set_rw_done


# loads environment
current_folder = os.path.dirname(os.path.abspath(__file__))
CONFIG_FOLDER = os.path.join(current_folder, "../../config")
load_dotenv(os.path.join(CONFIG_FOLDER, "folder_paths.config"))


bd_bp = Blueprint("backdoor", __name__, url_prefix="/bd")


@bd_bp.route("/done", methods=["PUT"])
def mark_done():
    set_rw_done()
    return "", HTTPStatus.NO_CONTENT


def send_reset_corpus():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((CLIENT_IP, 42667))
        sock.send(bytes("reset", encoding="utf-8"))
        print("Sent reset to client.")

@bd_bp.route("/terminate/<port>", methods=["PUT"])
def send_terminate(port: int):
    # get path of termination script
    script_folder = os.getenv("script_folder")
    termination_script = script_folder + "/terminate_screens.sh"

    # execute script
    subprocess.call([termination_script, str(port)])

    print("terminated malicious procedures.")
    return "", HTTPStatus.NO_CONTENT

