import json
import socket
from http import HTTPStatus

from flask import Blueprint, request

from environment.state_handling import get_storage_path, is_multi_fp_collection, get_specific_config_folder_for_fp
from utilities.metrics import write_metrics_to_file

fp_bp = Blueprint("fingerprint", __name__, url_prefix="/fp")


@fp_bp.route("/<mac>", methods=["POST"])
def report_fingerprint(mac):
    body = json.loads(request.data)

    write_metrics_to_file(str(body["fp"]), get_specific_config_folder_for_fp(), is_multi_fp_collection())

    return "", HTTPStatus.CREATED


def terminate_fingerprint(ip_address_of_client_device):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((ip_address_of_client_device, 42667))
        sock.send(bytes("terminate", encoding="utf-8"))
