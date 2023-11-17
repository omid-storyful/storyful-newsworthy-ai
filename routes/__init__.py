from flask import Blueprint, render_template, request, current_app
from . import routes  # noqa: F401

bp = Blueprint("routes", __name__)


@bp.route("/upload", methods=["GET"])
def upload_form():
    return render_template("upload_form.html")


@bp.route("/upload", methods=["POST"])
def upload():
    form_data = request.form

    current_app.logger.info(f"Form Data: {form_data}")

    return "CSV parsing logic goes here"
