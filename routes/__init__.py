import csv
from flask import Blueprint, render_template, request, current_app, redirect, url_for
from io import StringIO

bp = Blueprint("routes", __name__)


@bp.route("/upload", methods=["GET"])
def upload_form():
    return render_template("upload_form.html")


@bp.route("/upload", methods=["POST"])
def upload():
    text_body = request.form.get("text_body", "")

    if "csv_file" not in request.files:
        return "No file part"

    csv_file = request.files["csv_file"]

    if csv_file:
        data = parse_csv(csv_file)
        current_app.logger.info(f"CSV Data: {data}")

        dummy_data = [
            {
                "title": "Article 1",
                "summary": "Summary 1",
                "keywords": ["keyword1", "keyword2"],
            },
            {
                "title": "Article 2",
                "summary": "Summary 2",
                "keywords": ["keyword3", "keyword4"],
            },
        ]

        return render_template("upload_form.html", text_body=text_body, data=dummy_data)

    return redirect(url_for("routes.upload_form"))


def parse_csv(csv_file):
    csv_data = []
    csv_stream = StringIO(csv_file.stream.read().decode("UTF-8"))
    csv_reader = csv.DictReader(csv_stream)
    for row in csv_reader:
        csv_data.append(row)
    return csv_data
