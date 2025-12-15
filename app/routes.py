from flask import Blueprint, render_template, request
from app.services import run_optimization

main = Blueprint("main", __name__)

@main.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@main.route("/optimize", methods=["POST"])
def optimize():
    algorithm = request.form.get("algorithm")
    result = run_optimization(algorithm)

    return render_template(
        "index.html",
        result=result,
        selected_algo=algorithm
    )
