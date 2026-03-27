from flask import Flask, render_template, request, redirect, url_for, session
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
from nutrition_db import nutrition_db

app = Flask(__name__)
app.secret_key = "nutrilens_secret_key"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLO Models
models = [
    YOLO("runs/detect/nutrilens_yolov8n_cpu/weights/best.pt"),
    YOLO("runs/detect/train2/weights/best.pt"),
    YOLO("runs/detect/nutrilens_yolov8n_cpu22/weights/best.pt")
]

# ---------------- HELPER FUNCTIONS ----------------

def calculate_daily_needs(weight, height, age, gender):
    if gender.lower() == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    return round(bmr * 1.2, 1)


def generate_recommendation(running_total, daily_needs, carbs, proteins, fat):
    if running_total < daily_needs * 0.8:
        rec = "You need more calories today. "
    elif running_total > daily_needs * 1.2:
        rec = "You are exceeding your daily calorie limit. "
    else:
        rec = "You're close to your daily calorie goal. "

    if carbs > proteins * 3:
        rec += "Reduce carb-heavy foods and add more proteins."
    elif proteins < 10:
        rec += "Add more protein-rich foods."
    elif fat < 5:
        rec += "Include healthy fats."
    else:
        rec += "Your nutrient balance looks good!"

    return rec


def get_nutrition_info(label):
    return nutrition_db.get(label, {
        'calories': 0,
        'carbs': 0,
        'protein': 0,
        'fat': 0,
        'vitamins': []
    })


def analyze_freshness(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return "Unknown", 0.0

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:, :, 2])
        freshness_score = round((brightness / 255) * 100, 2)

        if freshness_score > 60:
            label = "Fresh"
        elif freshness_score > 40:
            label = "Moderate"
        else:
            label = "Not Fresh"

        return label, freshness_score
    except:
        return "Unknown", 0.0


def ensemble_predict(image_path, conf_threshold=0.25):
    label_conf = defaultdict(list)

    for model in models:
        results = model.predict(image_path, imgsz=416, conf=conf_threshold, save=False)
        preds = results[0]

        if hasattr(preds, "boxes") and preds.boxes is not None:
            for box in preds.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]
                label_conf[label].append(conf)

    if label_conf:
        averaged = {k: sum(v)/len(v) for k, v in label_conf.items()}
        best_label = max(averaged, key=averaged.get)
        best_conf = averaged[best_label]
        return best_label, round(best_conf, 4)
    else:
        return "Unknown", 0.0


# ---------------- ROUTES ----------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        session['height'] = float(request.form['height'])
        session['weight'] = float(request.form['weight'])
        session['age'] = int(request.form['age'])
        session['gender'] = request.form['gender']
        session['running_total_calories'] = 0

        return redirect(url_for('dashboard'))

    return render_template("login.html")


# ⭐ NEW DASHBOARD ROUTE
@app.route("/dashboard")
def dashboard():
    if 'height' not in session:
        return render_template("dashboard.html", new_user=True)

    return render_template(
        "dashboard.html",
        new_user=False,
        height=session.get('height'),
        weight=session.get('weight'),
        age=session.get('age'),
        gender=session.get('gender'),
        calories=session.get('running_total_calories', 0)
    )


@app.route("/upload", methods=["GET", "POST"])
def upload_food():

    if request.method == "POST":

        file = request.files['food_image']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        predicted_label, confidence = ensemble_predict(filepath)
        info = get_nutrition_info(predicted_label)

        calories = info.get('calories', 0)
        carbs = info.get('carbs', 0)
        proteins = info.get('protein', 0)
        fat = info.get('fat', 0)
        vitamins = ", ".join(info.get('vitamins', []))

        freshness_label, freshness_score = analyze_freshness(filepath)

        session['running_total_calories'] += calories
        running_total = session['running_total_calories']

        daily_needs = calculate_daily_needs(
            session['weight'],
            session['height'],
            session['age'],
            session['gender']
        )

        recommendation = generate_recommendation(
            running_total, daily_needs, carbs, proteins, fat
        )

        return render_template(
            "nutrition.html",
            uploaded_image_url=filepath,
            predicted_label=predicted_label,
            calories=calories,
            carbs=carbs,
            proteins=proteins,
            fat=fat,
            vitamins=vitamins,
            running_total_calories=running_total,
            daily_calorie_needs=daily_needs,
            recommendation_text=recommendation,
            freshness=freshness_label,
            freshness_score=freshness_score,
            confidence=round(confidence * 100, 2)
        )

    return render_template("nutrition.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=True)