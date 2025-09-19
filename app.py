from flask import Flask, render_template, request, redirect, url_for, session
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from nutrition_db import nutrition_db  # your nutrition dictionary

# Flask setup
app = Flask(__name__)
app.secret_key = "nutrilens_secret_key"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load all 3 YOLOv8 models (your saved weights from different runs)
models = [
    YOLO("runs/detect/nutrilens_yolov8n_cpu/weights/best.pt"),    # 10 epochs
    YOLO("runs/detect/train2/weights/best.pt"),   # 21 epochs
    YOLO("runs/detect/nutrilens_yolov8n_cpu22/weights/best.pt")   # 30 epochs
]

# ---------- Routes ----------

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
        session['food_log'] = []
        return redirect(url_for('upload_food'))
    return render_template("login.html")

@app.route("/upload", methods=["GET", "POST"])
def upload_food():
    if request.method == "POST":
        if 'food_image' not in request.files:
            return "No file part"
        file = request.files['food_image']
        if file.filename == '':
            return "No selected file"
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Run predictions with all models
        predictions = []
        for model in models:
            results = model.predict(filepath, imgsz=416, conf=0.5)
            preds = results[0]
            if preds.boxes is not None and len(preds.boxes) > 0:
                best_box = max(preds.boxes, key=lambda b: b.conf[0])
                class_id = int(best_box.cls[0])
                predicted_label = model.names[class_id]
                confidence = float(best_box.conf[0])
                predictions.append((predicted_label, confidence))

        # Pick label with highest confidence across all models
        if predictions:
            predicted_label, _ = max(predictions, key=lambda x: x[1])
        else:
            predicted_label = "Unknown"

        # ---------------- Nutrition calculation ----------------
        info = nutrition_db.get(predicted_label, {
            'calories': 0,
            'carbs': 0,
            'protein': 0,
            'fat': 0,
            'vitamins': []
        })
        calories = info['calories']
        carbs = info['carbs']
        proteins = info['protein']
        fat = info['fat']
        vitamins = ", ".join(info['vitamins'])

        # Running total calories
        if 'running_total_calories' not in session:
            session['running_total_calories'] = 0
        session['running_total_calories'] += calories
        running_total_calories = session['running_total_calories']

        # Daily needs estimate (basic)
        daily_calorie_needs = 2000  
        recommendation_text = "Keep balanced meals with carbs, proteins, and fats."
        freshness = "Fresh"

        # Save food log
        session['food_log'].append({
            'image': filepath,
            'food': predicted_label,
            'calories': calories,
            'carbs': carbs,
            'proteins': proteins,
            'fat': fat,
            'vitamins': vitamins
        })

        # Render nutrition.html
        return render_template("nutrition.html",
                               uploaded_image_url=filepath,
                               predicted_label=predicted_label,
                               calories=calories,
                               carbs=carbs,
                               proteins=proteins,
                               fat=fat,
                               vitamins=vitamins,
                               running_total_calories=running_total_calories,
                               daily_calorie_needs=daily_calorie_needs,
                               recommendation_text=recommendation_text,
                               freshness=freshness)

    # GET → upload page
    return render_template("nutrition.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('index'))

# Run Flask
if __name__ == "__main__":
    app.run(debug=True)
