import os
import glob
import time
import numpy as np
import cv2
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image, ImageChops, ImageEnhance

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model("forensic_ai_model.h5")

# ELA quality LOCKED to 75 — matches training pipeline
ELA_QUALITY = 75

def prepare_ela_image(path, quality=None):
    if quality is None:
        quality = ELA_QUALITY

    original = Image.open(path).convert("RGB")
    
    w, h = original.size
    crop_size = min(w, int(h * 0.7)) 
    left = 0
    top = h - crop_size
    right = w
    bottom = h
    working = original.crop((left, top, right, bottom))

    temp = "temp_resave.jpg"
    working.save(temp, "JPEG", quality=quality)
    resaved = Image.open(temp)

    ela_raw = ImageChops.difference(working, resaved)

    extrema = ela_raw.getextrema()
    max_diff = max([e[1] for e in extrema]) or 1
    ela = ImageEnhance.Brightness(ela_raw).enhance(255.0 / max_diff)

    ela.save(os.path.join(UPLOAD_FOLDER, "latest_ela.png"))

    gray = cv2.cvtColor(np.array(ela), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img = cv2.cvtColor(np.array(working), cv2.COLOR_RGB2BGR)
    cv_flag = False

    for c in contours:
        if cv2.contourArea(c) > 50:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
            cv_flag = True

    cv2.imwrite(os.path.join(UPLOAD_FOLDER, "latest_highlighted.png"), img)

    return ela.resize((128,128), Image.LANCZOS), cv_flag

@app.route("/", methods=["GET","POST"])
def index():

    for f in glob.glob(os.path.join(UPLOAD_FOLDER,"*")):
        try:
            os.remove(f)
        except:
            pass

    if request.method == "POST":

        file = request.files.get("file")

        if file and file.filename:

            filepath = os.path.join(UPLOAD_FOLDER,file.filename)
            file.save(filepath)

            ela_img, cv_flag = prepare_ela_image(filepath)

            # CRITICAL: Use MobileNetV2 preprocessing [-1, 1] not [0, 1]
            ela = np.array(ela_img, dtype=np.float32)
            ela = ela.reshape(1, 128, 128, 3)
            ela = preprocess_input(ela)

            prediction = float(model.predict(ela, verbose=0)[0][0])

            # =====================================================
            # THREE-TIER DECISION SYSTEM
            # Score > 0.70  → TAMPERED (high confidence)
            # Score 0.40-0.70 → SUSPICIOUS (manual review needed)
            # Score < 0.40  → AUTHENTIC (high confidence)
            # =====================================================
            TAMPERED_THRESHOLD = 0.70
            SUSPICIOUS_THRESHOLD = 0.40

            analysis = []
            analysis.append(f"🧠 CNN Score: {prediction:.3f}")

            if prediction > TAMPERED_THRESHOLD:
                result = "TAMPERED (FAKE)"
                confidence = prediction * 100
                analysis.append("⚠️ Strong forensic indicators of manipulation detected.")
                if cv_flag:
                    analysis.append("🔍 Suspicious edited regions highlighted in red.")

            elif prediction > SUSPICIOUS_THRESHOLD:
                result = "SUSPICIOUS (NEEDS MANUAL REVIEW)"
                confidence = prediction * 100
                analysis.append("⚠️ Some forensic anomalies detected but not conclusive.")
                analysis.append("📋 Recommend manual verification of this receipt.")
                if cv_flag:
                    analysis.append("🔍 Potential edited regions highlighted in red.")

            else:
                result = "AUTHENTIC (REAL)"
                confidence = (1 - prediction) * 100
                analysis.append("✅ No strong forensic evidence of tampering.")

            ela_raw = np.array(ela_img, dtype=np.float32) / 255.0
            ela_variance = float(np.var(ela_raw) * 1000)
            ela_peak = float(np.max(ela_raw) * 100)
            analysis.append(f"📊 ELA Stats — Variance: {ela_variance:.1f} | Peak: {ela_peak:.1f}")

            return render_template(
                "index.html",
                result=result,
                confidence=f"{confidence:.2f}%",
                filename=file.filename,
                ela_path="latest_ela.png",
                variance_score=f"Var: {ela_variance:.1f} | Peak: {ela_peak:.1f}",
                analysis_insights=analysis,
                timestamp=int(time.time())
            )

    return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True)