"""
app.py — CKD Score Flask Application
Uses the modular ocr/ package for all text extraction from lab reports.
"""

import os
import logging
import joblib
import pandas as pd
from flask import Flask, render_template, request

# ── OCR package ──────────────────────────────────────────────────────────────
from ocr import configure_tesseract, ocr_text, extract_cbc, extract_cbc_consensus, load_image, ocr_file
from ocr.pdf_extractor import extract_pdf_text
from ocr.ocr_engine import OCREngine as _OCREngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Tesseract (auto-detects path; override here if needed)
configure_tesseract()

app     = Flask(__name__)
model   = joblib.load("ckd_model.pkl")
imputer = joblib.load("ckd_imputer.pkl")
_engine = _OCREngine()          # shared OCR engine (loaded once)


# ─────────────────────────────────────────────────────────────────────────────
# OCR HELPER
# ─────────────────────────────────────────────────────────────────────────────

def ocr_image(path: str) -> str:
    """
    Hands off the image processing to the newly refactored dynamic OCREngine.
    Automatically handles multiple preprocessing, dynamic configs, filtering, 
    post-processing, and fallback to EasyOCR.
    """
    try:
        return ocr_file(path, output="text")
    except Exception as e:
        logger.error(f"[OCR] Engine completely failed: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# KIDNEY HEALTH SCORE & eGFR
# ─────────────────────────────────────────────────────────────────────────────

def calculate_egfr(creatinine: float, age: float) -> float:
    """Calculate estimated Glomerular Filtration Rate using simplified MDRD."""
    if creatinine <= 0 or age <= 0:
        return -1.0
    # Simplified MDRD formula (assumes generic male since gender isn't specified)
    egfr = 175 * (creatinine ** -1.154) * (age ** -0.203)
    return round(egfr, 1)

def get_ckd_stage(egfr_value: float) -> dict:
    """
    Categorize the stage of Chronic Kidney Disease (CKD) based on calculated eGFR.
    Uses KDIGO guidelines.
    """
    if egfr_value is None or egfr_value <= 0:
        return {"stage": "Invalid Value", "desc": "N/A"}
    if egfr_value >= 90:
        return {"stage": "Stage 1", "desc": "Kidney damage with normal function"}
    elif 60 <= egfr_value < 90:
        return {"stage": "Stage 2", "desc": "Mild loss of kidney function"}
    elif 45 <= egfr_value < 60:
        return {"stage": "Stage 3a", "desc": "Mild to moderate loss"}
    elif 30 <= egfr_value < 45:
        return {"stage": "Stage 3b", "desc": "Moderate to severe loss"}
    elif 15 <= egfr_value < 30:
        return {"stage": "Stage 4", "desc": "Severe loss of kidney function"}
    else:
        return {"stage": "Stage 5", "desc": "Kidney failure (End-stage)"}

def kidney_health_score(hemo, rc, wbc, pcv, creatinine, urea, bp, ckd_prob,
                         bgr=120, sod=140, pot=4.5, egfr=None):
    """
    Compute a 0–100 kidney health score from biomarkers + eGFR.

    Scoring philosophy:
    - Start at 100 (perfect health)
    - Deduct for EACH abnormal marker proportional to clinical severity
    - eGFR / CKD stage is the HEAVIEST factor (up to 35 pts)
    - CKD model probability adds up to 30 pts
    - Individual biomarkers each penalize 3–25 pts
    - A Stage 3a patient should score ~50–65 (Moderate / Poor)
    """
    penalties = {}

    # ── 1. eGFR / CKD Stage (dominant factor, up to 35 pts) ───────────────
    if egfr is not None and egfr > 0:
        if   egfr >= 90:  penalties["Kidney Function (eGFR)"] = 0
        elif egfr >= 75:  penalties["Kidney Function (eGFR)"] = 5    # Stage 2 - mild
        elif egfr >= 60:  penalties["Kidney Function (eGFR)"] = 12   # Stage 2 - lower end
        elif egfr >= 45:  penalties["Kidney Function (eGFR)"] = 22   # Stage 3a
        elif egfr >= 30:  penalties["Kidney Function (eGFR)"] = 28   # Stage 3b
        elif egfr >= 15:  penalties["Kidney Function (eGFR)"] = 32   # Stage 4
        else:             penalties["Kidney Function (eGFR)"] = 35   # Stage 5
    else:
        penalties["Kidney Function (eGFR)"] = 0

    # ── 2. CKD Model Probability (up to 30 pts) ──────────────────────────
    # Exponential curve: low probabilities barely penalize, high ones hit hard
    if ckd_prob > 0.5:
        penalties["CKD Risk (ML Model)"] = round(10 + (ckd_prob - 0.5) * 40, 1)
    elif ckd_prob > 0.2:
        penalties["CKD Risk (ML Model)"] = round(ckd_prob * 15, 1)
    else:
        penalties["CKD Risk (ML Model)"] = 0

    # ── 3. Serum Creatinine (up to 25 pts) ────────────────────────────────
    if   creatinine > 8:   penalties["Creatinine"] = 25
    elif creatinine > 4:   penalties["Creatinine"] = 20
    elif creatinine > 2.5: penalties["Creatinine"] = 16
    elif creatinine > 1.8: penalties["Creatinine"] = 12
    elif creatinine > 1.3: penalties["Creatinine"] = 8
    elif creatinine > 1.2: penalties["Creatinine"] = 4
    else:                  penalties["Creatinine"] = 0

    # ── 4. Hemoglobin (up to 20 pts) ──────────────────────────────────────
    if   hemo < 7:    penalties["Hemoglobin"] = 20
    elif hemo < 9:    penalties["Hemoglobin"] = 15
    elif hemo < 10.5: penalties["Hemoglobin"] = 10
    elif hemo < 12:   penalties["Hemoglobin"] = 6
    elif hemo > 18:   penalties["Hemoglobin"] = 8
    elif hemo > 17.5: penalties["Hemoglobin"] = 4
    else:             penalties["Hemoglobin"] = 0

    # ── 5. Blood Urea (up to 18 pts) ─────────────────────────────────────
    if   urea > 150: penalties["Blood Urea"] = 18
    elif urea > 100: penalties["Blood Urea"] = 14
    elif urea > 70:  penalties["Blood Urea"] = 10
    elif urea > 45:  penalties["Blood Urea"] = 6
    else:            penalties["Blood Urea"] = 0

    # ── 6. Blood Pressure (up to 12 pts) ──────────────────────────────────
    if   bp > 110: penalties["Blood Pressure"] = 12
    elif bp > 95:  penalties["Blood Pressure"] = 9
    elif bp > 85:  penalties["Blood Pressure"] = 6
    elif bp > 80:  penalties["Blood Pressure"] = 3
    else:          penalties["Blood Pressure"] = 0

    # ── 7. PCV / Haematocrit (up to 12 pts) ───────────────────────────────
    if   pcv < 20: penalties["PCV"] = 12
    elif pcv < 28: penalties["PCV"] = 8
    elif pcv < 36: penalties["PCV"] = 5
    elif pcv > 54: penalties["PCV"] = 5
    else:          penalties["PCV"] = 0

    # ── 8. Blood Glucose (up to 10 pts) ───────────────────────────────────
    if   bgr > 300: penalties["Blood Glucose"] = 10
    elif bgr > 200: penalties["Blood Glucose"] = 7
    elif bgr > 140: penalties["Blood Glucose"] = 4
    elif bgr < 60:  penalties["Blood Glucose"] = 5
    else:           penalties["Blood Glucose"] = 0

    # ── 9. Sodium (up to 8 pts) ───────────────────────────────────────────
    if   sod < 125: penalties["Sodium"] = 8
    elif sod < 130: penalties["Sodium"] = 5
    elif sod < 136: penalties["Sodium"] = 3
    elif sod > 150: penalties["Sodium"] = 6
    elif sod > 145: penalties["Sodium"] = 3
    else:           penalties["Sodium"] = 0

    # ── 10. Potassium (up to 10 pts) ──────────────────────────────────────
    if   pot > 6.5: penalties["Potassium"] = 10
    elif pot > 5.5: penalties["Potassium"] = 7
    elif pot > 5.0: penalties["Potassium"] = 4
    elif pot < 3.0: penalties["Potassium"] = 8
    elif pot < 3.5: penalties["Potassium"] = 4
    else:           penalties["Potassium"] = 0

    # ── 11. RBC Count (up to 6 pts) ───────────────────────────────────────
    if   rc < 3.0: penalties["RBC Count"] = 6
    elif rc < 3.8: penalties["RBC Count"] = 3
    elif rc > 6.5: penalties["RBC Count"] = 4
    elif rc > 6.0: penalties["RBC Count"] = 2
    else:          penalties["RBC Count"] = 0

    # ── 12. WBC Count (up to 6 pts) ───────────────────────────────────────
    if   wbc > 15000: penalties["WBC Count"] = 6
    elif wbc > 11000: penalties["WBC Count"] = 3
    elif wbc < 3000:  penalties["WBC Count"] = 5
    elif wbc < 4000:  penalties["WBC Count"] = 2
    else:             penalties["WBC Count"] = 0

    # ── Clamp & Grade ─────────────────────────────────────────────────────
    score = max(0, min(100, round(100 - sum(penalties.values()))))

    if score >= 85:   grade, status = "A", "Excellent"
    elif score >= 70: grade, status = "B", "Good"
    elif score >= 55: grade, status = "C", "Moderate"
    elif score >= 35: grade, status = "D", "Poor"
    else:             grade, status = "F", "Critical"

    return score, grade, status, penalties


# ─────────────────────────────────────────────────────────────────────────────
# MARKER STATUS
# ─────────────────────────────────────────────────────────────────────────────

def marker_status(name: str, value: float):
    """Return (status_label, pct_in_range) for a biomarker."""
    if name == "egfr":
        if value >= 90: return "Normal", 100
        elif value >= 60: return "Mildly Low", 80
        elif value >= 45: return "Low", 60
        elif value >= 30: return "Very Low", 40
        elif value >= 15: return "Severe", 20
        else:             return "Critical", 5

    ranges = {
        "hemoglobin": (12.0, 17.5),
        "creatinine": (0.6,  1.2),
        "urea":       (7.0,  45.0),
        "bp":         (60.0, 80.0),
        "pcv":        (36.0, 50.0),
        "wbc":        (4000, 11000),
        "rbc":        (3.8,  6.0),
        "sodium":     (136.0, 145.0),
        "potassium":  (3.5,  5.0),
        "glucose":    (70.0, 140.0),
    }
    lo, hi = ranges.get(name, (0, 1))
    span = hi - lo
    if value < lo:
        return "Low",    max(0, int(100 * value / lo))
    elif value > hi:
        excess = value - hi
        return "High",   max(20, min(100, int(100 - (excess / span) * 50)))
    else:
        return "Normal", 100


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/debug-ocr", methods=["POST"])
def debug_ocr():
    """POST an image/PDF to see raw OCR text and extracted CBC values."""
    report = request.files.get("report")
    if not report or not report.filename:
        return "No file uploaded", 400
    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", report.filename)
    report.save(filepath)
    text = extract_pdf_text(filepath, ocr_engine=_engine) if filepath.lower().endswith(".pdf") else ocr_image(filepath)
    extracted = extract_cbc(text)
    return f"<pre>RAW OCR:\n{text}\n\nEXTRACTED:\n{extracted}</pre>"


def _run_ocr_from_request(report_file):
    """Save uploaded file, run OCR, return (cbc_dict, pdf_warning)."""
    pdf_warning = None
    text = ""
    if not report_file or not report_file.filename:
        return {}, pdf_warning

    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", report_file.filename)
    report_file.save(filepath)

    if filepath.lower().endswith(".pdf"):
        try:
            import fitz as _fitz
            with open(filepath, "rb") as fh:
                _raw = fh.read()
            _doc = _fitz.open(stream=_raw, filetype="pdf")
            locked = _doc.needs_pass
            _doc.close()
            if locked:
                pdf_warning = ("Your PDF is password-protected. Please remove the password "
                               "or upload a PNG/JPG screenshot instead.")
                logger.warning("[PDF] Skipping password-protected PDF.")
            else:
                text = extract_pdf_text(filepath, ocr_engine=_engine)
        except Exception as e:
            logger.error(f"[OCR] PDF pre-check error: {e}")
            text = extract_pdf_text(filepath, ocr_engine=_engine)
        cbc = extract_cbc(text) if text else {}
    elif filepath.lower().endswith((".png", ".jpg", ".jpeg")):
        # ── Multi-pass consensus extraction for images ────────────────────
        # Run OCR on ALL preprocessing candidates and take a majority vote
        # per marker.  This fixes single-candidate digit misreads (e.g.
        # Tesseract reading '5' as '9' on one pipeline but not the others).
        try:
            img = load_image(filepath)
            all_texts = _engine.all_candidates_text(img)
            logger.info(f"[OCR] Ran {len(all_texts)} preprocessing candidates for consensus")
            cbc = extract_cbc_consensus(all_texts) if all_texts else {}
            # Also keep the single-best text for any fallback
            text = all_texts[0] if all_texts else ""
        except Exception as e:
            logger.error(f"[OCR] Consensus extraction failed on {filepath}: {e}")
            # Fallback to single-pass
            try:
                text = ocr_image(filepath)
                cbc = extract_cbc(text) if text else {}
            except Exception as e2:
                logger.error(f"[OCR] Single-pass fallback also failed: {e2}")
                cbc = {}
    else:
        cbc = {}

    if cbc:
        logger.info(f"[OCR] Extracted (consensus): {cbc}")
    return cbc, pdf_warning


# ── STEP 1: Upload → OCR → Editable review form ───────────────────────────────
@app.route("/review", methods=["POST"])
def review():
    """
    Run OCR on the uploaded report, then show all extracted values
    in editable fields so the user can correct any OCR mistakes before
    the model runs.
    """
    age = request.form.get("age", "")
    bp  = request.form.get("bp",  "")

    cbc, pdf_warning = _run_ocr_from_request(request.files.get("report"))

    # Each field: form key, display label, unit, CBC dict key, fallback default
    FIELDS = [
        ("hemo",           "Hemoglobin",    "g/dL",  "hemoglobin", ""),
        ("rbc_val",        "RBC Count",     "M/µL",  "rbc",        ""),
        ("wbc_val",        "WBC Count",     "/µL",   "wbc",        ""),
        ("pcv_val",        "PCV",           "%",     "pcv",        ""),
        ("creatinine_val", "Creatinine",    "mg/dL", "creatinine", ""),
        ("urea_val",       "Blood Urea",    "mg/dL", "urea",       ""),
        ("bgr_val",        "Blood Glucose", "mg/dL", "glucose",    ""),
        ("sod_val",        "Sodium",        "mEq/L", "sodium",     ""),
        ("pot_val",        "Potassium",     "mEq/L", "potassium",  ""),
    ]

    fields = []
    for key, label, unit, cbc_key, default in FIELDS:
        ocr_val = cbc.get(cbc_key)
        fields.append({
            "key":       key,
            "label":     label,
            "unit":      unit,
            "value":     str(round(ocr_val, 3)) if ocr_val is not None else default,
            "ocr_found": ocr_val is not None,
        })

    return render_template(
        "review.html",
        fields=fields,
        age=age,
        bp=bp,
        pdf_warning=pdf_warning,
    )


# ── STEP 2: Confirmed values → Predict ───────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    # All values come from the review form — user has already verified them
    age = float(request.form.get("age") or 0)
    bp  = float(request.form.get("bp")  or 0)

    def _f(key, default):
        v = request.form.get(key, "").strip()
        try:    return float(v) if v else default
        except: return default

    hemo       = _f("hemo",           13.5)
    rc         = _f("rbc_val",         5.0)
    wbc        = _f("wbc_val",        7800.0)
    pcv        = _f("pcv_val",          44.0)
    creatinine = _f("creatinine_val",    1.0)
    urea       = _f("urea_val",         40.0)
    bgr        = _f("bgr_val",         120.0)
    sod        = _f("sod_val",         140.0)
    pot        = _f("pot_val",           4.5)

    logger.info(f"[PREDICT] Received form values: hemo={hemo}, rbc={rc}, wbc={wbc}, "
                f"pcv={pcv}, creatinine={creatinine}, urea={urea}, bgr={bgr}, sod={sod}, pot={pot}")

    # Determine source label: "report" if a value was provided, otherwise "assumed <default>"
    def _src(key, default):
        v = request.form.get(key, "").strip()
        try:
            float(v)
            return "report"
        except (ValueError, TypeError):
            return f"assumed {default}"

    hemo_src = _src("hemo", 13.5)
    rc_src   = _src("rbc_val", 5.0)
    wbc_src  = _src("wbc_val", 7800)
    pcv_src  = _src("pcv_val", 44)
    sc_src   = _src("creatinine_val", 1.0)
    urea_src = _src("urea_val", 40)
    bgr_src  = _src("bgr_val", 120)
    sod_src  = _src("sod_val", 140)
    pot_src  = _src("pot_val", 4.5)

    rbc_flag = 1 if 3.8 <= rc <= 6.0 else 0

    # ── Model prediction ──────────────────────────────────────────────────────
    patient = pd.DataFrame([[
        age, bp, 1.02, 0, 0, rbc_flag, 1, 0, 0,
        bgr, urea, creatinine, sod, pot,
        hemo, pcv, wbc, rc,
        0, 0, 0, 1, 0, 0
    ]], columns=[
        "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
        "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc",
        "htn", "dm", "cad", "appet", "pe", "ane",
    ])

    patient_imp  = pd.DataFrame(imputer.transform(patient), columns=patient.columns)
    prediction   = model.predict(patient_imp)[0]
    proba        = model.predict_proba(patient_imp)[0]
    ckd_detected = (prediction == 1)
    ckd_prob     = float(proba[1])
    confidence   = int(round(max(proba) * 100))

    # ── Kidney health score & eGFR ────────────────────────────────────────────
    egfr = calculate_egfr(creatinine, age)
    ckd_stage_info = get_ckd_stage(egfr)
    score, grade, status, penalties = kidney_health_score(
        hemo, rc, wbc, pcv, creatinine, urea, bp, ckd_prob,
        bgr=bgr, sod=sod, pot=pot, egfr=egfr
    )

    # ── Biomarker rows ────────────────────────────────────────────────────────
    def mk(name, value, unit, normal, key):
        st, pct = marker_status(key, value)
        return {"name": name, "value": value, "unit": unit,
                "normal": normal, "status": st, "pct": pct}

    markers = [
        {**mk("eGFR",             egfr,                 "mL/min", "> 90",      "egfr"),        "source": "calculated"},
        {**mk("Hemoglobin",       round(hemo, 1),       "g/dL",   "12.0 – 17.5", "hemoglobin"),  "source": hemo_src},
        {**mk("RBC Count",        round(rc, 2),          "M/µL",   "3.8 – 6.0",   "rbc"),         "source": rc_src},
        {**mk("WBC Count",        int(wbc),              "/µL",    "4,000 – 11,000","wbc"),        "source": wbc_src},
        {**mk("PCV / Haematocrit",round(pcv, 1),         "%",      "36 – 50",     "pcv"),         "source": pcv_src},
        {**mk("Serum Creatinine", round(creatinine, 1),  "mg/dL",  "0.6 – 1.2",   "creatinine"),  "source": sc_src},
        {**mk("Blood Urea",       round(urea, 1),        "mg/dL",  "7 – 45",      "urea"),        "source": urea_src},
        {**mk("Blood Pressure",   int(bp),               "mmHg",   "60 – 80",     "bp"),          "source": "entered"},
        {**mk("Blood Glucose",    round(bgr, 1),         "mg/dL",  "70 – 140",    "glucose"),     "source": bgr_src},
        {**mk("Sodium",           round(sod, 1),         "mEq/L",  "136 – 145",   "sodium"),      "source": sod_src},
        {**mk("Potassium",        round(pot, 1),         "mEq/L",  "3.5 – 5.0",   "potassium"),   "source": pot_src},
    ]

    # ── Recommendations ───────────────────────────────────────────────────────
    tips = []
    if ckd_stage_info['stage'] in ['Stage 3b', 'Stage 4', 'Stage 5']:
        tips.append(f"Severe eGFR reading ({ckd_stage_info['stage']}). Immediate nephrology consultation required.")
    elif ckd_stage_info['stage'] == 'Stage 3a':
        tips.append(f"eGFR indicates mild-to-moderate kidney impairment ({ckd_stage_info['stage']}). Schedule a nephrology consultation.")
    elif ckd_stage_info['stage'] == 'Stage 2':
        tips.append("eGFR shows mild kidney function decline. Monitor regularly and manage risk factors.")
    if hemo < 12:       tips.append("Low hemoglobin — consider iron-rich foods and consult a physician for anaemia evaluation.")
    if creatinine > 1.3:tips.append("Elevated creatinine — reduce protein intake and schedule a kidney function panel.")
    if urea > 45:       tips.append("High blood urea — increase water intake and reduce high-protein foods.")
    if bp > 80:         tips.append("Elevated blood pressure — reduce sodium intake, exercise regularly, and monitor BP daily.")
    if pcv < 36:        tips.append("Low PCV indicates possible anaemia — seek haematology consultation.")
    if bgr > 140:       tips.append("High blood glucose — monitor for diabetes and consult an endocrinologist.")
    if sod < 136:       tips.append("Low sodium (hyponatremia) — may indicate fluid imbalance; consult a physician.")
    elif sod > 145:     tips.append("High sodium (hypernatremia) — increase water intake and reduce salt consumption.")
    if pot < 3.5:       tips.append("Low potassium (hypokalemia) — eat potassium-rich foods and consult a physician.")
    elif pot > 5.0:     tips.append("High potassium (hyperkalemia) — avoid high-potassium foods; this can affect heart rhythm.")
    if ckd_detected:    tips.append("CKD risk flagged — consult a nephrologist for a comprehensive kidney workup.")
    if not tips:        tips.append("All markers are within normal limits. Maintain a healthy diet, stay hydrated, and schedule annual check-ups.")

    return render_template(
        "index.html",
        ckd_detected=ckd_detected, confidence=confidence,
        ckd_prob=round(ckd_prob * 100, 1),
        score=score, grade=grade, status=status, penalties=penalties,
        age=int(age), bp=int(bp),
        markers=markers, tips=tips, score_val=score,
        egfr=egfr, ckd_stage_info=ckd_stage_info,
    )


if __name__ == "__main__":
    app.run(debug=False)