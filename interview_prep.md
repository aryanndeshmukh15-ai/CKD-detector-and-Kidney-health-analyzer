# KidneyIQ — Interview Prep: Questions & Answers

---

## Q1. Walk me through the end-to-end flow of your application.

**Answer:**

> The application has a two-step flow for report uploads and a single-step flow for manual entry.
>
> **Step 1 — Upload & OCR Extraction:**
> The user lands on the home page and chooses either "Upload Report" or "Enter Manually." If they upload a lab report (PNG, JPG, or PDF), the image goes to the `/review` route. There, I save the file to disk and pass it through my OCR pipeline.
>
> For **images**, I use a multi-pass consensus approach — the image goes through **6 different preprocessing pipelines** (raw grayscale, sharpened grayscale, printed, low-light, blurry, low-contrast) combined with **3 Tesseract PSM modes** each, giving up to 18 OCR passes. I run `extract_cbc()` on each output independently and then take a **majority vote per marker** — so if 14 out of 18 passes read RBC as "5.2" and 4 read it as "9.2", the correct "5.2" wins. This handles Tesseract digit-level misreads robustly.
>
> For **PDFs**, I first check if it's password-protected using PyMuPDF. If not, I extract text using a combination of native PDF text extraction and OCR fallback for scanned pages.
>
> The extracted values are then shown on an **editable review page** where the user can verify and correct any OCR mistakes. Fields extracted by OCR are highlighted in green with a "✓ OCR" badge; missing ones show "not found."
>
> **Step 2 — Prediction & Analysis:**
> When the user clicks "Confirm & Analyse," the confirmed values go to the `/predict` route. There, I construct a feature vector with 24 features (age, BP, specific gravity, albumin, sugar, RBC flag, and all the lab values), run it through a `SimpleImputer` (median strategy) to handle any missing values, and feed it into my **calibrated soft-voting ensemble** model.
>
> The model outputs a CKD probability. I then calculate the **eGFR** using the simplified MDRD formula, determine the **CKD stage** (KDIGO guidelines), and compute a **0–100 Kidney Health Score** using a weighted penalty system across 12 factors. Finally, the results dashboard shows the health score gauge, CKD verdict, biomarker analysis table, and personalized recommendations.

---

## Q2. What ML algorithm did you use and why?

**Answer:**

> I use a **soft-voting ensemble** of three models:
> 1. **Random Forest** (weight 4) — Tuned via GridSearchCV over `n_estimators`, `max_depth`, and `min_samples_split`. It's the strongest individual model and handles non-linear relationships well.
> 2. **Gradient Boosting** (weight 3) — Complements RF by focusing on sequential error correction. I used 300 estimators with a low learning rate (0.05) for better generalization.
> 3. **Logistic Regression** (weight 1) — Wrapped in a `StandardScaler` pipeline. Acts as a linear baseline that prevents the ensemble from overfitting on noise.
>
> The key design decision was **removing SVM** from the ensemble. SVM's probability estimates (via Platt scaling) are notoriously poorly calibrated, which was hurting the confidence scores.
>
> I also applied **`CalibratedClassifierCV` with isotonic regression** to both RF and GB. This is critical — raw tree-based models tend to be overconfident. Isotonic calibration gives us probability estimates that actually reflect real-world likelihood, so when the model says "92% CKD risk," it genuinely means ~92% of patients with similar markers have CKD.

---

## Q3. How did you handle class imbalance?

**Answer:**

> The dataset (UCI Kidney Disease dataset, 400 records) has a moderate class imbalance — more CKD cases than non-CKD. I applied **SMOTE** (Synthetic Minority Oversampling Technique) on the training set after the train-test split to balance the classes. SMOTE generates synthetic samples by interpolating between existing minority-class samples in feature space, rather than simple duplication.
>
> I also used `class_weight="balanced"` in the Random Forest, which adjusts sample weights inversely proportional to class frequencies. This gives double protection against bias.
>
> Importantly, I applied SMOTE **only on the training set** — the test set remains untouched to get honest evaluation metrics.

---

## Q4. What's your model's performance?

**Answer:**

> On the held-out test set (20%, stratified):
> - **Accuracy**: ~97-98%
> - **ROC-AUC**: ~0.99
> - I also track **sensitivity** (true positive rate) and **specificity** (true negative rate) separately because in a medical context, missing a CKD case (false negative) is far more dangerous than a false alarm.
> - **Average model confidence** for CKD cases is 88-97%, and for non-CKD cases it's similarly high — thanks to the isotonic calibration.
>
> I evaluate using a **5-fold stratified cross-validation** during tuning and then a final evaluation on the held-out test set to ensure there's no data leakage.

---

## Q5. What features does your model use?

**Answer:**

> The model uses **24 features** in total:
>
> - **Demographics**: Age, Blood Pressure (diastolic)
> - **Urinalysis**: Specific Gravity (sg), Albumin (al), Sugar (su)
> - **CBC markers**: Hemoglobin, PCV, WBC count, RBC count
> - **Kidney function**: Blood Urea (bu), Serum Creatinine (sc), Blood Glucose (bgr)
> - **Electrolytes**: Sodium, Potassium
> - **Clinical flags** (binary): RBC normal/abnormal, Pus Cells (pc), Pus Cell Clumps (pcc), Bacteria (ba), Hypertension (htn), Diabetes (dm), Coronary Artery Disease (cad), Appetite, Pedal Edema (pe), Anemia (ane)
>
> For the OCR/upload flow, I extract as many values as possible from the report (typically Hemoglobin, RBC, WBC, PCV, Creatinine, Urea, Glucose, Sodium, Potassium). Non-extractable features like specific gravity, albumin, and clinical flags use sensible clinical defaults (e.g., sg=1.02, albumin=0). The imputer handles any remaining missing values with median imputation from the training set.

---

## Q6. Why did you build an OCR pipeline instead of using a form-only approach?

**Answer:**

> In a clinical workflow, patients already have lab reports — making them re-type 9+ values manually is error-prone and tedious. OCR automation dramatically reduces friction. The user just uploads a photo of their CBC report and the system extracts everything in seconds.
>
> The key insight is the **review step** — I don't blindly trust the OCR output. After extraction, the user sees all values on an editable review page where they can verify and correct any mistakes before the model runs. This "human-in-the-loop" design gives the best of both worlds: automation speed + human accuracy.

---

## Q7. Explain your OCR preprocessing pipeline.

**Answer:**

> I have **6 preprocessing pipelines**, each optimized for different image conditions:
>
> 1. **Raw Grayscale**: Resize → Grayscale → CLAHE → Light denoise. No thresholding — preserves tonal detail for digit accuracy.
> 2. **Sharp Grayscale**: Higher resize (3000px) → Grayscale → Unsharp mask sharpening → CLAHE → Light denoise. Also no thresholding.
> 3. **Printed**: Resize → Grayscale → CLAHE → Light denoise → Otsu thresholding → Morphological cleaning. Best for clean printed reports.
> 4. **Low Light**: Resize → Grayscale → Aggressive CLAHE (clip=4.0) → Medium denoise → Adaptive thresholding. For poorly lit photos.
> 5. **Blurry**: Large resize (3000px) → Grayscale → Sharpening → CLAHE → Otsu. For out-of-focus images.
> 6. **Low Contrast**: Resize → Grayscale → Histogram equalization → Adaptive thresholding. For washed-out images.
>
> Before all pipelines, I run **deskew detection** — first using Tesseract OSD for 90°/180°/270° rotation correction, then Hough line detection for micro-angle (< 5°) deskew.
>
> The critical design decision was adding the non-thresholded pipelines (#1 and #2). Binarization (thresholding) can fill in digit features — for example, the horizontal bar of "5" can get connected to the curve, making it look like "9" to Tesseract. Keeping the original grayscale preserves these subtle differences.

---

## Q8. How does the consensus voting work for OCR?

**Answer:**

> Instead of picking the single "best" preprocessing and trusting it blindly, I run OCR on **all 6 preprocessing candidates** with **3 different Tesseract page segmentation modes** (PSM 6 for block text, PSM 4 for column text, PSM 3 for automatic) — that's up to **18 independent OCR passes**.
>
> For each pass, I run my `extract_cbc()` parser to extract marker values. Then, for each marker (e.g., RBC), I collect all the values extracted across all passes and take the **most frequently occurring value** using a `Counter`.
>
> For example, if 14 passes read RBC as 5.2 and 4 passes read it as 9.2, the consensus picks 5.2. The key insight is that different preprocessing pipelines and PSM modes make **different errors** — so a digit misread on one pipeline is unlikely to be replicated across all 18 passes.
>
> This approach doesn't limit valid ranges — if someone truly has an RBC of 9.2, all 18 passes will consistently read 9.2 and the consensus correctly returns 9.2.

---

## Q9. Walk me through how you extract a value like RBC from OCR text.

**Answer:**

> It's a two-pass strategy in `extract_cbc()`:
>
> **Pass 1 — Label Matching:**
> I scan each line of OCR text against regex patterns for each marker. For RBC, the patterns include `\brbc\b`, `r.b.c`, `red blood cell`, `erythrocyte`, `rbc count`, etc. When a label is found, I call `extract_value()` which:
> 1. Normalizes comma-separated thousands (7,800 → 7800)
> 2. Handles scientific notation (7.8 × 10^3 → 7800)
> 3. Strips reference ranges (e.g., "4.5 - 5.5") from the text
> 4. Finds the first remaining number that passes a **physiological sanity check** (RBC must be between 1.0 and 10.0)
>
> If no value is found on the label line, I search **forward** (up to 6 lines), then **backward** (up to 5 lines) — this handles multi-column PDF layouts where the value appears before or after the label. I also try merging adjacent lines for cases where value and unit are on separate lines (e.g., "7.05\nthou/mm3").
>
> **Pass 2 — Range Fingerprinting:**
> For markers missed in Pass 1 (when labels are garbled by OCR), I look for recognizable reference ranges. For example, if I see "4.5 - 5.5" on a line, that's almost certainly the RBC row, so I extract the value from that line. This handles cases where "RBC" was misread as "R8C" or similar.

---

## Q10. How does the Kidney Health Score work?

**Answer:**

> It's a **penalty-based scoring system** that starts at 100 (perfect health) and deducts points for each abnormal marker. The penalties are clinically weighted:
>
> | Factor | Max Penalty | Rationale |
> |--------|-------------|-----------|
> | eGFR / CKD Stage | 35 pts | Most important single indicator of kidney function |
> | CKD Model Probability | 30 pts | ML model's assessment |
> | Serum Creatinine | 25 pts | Direct kidney function marker |
> | Hemoglobin | 20 pts | Anemia is a major CKD complication |
> | Blood Urea | 18 pts | Kidney waste clearance indicator |
> | Blood Pressure | 12 pts | Hypertension both causes and results from CKD |
> | PCV | 12 pts | Correlates with anemia severity |
> | Blood Glucose | 10 pts | Diabetic nephropathy risk |
> | Potassium | 10 pts | Life-threatening if too high in CKD |
> | Sodium | 8 pts | Fluid balance indicator |
> | RBC Count | 6 pts | Secondary anemia marker |
> | WBC Count | 6 pts | Infection/inflammation indicator |
>
> The penalties use **tiered thresholds** — e.g., creatinine of 1.4 gets 8 points, but creatinine of 8+ gets 25 points. The final score maps to grades: A (85+, Excellent), B (70+, Good), C (55+, Moderate), D (35+, Poor), F (<35, Critical).

---

## Q11. What is eGFR and how do you calculate it?

**Answer:**

> **eGFR (estimated Glomerular Filtration Rate)** measures how well the kidneys filter waste from the blood. It's the gold standard for staging CKD.
>
> I use the **simplified MDRD formula**:
> ```
> eGFR = 175 × (creatinine ^ -1.154) × (age ^ -0.203)
> ```
>
> This gives mL/min/1.73m². I then classify into **KDIGO CKD stages**:
> - ≥90: Stage 1 (normal function, possible kidney damage)
> - 60-89: Stage 2 (mild loss)
> - 45-59: Stage 3a (mild-moderate loss)
> - 30-44: Stage 3b (moderate-severe loss)
> - 15-29: Stage 4 (severe loss)
> - <15: Stage 5 (kidney failure)
>
> A limitation is that I'm using a simplified version that doesn't account for gender or race, since those aren't collected from the lab report. In production, I'd add gender as an input field.

---

## Q12. What's the tech stack?

**Answer:**

> - **Backend**: Python, Flask
> - **ML**: Scikit-learn (Random Forest, Gradient Boosting, Logistic Regression, VotingClassifier, CalibratedClassifierCV), pandas, numpy, joblib for model serialization
> - **OCR**: Tesseract (via pytesseract) as primary engine, EasyOCR as fallback
> - **Image Processing**: OpenCV (cv2) for preprocessing — CLAHE, denoising, thresholding, deskewing, sharpening
> - **PDF Handling**: PyMuPDF (fitz) for native text extraction and rendering PDF pages to images
> - **Data Balancing**: imbalanced-learn (SMOTE)
> - **Frontend**: Vanilla HTML/CSS/JavaScript with Jinja2 templating, Google Fonts (Inter)
> - **Design**: Custom CSS with CSS variables, responsive grid layout, animated gauge using SVG

---

## Q13. What challenges did you face?

**Answer:**

> 1. **OCR Digit Confusion (5 → 9)**: Tesseract's thresholding was consistently misreading "5" as "9" in certain fonts. The binarization process filled in the horizontal bar of "5", making it look like a closed "9". I solved this by adding non-thresholded preprocessing candidates and implementing multi-pass consensus voting across 18 OCR configurations.
>
> 2. **Multi-Column PDF Layouts**: Lab reports from different pathology labs have wildly different formats. Some put values before the label (right-to-left column layout). I had to implement both forward and backward search from label positions, with intelligent stop conditions to avoid crossing into other markers.
>
> 3. **Model Confidence Calibration**: The initial model was "overconfident" — giving 99.9% for everything. Removing SVM from the ensemble and adding isotonic calibration fixed this, giving genuine probability estimates in the 88-97% range.
>
> 4. **Password-Protected PDFs**: Some lab PDFs are encrypted. Instead of silently failing, I added detection using PyMuPDF and show a clear warning asking the user to upload a screenshot instead.
>
> 5. **Reference Range Confusion**: The OCR would sometimes pick up numbers from the reference range column (e.g., "4.5" from "4.5 - 5.5") instead of the result column. I solved this by stripping reference ranges (detected via regex) before extracting values.

---

## Q14. If you had more time, what improvements would you make?

**Answer:**

> 1. **Deep Learning OCR**: Replace Tesseract with a transformer-based document understanding model (like LayoutLMv3 or Donut) that understands table structure natively — no need for regex parsing.
>
> 2. **Longitudinal Tracking**: Let users create accounts and track their kidney health over time with trend charts. Detecting deterioration over 3-6 months is more clinically meaningful than a single snapshot.
>
> 3. **SHAP Explainability**: Add SHAP (SHapley Additive exPlanations) to show which markers contributed most to the CKD prediction, making the model more interpretable for doctors.
>
> 4. **Larger Dataset**: The UCI dataset has only 400 records. Training on a larger, more diverse clinical dataset would improve generalization.
>
> 5. **Gender/Race in eGFR**: Collect gender to use the CKD-EPI formula (more accurate than MDRD for higher GFR values).
>
> 6. **Mobile App**: A React Native or Flutter frontend for easy photo capture from phone cameras.

---

## Q15. How do you ensure the app doesn't give a wrong prediction?

**Answer:**

> Multiple safety layers:
>
> 1. **Human-in-the-loop**: The review page lets users verify and correct all OCR values before prediction. No value goes to the model unchecked.
>
> 2. **Sanity ranges**: Every extracted value is validated against physiological ranges (e.g., RBC must be 1.0-10.0). Values outside these ranges are rejected.
>
> 3. **Consensus voting**: 18 OCR passes with majority voting minimize extraction errors.
>
> 4. **Confidence reporting**: The model shows its confidence percentage — a 60% confidence prediction should be weighted differently than a 95% one.
>
> 5. **Medical disclaimer**: The UI prominently states this is "not a substitute for professional medical advice." It's a screening tool, not a diagnostic tool.
>
> 6. **Sensible defaults**: Missing markers use population median defaults rather than extreme values, preventing the model from being skewed by missing data.

---

## Q16. Why Flask and not Django or FastAPI?

**Answer:**

> Flask was the right choice because:
> - **Simplicity**: The app has only 3 routes (`/`, `/review`, `/predict`) — Django's ORM, admin panel, and middleware would be overkill.
> - **ML Integration**: Flask makes it trivial to load models at startup and run inference per-request without async complexity.
> - **Rapid Prototyping**: This is a proof-of-concept / academic project, and Flask let me iterate fast.
>
> If I were scaling this for production, I'd consider **FastAPI** for its async support and automatic OpenAPI documentation, especially if I was building a REST API consumed by a separate frontend.

---

## Q17. Explain the `SimpleImputer` — why median and not mean?

**Answer:**

> Median imputation is more robust to outliers than mean. Medical lab values often have extreme outliers (e.g., creatinine of 15 mg/dL in dialysis patients), and the mean would be pulled toward these extremes. The median gives a more representative "typical" value.
>
> The imputer is fit on the training data and saved alongside the model (`ckd_imputer.pkl`). At prediction time, I use the same trained imputer to transform the patient's data, ensuring consistency.

---

## Bonus: Quick-Fire Concepts

| Question | Answer |
|----------|--------|
| **What's CLAHE?** | Contrast Limited Adaptive Histogram Equalization — enhances local contrast in image sub-regions without amplifying noise |
| **What's Otsu thresholding?** | Automatically finds the optimal threshold to separate foreground from background in a bimodal histogram |
| **What's PSM in Tesseract?** | Page Segmentation Mode — tells Tesseract how to interpret the image layout (block, column, single line, etc.) |
| **What's soft voting?** | The ensemble averages the probability outputs of all models rather than taking a majority of predictions |
| **What's isotonic calibration?** | A non-parametric method that fits a monotonically increasing step function to map raw model scores to true probabilities |
| **What's SMOTE?** | Creates synthetic minority samples by interpolating between existing minority samples in feature space |
| **What's KDIGO?** | Kidney Disease: Improving Global Outcomes — the international guideline organization for CKD staging |
