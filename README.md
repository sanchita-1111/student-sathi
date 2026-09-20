# Student Sathi

Student Sathi is a Design Thinking + Data Science prototype for early student support.

## What changed
- Three simple working areas: Home, Student Check-in, Mentor Hub.
- No fixed subject list in the interface. Students enter their own subjects and marks.
- Mentor Hub groups support cues by the subject names entered by students.
- Human-first workflow: signal -> mentor -> subject teacher -> guidance -> follow-up.
- Visual UI with a calm student-facing interface.
- Synthetic demo dataset: 1,600 records.
- Power BI-ready Excel workbook included.

## Model
Logistic Regression with StandardScaler and balanced class weights.
Held-out synthetic test metrics: Accuracy 73.4%, Precision 56.0%, Recall 70.0%, F1 62.2%, ROC-AUC 79.7%.

These metrics are demonstration-only and should not be presented as validated real-world performance.
