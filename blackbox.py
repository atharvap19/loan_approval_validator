"""
Black-box client for the Instant Loan Approval Model.

    from blackbox import predict, predict_batch

    applicant = {
        "age": 34,                  # years
        "monthly_income": 72000,    # INR per month
        "loan_amount": 200000,      # INR
        "tenure_months": 24,        # 12, 18, 24, 36, 48 or 60
        "employment_years": 7,      # years in current job / business
        "credit_score": 760,        # 300-900
        "existing_loans": 1,        # number of active loans
        "pincode": 560034,          # 6-digit Indian PIN code
    }
    predict(applicant)
    # -> {'risk_score': 0.01, 'decision': 'approve'}

    predict_batch([applicant1, applicant2, ...])   # much faster for many applicants

The decision threshold is 0.5. Treat the model as a black box: call it only through
these functions. Do not open, decompile or inspect the .onnx file.
"""
import os
import numpy as np
import onnxruntime as ort

FEATURES = ["age", "monthly_income", "loan_amount", "tenure_months",
            "employment_years", "credit_score", "existing_loans", "pincode"]

_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "loan_model.onnx")
_sess = ort.InferenceSession(_MODEL, providers=["CPUExecutionProvider"])


def predict_batch(applicants):
    applicants = list(applicants)
    if not applicants:
        return []
    for a in applicants:
        missing = [f for f in FEATURES if f not in a]
        if missing:
            raise ValueError(f"missing fields: {missing}")
    X = np.array([[float(a[f]) for f in FEATURES] for a in applicants], dtype=np.float32)
    probs = _sess.run(None, {"applicant": X})[1][:, 1]
    return [{"risk_score": round(float(p), 4),
             "decision": "reject" if p >= 0.5 else "approve"} for p in probs]


def predict(applicant: dict):
    return predict_batch([applicant])[0]


if __name__ == "__main__":
    a = {"age": 34, "monthly_income": 72000, "loan_amount": 200000, "tenure_months": 24,
         "employment_years": 7, "credit_score": 760, "existing_loans": 1, "pincode": 560034}
    print(predict(a))
