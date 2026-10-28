# Take-home task: Audit a black-box AI model
**Zeal Connect · AI Engineer (Intern) · Time limit: 5 hours**

---

## The situation

A digital lending company (an NBFC) uses an AI model to decide instant personal-loan applications in its app. Every application is sent to one of two outcomes:

- **approve**: the loan offer is shown to the customer immediately
- **reject**: the application is declined, and the customer cannot reapply for 90 days

The model was trained on three years of historical approve/reject decisions made by the company's credit team. On the company's validation set, it scored **over 98% accuracy**, so it was put into production.

A few weeks later, the business team is not happy:

> *"Some very well-off salaried applicants were rejected instantly and complained on social media. Approval numbers also look strange in places, and nobody can say why."*

Nobody can explain why. The original team has left, and there is no documentation.

**Your job: find out what is wrong with this model, and prove it.**

---

## What you get

| File | What it is |
|---|---|
| `blackbox.py` | The only way to use the model. `predict(applicant)` and `predict_batch(applicants)` |
| `model/loan_model.onnx` | The model file, used by `blackbox.py` |
| `data/validation_applications.csv` | 600 applications with the decisions used to validate the model |
| `FINDINGS_TEMPLATE.md` | The format for your final report |
| `requirements.txt` | Python dependencies |

**Setup (2 minutes)**
```bash
pip install -r requirements.txt
python blackbox.py        # should print one prediction
```
It runs on any laptop CPU. No GPU and no internet needed.

**Input fields:** `age`, `monthly_income` (₹/month), `loan_amount` (₹), `tenure_months` (12/18/24/36/48/60), `employment_years`, `credit_score` (300–900), `existing_loans` (count of active loans), `pincode` (6-digit Indian PIN code). See `blackbox.py` for an example.

---

## Rules

1. **Use the model only through `blackbox.py`.** Do not open, decompile or inspect the `.onnx` file, its weights or its graph. Your conclusions must come from experiments on inputs and outputs. You will be asked how you found each issue.
2. **AI tools (ChatGPT, Claude, Copilot and so on) are allowed** for writing code and reading up on methods. You must understand and be able to explain everything you submit.
3. **Work alone.** This task is individual. Other candidates have a different task.
4. **Time limit: 5 hours.** A partial result with solid evidence is worth more than a long list of guesses.

---

## What we expect from you

The model has **more than one** problem. Some are easy to find; some are not. For each problem you find:

1. **What it is.** State it precisely. "It is biased" is not enough; say exactly what input property causes exactly what wrong behaviour.
2. **Evidence.** Controlled experiments that prove it, for example the same applicant with one field changed, repeated over many applicants, with numbers.
3. **How you found it.** What made you suspect it, and what you tried that did *not* pan out.
4. **Likely cause.** Your best explanation of why the model learned this.
5. **Impact.** Who is affected and how badly.
6. **Fix.** What you would change: data, labels, model or process.

Then answer one final question: **Should this model stay in production? Why or why not?**

---

## Deliverables

Put everything in one folder or GitHub repo:

- `FINDINGS.md`: your report, using the template provided
- Your code: notebooks or scripts, runnable, showing every experiment behind your claims
- Any probe datasets you created (CSV is fine)

You will then do a **30-minute walkthrough** with us. Be ready to run your experiments live and to answer questions about them.

---

## How you will be evaluated

| Area | Weight |
|---|---|
| Problems found, stated precisely | 35% |
| Quality of evidence (controlled, repeatable, quantified) | 25% |
| Research process (how you searched, what you ruled out) | 15% |
| Root cause, impact and fixes | 10% |
| Walkthrough | 15% |

A short report with three well-proven findings will beat a long report with eight unproven ones.

---

## A note on approach

There is no single correct method. Think like a scientist: form a hypothesis, design a test that could prove it wrong, run it, and look at the numbers. Read about how people audit and stress-test models; it will help. Look closely at the data you have been given, and at how it was labelled.

Good luck.
