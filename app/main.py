import uuid
import csv
import io
import os
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from scorer.pipeline import score_leads

app = FastAPI(title="Lead Scorer", version="1.0.0")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# In-memory store for completed jobs
_jobs: dict[str, bytes] = {}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/score")
async def score(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Form("inxy_leads"),
):
    if mode not in ("inxy_leads", "founders_pl"):
        mode = "inxy_leads"

    raw = await file.read()
    text = raw.decode("utf-8-sig")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return HTMLResponse("<h3>Empty or invalid CSV</h3>", status_code=400)

    rows = list(reader)
    if not rows:
        return HTMLResponse("<h3>No data rows in CSV</h3>", status_code=400)

    domain_col = _find_domain_column(reader.fieldnames)
    if not domain_col:
        return HTMLResponse(
            "<h3>Column 'Company Domain' not found. "
            "Please include a column with header containing 'domain'.</h3>",
            status_code=400,
        )

    enriched = await score_leads(rows, domain_col, mode)

    job_id = str(uuid.uuid4())
    output = _build_csv(reader.fieldnames, enriched, mode)
    _jobs[job_id] = output.encode("utf-8")

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "job_id": job_id,
            "total": len(enriched),
            "cat_a": sum(1 for r in enriched if r.get("_category") == "A"),
            "cat_b": sum(1 for r in enriched if r.get("_category") == "B"),
            "cat_c": sum(1 for r in enriched if r.get("_category") == "C"),
            "cat_reject": sum(1 for r in enriched if r.get("_category") == "Reject"),
        },
    )


@app.get("/download/{job_id}")
async def download(job_id: str):
    data = _jobs.get(job_id)
    if not data:
        return HTMLResponse("<h3>Job not found or expired</h3>", status_code=404)

    return StreamingResponse(
        io.BytesIO(data),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=scored_{job_id[:8]}.csv"},
    )


ENRICHED_COLUMNS = [
    "Use Case",
    "Detected Industry",
    "Business Model",
    "Headcount Estimate",
    "Crypto Adoption Likelihood",
    "Risk Flags",
    "Score",
    "Category",
    "Reason",
    "Reasons",
    "Confidence",
    "Opener",
    "Next Action",
    "Checked At",
]


def _find_domain_column(fieldnames: list[str]) -> str | None:
    for f in fieldnames:
        if "domain" in f.lower():
            return f
    return None


def _build_csv(original_fields: list[str], enriched: list[dict], mode: str) -> str:
    out = io.StringIO()
    all_fields = list(original_fields) + ENRICHED_COLUMNS
    writer = csv.DictWriter(out, fieldnames=all_fields, extrasaction="ignore")
    writer.writeheader()

    for row in enriched:
        result = row.get("_result", {})
        row["Use Case"] = mode
        row["Detected Industry"] = result.get("industry", "Unknown")
        row["Business Model"] = result.get("business_model", "Unknown")
        row["Headcount Estimate"] = result.get("headcount_estimate", "Unknown")
        row["Crypto Adoption Likelihood"] = result.get("crypto_adoption_likelihood", "Low")
        row["Risk Flags"] = ", ".join(result.get("risk_flags", []))
        row["Score"] = result.get("score", 1)
        row["Category"] = result.get("category", "Reject")
        row["Reason"] = result.get("reason_short", "")
        row["Reasons"] = " \u2022 ".join(result.get("reasons_bullets", []))
        row["Confidence"] = result.get("confidence", "Low")
        row["Opener"] = result.get("opener", "")
        row["Next Action"] = result.get("next_action", "")
        row["Checked At"] = datetime.now(timezone.utc).isoformat()
        writer.writerow(row)

    return out.getvalue()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
