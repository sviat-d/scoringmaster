"""Google Sheets integration: read domains, score, write results back in real-time."""

import os
import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import AsyncGenerator

import gspread

from scorer.pipeline import score_single

logger = logging.getLogger(__name__)

# Import modes to trigger registration (same as pipeline.py)
import scorer.mode_inxy  # noqa: F401
import scorer.mode_founders  # noqa: F401

ENRICHED_HEADERS = [
    "Use Case",
    "Detected Industry",
    "Business Model",
    "Product Use Cases",
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

# Batch size for writing results (balance between real-time feel and API limits)
WRITE_BATCH_SIZE = 5
CONCURRENCY_LIMIT = 3


def _get_gspread_client() -> gspread.Client:
    """Create gspread client from service account credentials."""
    # Option 1: JSON credentials in env var
    creds_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if creds_json:
        creds_dict = json.loads(creds_json)
        return gspread.service_account_from_dict(creds_dict)

    # Option 2: Path to credentials file
    creds_path = os.environ.get(
        "GOOGLE_CREDENTIALS_PATH",
        "credentials.json",
    )
    if os.path.exists(creds_path):
        return gspread.service_account(filename=creds_path)

    raise RuntimeError(
        "Google credentials not found. Set GOOGLE_CREDENTIALS_JSON env var "
        "or place credentials.json in project root."
    )


def is_sheets_available() -> bool:
    """Check if Google Sheets credentials are configured."""
    if os.environ.get("GOOGLE_CREDENTIALS_JSON"):
        return True
    creds_path = os.environ.get("GOOGLE_CREDENTIALS_PATH", "credentials.json")
    return os.path.exists(creds_path)


def _parse_sheet_id(url_or_id: str) -> str:
    """Extract spreadsheet ID from URL or return as-is if already an ID."""
    url_or_id = url_or_id.strip()
    # Full URL: https://docs.google.com/spreadsheets/d/SPREADSHEET_ID/edit...
    if "docs.google.com" in url_or_id:
        parts = url_or_id.split("/d/")
        if len(parts) > 1:
            return parts[1].split("/")[0].split("?")[0]
    # Already an ID
    return url_or_id


def _find_domain_col(headers: list[str]) -> int | None:
    """Find the column index (0-based) containing 'domain' in header."""
    for i, h in enumerate(headers):
        if "domain" in h.lower():
            return i
    return None


def _result_to_row(result: dict, mode_id: str) -> list[str]:
    """Convert scoring result dict to a row of enriched column values."""
    return [
        mode_id,
        result.get("industry", "Unknown"),
        result.get("business_model", "Unknown"),
        ", ".join(result.get("use_cases", [])),
        result.get("headcount_estimate", "Unknown"),
        result.get("crypto_adoption_likelihood", "Low"),
        ", ".join(result.get("risk_flags", [])),
        str(result.get("score", 1)),
        result.get("category", "Reject"),
        result.get("reason_short", ""),
        " \u2022 ".join(result.get("reasons_bullets", [])),
        result.get("confidence", "Low"),
        result.get("opener", ""),
        result.get("next_action", ""),
        datetime.now(timezone.utc).isoformat(),
    ]


async def score_sheet(
    sheet_url: str,
    mode_id: str,
    sheet_name: str = "",
    llm_provider: str = "",
) -> AsyncGenerator[dict, None]:
    """Score all domains in a Google Sheet. Yields progress events.

    Args:
        sheet_url: Google Sheet URL or spreadsheet ID.
        mode_id: Scoring mode (inxy_leads or founders_pl).
        sheet_name: Worksheet tab name. Empty string = first sheet.
        llm_provider: LLM provider to use ("anthropic", "gemini", etc).

    Events:
        {"event": "start", "total": N, "sheet_title": "..."}
        {"event": "progress", "current": i, "total": N, "domain": "...", "category": "..."}
        {"event": "batch_written", "rows_written": N}
        {"event": "done", "total": N, "summary": {...}}
        {"event": "error", "message": "..."}
    """
    try:
        gc = _get_gspread_client()
    except Exception as e:
        yield {"event": "error", "message": f"Google auth failed: {e}"}
        return

    sheet_id = _parse_sheet_id(sheet_url)

    try:
        spreadsheet = gc.open_by_key(sheet_id)
        if sheet_name.strip():
            try:
                worksheet = spreadsheet.worksheet(sheet_name.strip())
            except gspread.exceptions.WorksheetNotFound:
                available = [ws.title for ws in spreadsheet.worksheets()]
                yield {
                    "event": "error",
                    "message": f"Sheet tab '{sheet_name}' not found. "
                    f"Available tabs: {', '.join(available)}",
                }
                return
        else:
            worksheet = spreadsheet.sheet1
    except gspread.exceptions.SpreadsheetNotFound:
        yield {
            "event": "error",
            "message": "Spreadsheet not found. Make sure you shared it with the service account email.",
        }
        return
    except gspread.exceptions.APIError as e:
        yield {"event": "error", "message": f"Google API error: {e}"}
        return
    except Exception as e:
        yield {"event": "error", "message": f"Failed to open spreadsheet: {e}"}
        return

    # Read all data
    all_values = worksheet.get_all_values()
    if not all_values:
        yield {"event": "error", "message": "Sheet is empty"}
        return

    headers = all_values[0]
    domain_col = _find_domain_col(headers)
    if domain_col is None:
        yield {
            "event": "error",
            "message": "No column with 'domain' in header found. "
            "Please add a column like 'Company Domain'.",
        }
        return

    data_rows = all_values[1:]
    total = len(data_rows)

    if total == 0:
        yield {"event": "error", "message": "No data rows in sheet"}
        return

    # Add enriched headers if not already present
    # Find last non-empty header column (don't count empty grid columns)
    last_filled_col = 0
    for i, h in enumerate(headers):
        if h.strip():
            last_filled_col = i + 1

    # Check if enriched headers already exist
    enriched_header_set = set(ENRICHED_HEADERS)
    existing_headers_set = set(h.strip() for h in headers if h.strip())

    if enriched_header_set.issubset(existing_headers_set):
        # Headers already exist — find where they start
        for i, h in enumerate(headers):
            if h == "Score":
                enriched_start_col = i - ENRICHED_HEADERS.index("Score")
                break
    else:
        # Place enriched columns right after last filled column
        enriched_start_col = last_filled_col

        # Expand grid if needed (sheet might not have enough columns)
        needed_cols = enriched_start_col + len(ENRICHED_HEADERS)
        current_cols = worksheet.col_count
        if needed_cols > current_cols:
            worksheet.resize(cols=needed_cols)

        # Write enriched headers
        header_cells = []
        for i, h in enumerate(ENRICHED_HEADERS):
            col_letter = _col_to_letter(enriched_start_col + i)
            header_cells.append({"range": f"{col_letter}1", "values": [[h]]})
        worksheet.batch_update(header_cells)

    # Detect already-scored rows by checking "Score" column
    score_col_idx = enriched_start_col + ENRICHED_HEADERS.index("Score")
    rows_to_score: list[int] = []  # indices into data_rows
    skipped = 0

    for i, row in enumerate(data_rows):
        # Row already scored if Score cell has a value
        if score_col_idx < len(row) and row[score_col_idx].strip():
            skipped += 1
        else:
            rows_to_score.append(i)

    to_score = len(rows_to_score)

    yield {
        "event": "start",
        "total": to_score,
        "skipped": skipped,
        "sheet_title": spreadsheet.title,
    }

    if to_score == 0:
        yield {
            "event": "done",
            "total": 0,
            "skipped": skipped,
            "summary": {"A": 0, "B": 0, "C": 0, "Reject": 0},
        }
        return

    # Score domains with concurrency control
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    cache: dict = {}

    async def _score_one(row_idx: int, domain: str):
        async with semaphore:
            if not domain or domain.lower() in ("n/a", "na", "-", ""):
                return (row_idx, {
                    "score": 0,
                    "category": "Reject",
                    "reason_short": "No domain provided",
                })
            result = await score_single(domain, mode_id, cache, llm_provider=llm_provider)
            return (row_idx, result)

    # Process only unscored rows, yield progress, write in batches
    pending_writes: list[tuple[int, list[str]]] = []
    summary = {"A": 0, "B": 0, "C": 0, "Reject": 0}

    tasks = [
        _score_one(i, data_rows[i][domain_col] if domain_col < len(data_rows[i]) else "")
        for i in rows_to_score
    ]

    for coro in asyncio.as_completed(tasks):
        row_idx, result = await coro
        domain = data_rows[row_idx][domain_col] if domain_col < len(data_rows[row_idx]) else ""
        category = result.get("category", "Reject")
        summary[category] = summary.get(category, 0) + 1

        yield {
            "event": "progress",
            "current": sum(summary.values()),
            "total": to_score,
            "domain": domain,
            "score": result.get("score", 0),
            "category": category,
        }

        enriched_row = _result_to_row(result, mode_id)
        pending_writes.append((row_idx, enriched_row))

        # Write batch when enough accumulated
        if len(pending_writes) >= WRITE_BATCH_SIZE:
            _write_batch(worksheet, pending_writes, enriched_start_col)
            yield {"event": "batch_written", "rows_written": len(pending_writes)}
            pending_writes = []

    # Write remaining
    if pending_writes:
        _write_batch(worksheet, pending_writes, enriched_start_col)
        yield {"event": "batch_written", "rows_written": len(pending_writes)}

    yield {
        "event": "done",
        "total": to_score,
        "skipped": skipped,
        "summary": summary,
    }


def _write_batch(
    worksheet: gspread.Worksheet,
    rows: list[tuple[int, list[str]]],
    start_col: int,
) -> None:
    """Write a batch of enriched results to the worksheet."""
    batch = []
    for row_idx, values in rows:
        sheet_row = row_idx + 2  # +1 for header, +1 for 1-based indexing
        start_letter = _col_to_letter(start_col)
        end_letter = _col_to_letter(start_col + len(values) - 1)
        cell_range = f"{start_letter}{sheet_row}:{end_letter}{sheet_row}"
        batch.append({"range": cell_range, "values": [values]})

    try:
        worksheet.batch_update(batch)
    except gspread.exceptions.APIError as e:
        logger.warning(f"Batch write failed, retrying in 2s: {e}")
        import time
        time.sleep(2)
        try:
            worksheet.batch_update(batch)
        except Exception:
            logger.error(f"Batch write retry failed: {e}")


def _col_to_letter(col_idx: int) -> str:
    """Convert 0-based column index to spreadsheet letter (0=A, 25=Z, 26=AA)."""
    result = ""
    idx = col_idx
    while True:
        result = chr(65 + idx % 26) + result
        idx = idx // 26 - 1
        if idx < 0:
            break
    return result
