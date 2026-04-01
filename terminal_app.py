#!/usr/bin/env python3
"""
Finance Internship Aggregator — Claude API Terminal App

Replaces Apify/Playwright scraping with Claude Opus 4.6 + built-in web search.
Claude searches multiple job boards, scores each listing intelligently, and
displays results in a rich terminal UI with interactive commands.

Requirements:
    pip install anthropic rich

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python terminal_app.py
    python terminal_app.py --location "New York" --hours 48
    python terminal_app.py --limit 30 --no-interactive
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEN_FILE    = OUTPUT_DIR / "seen_jobs.json"
APPLIED_FILE = OUTPUT_DIR / "applied_jobs.json"

FIRE_THRESHOLD = 15
GOOD_THRESHOLD = 10

console = Console()

# ── State tracking ────────────────────────────────────────────────────────────

def load_seen_urls() -> set[str]:
    if SEEN_FILE.exists():
        return {e["url"] for e in json.loads(SEEN_FILE.read_text()).get("seen", [])}
    return set()


def save_seen_urls(jobs: list[dict[str, Any]]) -> None:
    seen: list[dict] = []
    if SEEN_FILE.exists():
        seen = json.loads(SEEN_FILE.read_text()).get("seen", [])
    existing = {e["url"] for e in seen}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    for job in jobs:
        if job.get("url") and job["url"] not in existing:
            seen.append({"url": job["url"], "seen_at": now})
            existing.add(job["url"])
    SEEN_FILE.write_text(json.dumps({"seen": seen}, indent=2))


def load_applied_urls() -> set[str]:
    if APPLIED_FILE.exists():
        return {e["url"] for e in json.loads(APPLIED_FILE.read_text()).get("applied", [])}
    return set()


def mark_applied(job: dict[str, Any]) -> None:
    applied: list[dict] = []
    if APPLIED_FILE.exists():
        applied = json.loads(APPLIED_FILE.read_text()).get("applied", [])
    if job.get("url") and job["url"] not in {e["url"] for e in applied}:
        applied.append({
            "url":        job["url"],
            "title":      job.get("title", ""),
            "company":    job.get("company", ""),
            "applied_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        })
        APPLIED_FILE.write_text(json.dumps({"applied": applied}, indent=2))

# ── Claude job search ─────────────────────────────────────────────────────────

# User-defined tool: Claude calls this to hand back structured job data.
# Web search (server-side) is declared separately and Claude uses it freely.
REPORT_TOOL: dict[str, Any] = {
    "name": "report_jobs",
    "description": (
        "Call this tool once you have finished web-searching to report all "
        "finance internship jobs you found. Sort jobs by score descending."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "jobs": {
                "type": "array",
                "description": "All finance internship jobs found, sorted by score desc.",
                "items": {
                    "type": "object",
                    "properties": {
                        "title":       {"type": "string"},
                        "company":     {"type": "string"},
                        "location":    {"type": "string"},
                        "url":         {"type": "string"},
                        "posted":      {"type": "string",
                                        "description": "Date posted (YYYY-MM-DD or relative)"},
                        "description": {"type": "string",
                                        "description": "1-2 sentence role summary"},
                        "score":       {"type": "integer", "description": "Relevance score 0-25+"},
                        "reason":      {"type": "string",  "description": "Scoring explanation"},
                        "priority":    {"type": "string",
                                        "enum": ["fire", "good", "neutral"]},
                    },
                    "required": [
                        "title", "company", "location", "url",
                        "description", "score", "reason", "priority",
                    ],
                },
            },
        },
        "required": ["jobs"],
    },
}

SYSTEM_PROMPT = """\
You are a finance internship aggregator. Search for internship opportunities
using multiple web queries, then call report_jobs with all results.

Search strategy — run these queries:
1. Investment banking summer analyst internships (Goldman Sachs, Morgan Stanley, JPMorgan, Citi, etc.)
2. Private equity analyst intern positions (Blackstone, KKR, Apollo, Carlyle, etc.)
3. M&A analyst internship opportunities
4. Hedge fund analyst intern positions (Bridgewater, Citadel, Two Sigma, etc.)
5. Asset management internship positions
6. Venture capital analyst internship positions
7. Bulge-bracket / elite boutique finance internships (Lazard, Evercore, Moelis, etc.)

For each posting extract: title, company, location, URL, posting date, description.

Scoring (add all that apply):
  +5  M&A role
  +4  Investment banking / IB role
  +4  Private equity / PE role
  +3  Hedge fund role
  +3  Asset management role
  +3  Venture capital / VC role
  +4  Top-tier firm (GS, MS, JPM, Citi, BofA, Blackstone, KKR, Apollo, Carlyle,
      Bridgewater, Citadel, Two Sigma, Lazard, Evercore, Moelis, Rothschild, etc.)
  +2  Finance / investment keyword in title
  +1  Analyst role
  +1  Intern / internship keyword
  +3  Posted < 2 days ago
  +2  Posted 2–7 days ago
  +1  Posted 8–30 days ago

Priority: score >= 15 → "fire", score >= 10 → "good", else → "neutral"

After all searches, call report_jobs once with all results (aim for 10–20 jobs),
sorted by score descending.\
"""


def search_jobs(
    client: anthropic.Anthropic,
    location: str | None,
    hours: int,
    limit: int,
) -> list[dict[str, Any]]:
    """Run Claude with web search to find and score finance internships."""

    location_clause = f" in {location}" if location else ""
    time_clause = (
        "posted in the last 24 hours" if hours <= 24
        else f"posted in the last {hours // 24} days"
    )

    user_message = (
        f"Search for finance internship opportunities{location_clause}, "
        f"{time_clause}. "
        f"Use multiple search queries and aim for at least {min(limit, 15)} "
        f"high-quality results. Then call report_jobs."
    )

    messages: list[dict] = [{"role": "user", "content": user_message}]
    tools: list[dict] = [
        {"type": "web_search_20260209", "name": "web_search"},
        REPORT_TOOL,
    ]

    found_jobs: list[dict[str, Any]] = []
    search_count = 0
    max_iterations = 6  # guard against infinite loops

    for _ in range(max_iterations):
        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
            thinking={"type": "adaptive"},
        ) as stream:
            for event in stream:
                if (
                    event.type == "content_block_start"
                    and hasattr(event.content_block, "type")
                    and event.content_block.type == "server_tool_use"
                ):
                    search_count += 1
                    console.print(f"  [dim cyan]🔍 Web search #{search_count}…[/dim cyan]")

            response = stream.get_final_message()

        # Append assistant turn for conversation history
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason == "pause_turn":
            # Server-side tool loop hit its iteration cap — re-send to continue.
            # The API detects the trailing server_tool_use block and resumes.
            continue

        if response.stop_reason == "tool_use":
            # Handle our user-defined report_jobs tool call
            tool_results: list[dict] = []
            for block in response.content:
                if block.type == "tool_use" and block.name == "report_jobs":
                    found_jobs = block.input.get("jobs", [])
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     f"Received {len(found_jobs)} jobs.",
                    })

            if found_jobs:
                # We have what we need — no need to continue the loop
                break

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

    return found_jobs


# ── Terminal display ──────────────────────────────────────────────────────────

PRIORITY_ICON: dict[str, str] = {"fire": "🔥", "good": "🟡", "neutral": "⚪"}


def render_table(
    jobs: list[dict[str, Any]],
    seen: set[str],
    applied: set[str],
) -> None:
    t = Table(
        title=f"[bold]Finance Internships[/bold] — {len(jobs)} results",
        box=box.ROUNDED,
        show_lines=False,
        header_style="bold cyan",
        border_style="dim",
    )
    t.add_column("#",        style="dim",        width=3,  justify="right")
    t.add_column("",         width=2)
    t.add_column("Title",    style="bold white", min_width=32)
    t.add_column("Company",  style="cyan",       min_width=18)
    t.add_column("Location", style="green",      min_width=14)
    t.add_column("Score",    justify="right",    width=6)
    t.add_column("Status",   width=9)

    for i, job in enumerate(jobs, 1):
        url        = job.get("url", "")
        is_applied = url in applied
        is_new     = url not in seen
        priority   = job.get("priority", "neutral")
        score      = job.get("score", 0)

        status = (
            "[red]Applied[/red]"           if is_applied else
            "[bright_green]NEW[/bright_green]" if is_new  else ""
        )
        score_str = (
            f"[red bold]{score}[/red bold]"        if score >= FIRE_THRESHOLD else
            f"[yellow bold]{score}[/yellow bold]"  if score >= GOOD_THRESHOLD else
            f"[dim]{score}[/dim]"
        )
        t.add_row(
            str(i),
            PRIORITY_ICON.get(priority, "⚪"),
            job.get("title",    "—"),
            job.get("company",  "—"),
            job.get("location", "—"),
            score_str,
            status,
        )

    console.print(t)


def render_detail(job: dict[str, Any]) -> None:
    score    = job.get("score", 0)
    icon     = PRIORITY_ICON.get(job.get("priority", "neutral"), "⚪")
    score_color = (
        "red bold"    if score >= FIRE_THRESHOLD else
        "yellow bold" if score >= GOOD_THRESHOLD else
        "white"
    )

    lines = []
    if job.get("posted"):
        lines.append(f"[dim]Posted: {job['posted']}[/dim]")
    lines.append(f"[bold]{job.get('title', '—')}[/bold]")
    lines.append(
        f"[cyan]{job.get('company', '—')}[/cyan]  ·  "
        f"[green]{job.get('location', '—')}[/green]"
    )
    lines.append("")
    lines.append(f"Score: [{score_color}]{score}[/{score_color}] {icon}")
    lines.append(f"[dim]{job.get('reason', '—')}[/dim]")
    lines.append("")
    lines.append(job.get("description", "No description available."))
    lines.append("")
    lines.append(f"[blue underline]{job.get('url', 'No URL')}[/blue underline]")

    console.print(Panel("\n".join(lines), border_style="cyan", title="Job Detail"))


# ── Interactive session ───────────────────────────────────────────────────────

def run_interactive(
    jobs: list[dict[str, Any]],
    seen: set[str],
    applied: set[str],
) -> str:
    """Return 'refresh' or 'quit'."""
    console.print(
        "\n[dim]Commands:  [cyan]<n>[/cyan] view detail  "
        "[cyan]a<n>[/cyan] mark applied  "
        "[cyan]r[/cyan] refresh  "
        "[cyan]q[/cyan] quit[/dim]\n"
    )

    while True:
        try:
            cmd = console.input("[bold]→ [/bold]").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return "quit"

        if cmd == "q":
            return "quit"
        if cmd == "r":
            return "refresh"

        if cmd.startswith("a") and len(cmd) > 1 and cmd[1:].isdigit():
            idx = int(cmd[1:]) - 1
            if 0 <= idx < len(jobs):
                mark_applied(jobs[idx])
                applied.add(jobs[idx].get("url", ""))
                console.print(
                    f"[green]✓ Applied: {jobs[idx].get('title')} "
                    f"@ {jobs[idx].get('company')}[/green]"
                )
                render_table(jobs, seen, applied)
            else:
                console.print(f"[red]Enter a number between 1 and {len(jobs)}[/red]")
            continue

        if cmd.isdigit():
            idx = int(cmd) - 1
            if 0 <= idx < len(jobs):
                render_detail(jobs[idx])
            else:
                console.print(f"[red]Enter a number between 1 and {len(jobs)}[/red]")
            continue

        if cmd:
            console.print("[dim]Unknown command.[/dim]")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Finance Internship Aggregator powered by Claude API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--hours",          type=int,  default=168,
                   help="Lookback window in hours (168 = 1 week)")
    p.add_argument("--limit",          type=int,  default=20,
                   help="Target number of jobs to find")
    p.add_argument("--location",       type=str,  default=None,
                   help="Location filter, e.g. 'New York', 'London'")
    p.add_argument("--no-interactive", action="store_true",
                   help="Print results and exit (no interactive prompt)")
    args = p.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red bold]Error:[/red bold] ANTHROPIC_API_KEY is not set.")
        console.print("  Set it with:  [cyan]export ANTHROPIC_API_KEY=sk-ant-...[/cyan]")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    console.print(Panel.fit(
        "[bold cyan]Finance Internship Aggregator[/bold cyan]\n"
        "[dim]Claude Opus 4.6 · Web Search · Terminal UI[/dim]",
        border_style="cyan",
    ))
    console.print()

    while True:
        seen    = load_seen_urls()
        applied = load_applied_urls()

        # Print search parameters
        params = []
        if args.location:
            params.append(f"location=[green]{args.location}[/green]")
        params.append(
            f"window=[yellow]{args.hours}h[/yellow] "
            f"({'last 24h' if args.hours <= 24 else f'last {args.hours // 24}d'})"
        )
        params.append(f"target=[cyan]{args.limit} jobs[/cyan]")
        console.print("  " + "  ·  ".join(params) + "\n")

        t0   = time.perf_counter()
        jobs = search_jobs(client, args.location, args.hours, args.limit)
        elapsed = time.perf_counter() - t0

        if not jobs:
            console.print(
                "[yellow]No jobs found. Try extending --hours or removing --location.[/yellow]"
            )
        else:
            new_count = sum(1 for j in jobs if j.get("url") and j["url"] not in seen)
            console.print(
                f"\n[bold green]✓ Found {len(jobs)} internships[/bold green] "
                f"([bright_green]{new_count} new[/bright_green]) "
                f"in {elapsed:.0f}s\n"
            )
            save_seen_urls(jobs)
            seen = load_seen_urls()
            render_table(jobs, seen, applied)

        if args.no_interactive:
            break

        action = run_interactive(jobs, seen, applied)
        if action != "refresh":
            break
        console.print()

    console.print("[dim]\nGoodbye![/dim]")


if __name__ == "__main__":
    main()
