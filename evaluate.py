"""
Standalone CLI runner for DebateArena Evaluation.

Usage:
    python evaluate.py --latest
    python evaluate.py --transcript data/evals/transcript_20250101_120000.json --llm-eval
    python evaluate.py --batch data/evals/
"""
import os
import sys
import json
import argparse
from pathlib import Path
from evaluation import evaluate_debate, DebateEvalReport
from ui import render_eval_report


def find_latest_transcript(search_dir: Path) -> Path:
    if not search_dir.exists():
        raise FileNotFoundError(f"Directory {search_dir} does not exist.")
    transcripts = list(search_dir.glob("transcript_*.json")) + list(search_dir.glob("*.json"))
    # Filter out files that are eval reports
    transcripts = [t for t in transcripts if not t.name.startswith("eval_report_")]
    if not transcripts:
        raise FileNotFoundError(f"No debate transcript JSON files found in {search_dir}.")
    transcripts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return transcripts[0]


def main():
    parser = argparse.ArgumentParser(description="Evaluate DebateArena debate transcripts.")
    parser.add_argument("--transcript", type=str, help="Path to transcript JSON file.")
    parser.add_argument("--batch", type=str, help="Directory containing multiple transcript JSON files.")
    parser.add_argument("--latest", action="store_true", help="Evaluate the most recent transcript in data/evals/.")
    parser.add_argument("--llm-eval", action="store_true", help="Enable LLM-judge metrics.")
    parser.add_argument("--output", type=str, help="Optional output JSON path to save evaluation report.")

    args = parser.parse_args()
    evals_dir = Path(__file__).parent / "data" / "evals"

    if args.batch:
        batch_dir = Path(args.batch)
        if not batch_dir.exists():
            print(f"Batch directory '{batch_dir}' not found.")
            sys.exit(1)
        files = [f for f in batch_dir.glob("*.json") if not f.name.startswith("eval_report_")]
        print(f"Running batch evaluation on {len(files)} transcripts in {batch_dir}...")

        batch_reports = []
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                report = evaluate_debate(data, use_llm_judge=args.llm_eval)
                batch_reports.append((file_path.name, report))
                print(f"[OK] Evaluated {file_path.name} (Turns: {report.total_turns}, Support Rate: {report.overall_citation_support_rate:.0%})")
            except Exception as e:
                print(f"[FAIL] Failed to evaluate {file_path.name}: {e}")

        print(f"\nBatch evaluation complete. Processed {len(batch_reports)} debates.")
        return

    transcript_path = None
    if args.transcript:
        transcript_path = Path(args.transcript)
    elif args.latest or not args.transcript:
        try:
            transcript_path = find_latest_transcript(evals_dir)
            print(f"Evaluating latest transcript: {transcript_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Run a debate first with `python main.py` or specify `--transcript <path>`.")
            sys.exit(1)

    if not transcript_path.exists():
        print(f"File not found: {transcript_path}")
        sys.exit(1)

    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    report = evaluate_debate(data, use_llm_judge=args.llm_eval)
    render_eval_report(report)

    # Save output report
    output_path = args.output
    if not output_path:
        os.makedirs(evals_dir, exist_ok=True)
        report_name = f"eval_report_{transcript_path.stem}.json"
        output_path = evals_dir / report_name

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"\nSaved evaluation report to: {output_path}")


if __name__ == "__main__":
    main()
