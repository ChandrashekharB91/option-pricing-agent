"""Phase 1 regression suite: compare every frozen output with the golden file and run the QuantLib gates.

    python scripts/regression.py                         check; exit 1 on any failure
    python scripts/regression.py --update --reason "..." accept the current numbers as the new golden values

Writes reports/regression_report.md either way. Run it before a demo and before every merge.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optlib_app import regression as reg  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update", action="store_true", help="write the current numbers as the golden values")
    ap.add_argument("--reason", default="", help="why the golden values change (required with --update)")
    ap.add_argument("--section", action="append", choices=list(reg.SECTIONS), help="run only these sections")
    ap.add_argument("--report", default=str(ROOT / "reports" / "regression_report.md"))
    args = ap.parse_args()
    if args.update and not args.reason.strip():
        ap.error("--update needs --reason: say what changed and why")

    t0 = time.time()
    current = reg.compute(args.section)
    if args.update:
        if args.section:
            ap.error("--update rewrites every section; run it without --section")
        changed = reg.write_golden(current, args.reason.strip())
        print(f"Golden values written: {sum(map(len, current.values()))} values, {len(changed)} changed or added.")
        for r in changed[:30]:
            print(f"  {r['status']:<7} {r['section']}/{r['key']}: {r['golden']} -> {r['current']}")
        print(f"Logged in {reg.CHANGELOG.relative_to(ROOT)}. Commit both files and get the change reviewed.")

    golden = reg.load_golden()
    rows = reg.compare(golden["values"], current)
    gate_rows = reg.gates()
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(reg.report_markdown(rows, gate_rows, golden["meta"]))

    summary = reg.summarise(rows)
    width = max(len(s["section"]) for s in summary)
    print(f"\nGolden values ({golden['meta']['updated']}: {golden['meta']['reason']})")
    for s in summary:
        print(f"  {s['status']:<4}  {s['section']:<{width}}  {s['values']:>5} values, {s['failed']} failed, "
              f"largest difference {s['max_diff']:.1e}")
    print("Benchmark gates")
    for g in gate_rows:
        print(f"  {'PASS' if g['passed'] else 'FAIL':<4}  {g['gate']:<24} {g['detail']} (limit {g['limit']})")
    ok = all(s["status"] == "PASS" for s in summary) and all(g["passed"] for g in gate_rows)
    print(f"\n{'PASS' if ok else 'FAIL'} in {time.time() - t0:.0f}s; report: {Path(args.report).relative_to(ROOT)}")
    if not ok:
        for r in [r for r in rows if r["status"] != "PASS"][:20]:
            print(f"  {r['status']:<7} {r['section']}/{r['key']}: golden {r['golden']}, now {r['current']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
