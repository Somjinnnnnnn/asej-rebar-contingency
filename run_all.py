"""Reproduce every table and figure of the paper, in order.

Steps (outputs/ folder in brackets):
   1  data audit and code splicing                     [audit_2026-10]
   2  global driver coverage and lead-lag check        [audit_2026-10]
   3  backtest round 1, SD40 12 mm                     [backtest_main]
   4  backtest round 1, SD50 and SR24 12 mm            [backtest_robust_grades]
   5  backtest round 1, drivers lagged one more month  [backtest_robust_lag1]
   6  backtest round 2 (CV tuning, two driver sets), run in parallel shards and assembled
                                                       [backtest_v2]
   7  conformal recalibration                          [backtest_v2]
   8  1-20-year contingency bands                      [projection_2026-09]
   9  validation on the US steel PPI, nominal and real [validation_wpu1017, validation_wpu1017_real]
  10  analyses of the pre-submission reviews (uncertainty, procurement example, sensitivity)
                                                       [revision_2026-10]
  11  manuscript tables and figures                    [manuscript]
  12  submission Word files (manuscript, supplementary material, highlights)  [manuscript]

Usage:
    python run_all.py                 # everything (about 1.5-2 hours on a 16-thread laptop)
    python run_all.py --from 8        # start at step 8
    python run_all.py --workers 6     # fewer parallel processes in steps 3-6 and 9
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = [sys.executable, '-W', 'ignore']
DRAFT = HERE / 'outputs' / 'manuscript' / 'Manuscript_rebar_ASEJ_draft.docx'  # manuscript text, not archived
V2 = ['--out-dir', 'backtest_v2', '--driver-sets', 'all drivers', 'scrap+ore', '--tune', 'both']


def run(cmd: list[str]) -> None:
    print('>', ' '.join(cmd), flush=True)
    subprocess.run(PY + cmd, cwd=HERE, check=True)


def run_parallel(cmds: list[list[str]], workers: int) -> None:
    pending, running = list(cmds), []
    while pending or running:
        while pending and len(running) < workers:
            cmd = pending.pop(0)
            print('>', ' '.join(cmd), flush=True)
            running.append(subprocess.Popen(PY + cmd, cwd=HERE, stdout=subprocess.DEVNULL))
        for p in list(running):
            if p.poll() is not None:
                running.remove(p)
                if p.returncode != 0:
                    raise SystemExit(f'failed: {" ".join(p.args)}')
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--from', dest='start', type=int, default=1)
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()
    steps = {
        1: lambda: run(['audit_rebar.py']),
        2: lambda: run(['check_drivers.py']),
        3: lambda: run_parallel([['backtest_rebar.py', '--out-dir', 'backtest_main'],
                                 ['backtest_rebar.py', '--series', 'SD50 12mm', 'SR24 12mm',
                                  '--out-dir', 'backtest_robust_grades'],
                                 ['backtest_rebar.py', '--extra-lag', '1', '--out-dir', 'backtest_robust_lag1']],
                                args.workers),
        4: lambda: None,  # run together with step 3
        5: lambda: None,  # run together with step 3
        6: lambda: (run_parallel([['backtest_rebar.py', *V2, '--shard', f'{i}/{args.workers}']
                                  for i in range(args.workers)], args.workers),
                    run(['backtest_rebar.py', *V2])),
        7: lambda: run(['conformal_rebar.py', '--run', 'backtest_v2']),
        8: lambda: run(['projection_rebar.py']),
        9: lambda: (run(['validate_long_horizon.py', '--workers', str(args.workers)]),
                    run(['validate_long_horizon.py', '--deflator', 'CPIAUCSL', '--out-dir', 'validation_wpu1017_real',
                         '--workers', str(args.workers)])),
        10: lambda: (run(['revision_analyses.py', '--workers', str(args.workers)]), run(['revision2_analyses.py']),
                     run(['revision3_analyses.py'])),
        11: lambda: (run(['manuscript_tables.py']), run(['revision_tables.py']), run(['manuscript_figures.py'])),
        12: lambda: (run(['build_manuscript_docx.py']) if DRAFT.exists()
                     else print(f'step 12 skipped: {DRAFT.name} (manuscript text) is not in the repository')),
    }
    t0 = time.time()
    for k in sorted(steps):
        if k >= args.start:
            steps[k]()
    print(f'done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
