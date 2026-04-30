import sys, os, re, io

from src.allocators.Naive.main import run_simulation as naive_sim
from src.allocators.SSIA.main import run_simulation as ssia_sim
from src.allocators.SSICA.main import run_simulation as ssica_sim
from src.allocators.SSIA_collateral.main import run_simulation as collateral_sim

maps = [os.path.join('generated', 'darpa%d.txt' % i) for i in range(1, 8)]

RUNS = [
    (naive_sim,         'results/naive_results.txt'),
    (ssia_sim,          'results/ssia_results.txt'),
    (ssica_sim,         'results/ssica_results.txt'),
    (collateral_sim,    'results/ssia_collateral_results.txt'),
]

for sim, out_path in RUNS:
    results = []
    for m in maps:
        sys.stdout = io.StringIO()
        sim(m, max_steps=500, verbose=False, use_vis=False)
        out = sys.stdout.getvalue()
        sys.stdout = sys.__stdout__
        match = re.search(r'finished in (\d+) steps', out)
        results.append((m, match.group(1) if match else 'TIMEOUT'))

    with open(out_path, 'w') as f:
        for m, r in results:
            f.write(f'{m}: {r}\n')

    print(f'Done: {out_path}')
