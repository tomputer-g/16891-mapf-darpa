import sys, re, io

maps = ['generated/darpa%d.txt' % i for i in range(1, 8)]

from SSICA_main import run_simulation as sim

results = []
for m in maps:
    sys.stdout = io.StringIO()
    sim(m, max_steps=500, verbose=False, use_vis=False)
    out = sys.stdout.getvalue()
    sys.stdout = sys.__stdout__
    match = re.search(r'finished in (\d+) steps', out)
    r = match.group(1) if match else 'TIMEOUT'
    results.append((m, r))

with open('ssica_results.txt', 'w') as f:
    for m, r in results:
        f.write(f'{m}: {r}\n')

print('Done.')
