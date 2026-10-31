# Uncharted

An AI research lab built on Meta's Muse models that hunts for new world records on open mathematical problems.

- **Muse Spark** workers write solver programs. Each program runs in a sandbox and is scored by an exact verifier.
- Workers **look at a rendered picture** of every solution (contact graphs, slack profiles, packings) before writing the next program.
- **Muse Wonder** acts as principal investigator: it reads the top programs and the best picture, diagnoses why progress stalls, and assigns new research directions.

## Results

| Problem | Published record | Ours | Status |
|---|---|---|---|
| Spherical code, 153 points in 4D (max cosine, lower is better) | 0.849102358280 ([spherical-codes.org](https://spherical-codes.org/)) | **0.849102250950** | New record, verified at 50 digits |
| Spherical code, 163 points in 4D | 0.856415430312 | **0.856414859773** | New record, verified at 50 digits |
| Spherical code, 168 points in 4D | 0.859313315478 | **0.859313146285** | New record, verified at 50 digits |
| Spherical code, 272 points in 4D | 0.898160315150 | **0.898159960309** | New record, verified at 50 digits |
| 26 circles in a square (sum of radii, higher is better) | 2.6359830849 | 2.6359404147 | Beats AlphaEvolve's 2.635863 |

Evidence for each record (coordinates, the AI-written program, picture, verification) is in [`records/`](records/).

## Watch the lab

**[Live replay page](https://eg-shiva.github.io/Uncharted/)**: replay any run program by program. You see the idea behind each program, the picture the worker studied, Muse Wonder's diagnoses, and the moment the record falls.

Locally, `python -m discovery.web` serves the same page at http://127.0.0.1:8765, reading `runs/` live while the engine works. `python tools/export_site.py` rebuilds the static copy in `docs/`.

## Run it

```powershell
python -m venv .venv
.\.venv\Scripts\pip install openai numpy scipy matplotlib python-dotenv mpmath
# put MODEL_API_KEY=... in .env
.\.venv\Scripts\python -m discovery.run --problem spherical_code_4d_168 --iterations 100 --workers 8 --budget 5
```

Verify a code independently:

```powershell
.\.venv\Scripts\python tools\verify_code.py records\spherical_4d_168\code_4_168.csv 0.859313315479
```
