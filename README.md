# Uncharted

An AI research lab built on Meta's Muse models that hunts for new world records on open mathematical problems.

- **Muse Spark** workers write solver programs. Each program runs in a sandbox and is scored by an exact verifier.
- Workers **look at a rendered picture** of every solution (contact graphs, slack profiles, packings) before writing the next program.
- **Muse Wonder** acts as principal investigator: it reads the top programs and the best picture, diagnoses why progress stalls, and assigns new research directions.

## Results

| Problem | Published record | Ours | Status |
|---|---|---|---|
| Spherical code, 68 points in 12D (max cosine, lower is better) | 0.219978834878 ([spherical-codes.org](https://spherical-codes.org/)) | **0.219960833863** | New record, verified at 50 digits |
| Spherical code, 71 points in 12D | 0.230211592242 | **0.230137617429** | New record, verified at 50 digits |
| Spherical code, 173 points in 5D | 0.758561688405 | **0.758561249594** | New record, verified at 50 digits |
| Spherical code, 82 points in 14D | 0.198235113449 | **0.198206312378** | New record, verified at 50 digits |
| Spherical code, 453 points in 5D | 0.848240351265 | **0.848235083286** | New record, verified at 50 digits |
| Spherical code, 54 points in 21D | 0.066663628471 | **0.066663400245** | New record, verified at 50 digits |
| Spherical code, 323 points in 7D | 0.664998948391 | **0.664998796049** | New record, verified at 50 digits |
| Spherical code, 55 points in 21D | 0.066665975723 | **0.066665901555** | New record, verified at 50 digits |
| Spherical code, 78 points in 15D | 0.168274222547 | **0.168274198183** | New record, verified at 50 digits |
| Spherical code, 189 points in 12D | 0.354668194086 | **0.354668191490** | New record (small: 2.6e-9), verified at 50 digits |
| Spherical code, 76 points in 15D | 0.168263492524 | **0.168263490248** | New record (small: 2.3e-9), verified at 50 digits |
| Spherical code, 116 points in 6D | 0.596363660542 | **0.596362844051** | New record, verified at 50 digits |
| Spherical code, 153 points in 4D | 0.849102358280 | **0.849102176174** | New record, verified at 50 digits |
| Spherical code, 163 points in 4D | 0.856415430312 | **0.856414859773** | New record, verified at 50 digits |
| Spherical code, 168 points in 4D | 0.859313315478 | **0.859313146285** | New record, verified at 50 digits |
| Spherical code, 272 points in 4D | 0.898160315150 | **0.898159960309** | New record, verified at 50 digits |
| Spherical code, 292 points in 4D | 0.903471504338 | **0.903470816563** | New record, verified at 50 digits |
| 26 circles in a square (sum of radii, higher is better) | 2.6359830849 | 2.6359404147 | Beats AlphaEvolve's 2.635863, not the current best |

The sphere searches start from the best published arrangement and look for a better one. Plain local polishing of the published codes gives no improvement, so each record required the AI to rearrange the points.

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
