# Linear Programming Solver

Solve linear programs with 1–8 decision variables and up to 12 constraints.
The solver works through the problem step by step, lists every corner point,
reports shadow prices, flags special cases (infeasible, unbounded, multiple
optima, degenerate) and draws the result.

The same engine runs in two ways:

| Front end | File | Start it with |
|---|---|---|
| Web app (Streamlit) | `streamlit_app.py` | `streamlit run streamlit_app.py` |
| Terminal program | `lp_app.py` | `python lp_app.py` |

## Repository layout

```
├── streamlit_app.py        # web front end (entry point for Streamlit Cloud)
├── lp_app.py               # terminal front end
├── lp_engine.py            # parsing, solving, corner points, shadow prices
├── lp_input_guide.py       # examples, input/symbol guides, special cases
├── lp_plotting.py          # feasible region / 3-D region / solution charts
├── lp_utils.py             # shared helpers
├── test_streamlit_app.py   # runs every example through the web app
├── requirements.txt        # packages Streamlit Cloud installs
├── .streamlit/config.toml  # colours and server settings
└── .gitignore
```

All `lp_*.py` files must stay in the same folder as `streamlit_app.py`.

## Run it on your own computer

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app opens at http://localhost:8501.

## Put it on GitHub

1. Create a new repository on GitHub (it can be public or private).
2. From this folder:

   ```bash
   git init
   git add .
   git commit -m "Linear programming solver with Streamlit front end"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```

   Make sure the hidden `.streamlit` folder is pushed too (`git add .` includes it).

## Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with your GitHub account.
2. Click **Create app** and choose to deploy from a GitHub repository.
3. Fill in:
   - **Repository:** `<your-username>/<your-repo>`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
4. Optional: under **Advanced settings**, pick a Python version (3.10 or newer).
5. Click **Deploy**. Streamlit installs `requirements.txt` and gives you a public URL to share with students.

Every `git push` to `main` updates the live app automatically.

## Using the web app

- **Example problem** – pick any of the 13 built-in examples (basic, intermediate, advanced, special cases) and press **Solve**.
- **Build your own** – set the number of variables, their names, max/min, the objective and one constraint per line, e.g.

  ```
  2x + y <= 100 # Machine 1
  x + 2y <= 80  # Machine 2
  ```

  Non-negativity is added automatically. The objective can be typed as `40x + 30y` or `40, 30`.
- **Guides** – the input guide, symbol guide and special-cases table.

Results appear in four tabs: **Picture** (with a PNG download), **Step by step**,
**Corner points** and **Sensitivity** (slack, binding constraints, shadow prices).

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: lp_engine` | All `lp_*.py` files must be in the repository root next to `streamlit_app.py`. |
| `ModuleNotFoundError: scipy` (or another package) | Check that `requirements.txt` is in the repository root and spelled exactly like that. |
| App works locally but not on the cloud | Open **Manage app → Logs** on Streamlit Cloud to see the error message. |
| Large models feel slow | 8 variables × 12 constraints checks up to 125,970 corner combinations; results are cached, so the same model is only solved once. |
