# Setup notes — record of how this environment was built

A plain record of the commands used to set up the project, in order, with the
reasons behind the non-obvious ones. For a one-command rebuild, see
`bootstrap.sh`; this file is the human-readable history.

## 1. Project folder + virtual environment

```bash
cd microbiome-agent          # work inside the project folder
python3 -m venv .venv        # (first attempt — see note below)
source .venv/bin/activate    # activate; prompt shows (.venv)
```

> Note: the first venv was built on conda base's Python **3.8**, which was too
> old (the project requires >= 3.10). It was deleted and rebuilt on 3.12 — see
> step 3.

## 2. Correcting the conda state

There were two anaconda installs on the machine; the shell had landed on the
wrong one. Realigned onto the conda that owns the real environments:

```bash
conda activate base          # prompt returns to (base) at /opt/anaconda3
conda env list               # confirm * is on the right base
```

## 3. Rebuild the venv on Python 3.12 (the important fix)

```bash
deactivate                   # leave the old (3.8) venv
rm -rf .venv                 # delete it — a venv is disposable

conda create -n py312 python=3.12 -y     # py312 = a SOURCE of Python 3.12
conda run -n py312 python -m venv .venv  # build venv FROM py312's Python
source .venv/bin/activate
python --version             # must report Python 3.12.x
```

> Key idea: conda only supplied the 3.12 interpreter. The venv is independent
> after creation. What decides which Python runs is whichever env is *active*
> now — verify with `which python` (should end in .../microbiome-agent/.venv/bin/python).

## 4. Install dependencies + the project, run tests

```bash
python -m pip install --upgrade pip      # old pip couldn't read pyproject.toml
pip install -r requirements.txt
pip install -e .                         # editable install -> makes the package importable
pytest -q                                # expect: all tests pass
```

> Note: `pip install -e .` only worked after upgrading pip AND being on Python
> >= 3.10. On the old pip/old Python it failed two different ways.

## 5. Daily use (every new terminal)

```bash
source .venv/bin/activate    # re-activate each session
which python                 # source of truth if the prompt looks ambiguous
```

## 6. Git / GitHub (one-time)

```bash
# install + log in
conda install -c conda-forge gh -y       # or: brew install gh
gh --version
gh auth login                            # GitHub.com -> HTTPS -> browser login

# identity (first time only)
git config --global user.name "Yihui Sun"
git config --global user.email "yhsun0207@gmail.com"

# first commit + push
git add .
git status                               # CHECK: .venv must NOT appear
git commit -m "Phase 1: analysis tools with tests"
gh repo create microbiome-agent --public --source=. --remote=origin --push

# everyday rhythm afterward
git add .
git commit -m "describe the change"
git push
```

## Cosmetic prompt quirks (harmless)

- `((.venv) ) (base)` vs `(base) ((.venv) )` — prefix ORDER is just which env
  labelled the prompt most recently. It changes nothing. `which python` is truth.
- `(/Users/yihui/opt/anaconda3)` instead of `(base)` — base shown by full path;
  cosmetic, caused by the duplicate conda install.

## Known cleanup for another day (not urgent)

- Two anaconda installs exist (`/opt/anaconda3` and `~/opt/anaconda3`). Removing
  the duplicate is a careful operation — defer it; don't risk it mid-project.
