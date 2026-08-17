# Plant Disease Classification

Learning project: understand CNNs from scratch in PyTorch, then progress through
transfer learning, feature extraction, partial fine-tuning, and full fine-tuning --
all compared on the same dataset, splits, and metrics.

Dataset: [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) (color variant,
38 classes, 14 crop species, ~54K images).

## Project layout

```
configs/        YAML config (paths, hyperparams) -- single source of truth
src/
  utils/        env detection, config loader
  data/         dataset fetch script, Dataset/DataLoader (later phases)
notebooks/      exploratory / milestone notebooks (call into src/, no logic lives here)
experiments/
  logs/         TensorBoard + CSV run logs
  checkpoints/  model weights
```

The dataset itself is **not** stored in this folder (kept out of version control /
Drive sync on purpose, to avoid tracking large binary files):

- **Local**: `~/plant_disease_dataset` by default (resolves under the current user's
  home directory on any machine/OS, no setup needed) -- override with the `DATA_ROOT`
  environment variable for a custom location (e.g. a specific drive with more space).
- **Colab**: `/content/dataset` (downloaded fresh each session -- Colab runtimes are
  ephemeral anyway, and local disk is much faster than Drive-mounted I/O for training)

`src/utils/config.py` resolves which root to use automatically (`src/utils/env.py`
detects local vs. Colab), so the same code runs unmodified in both places -- and
unmodified across machines, since nothing about the path is hardcoded.

## Setup -- local

A venv's thousands of small package files are bad to sync/version -- so the venv
lives outside this project folder entirely, on local disk. Unlike the dataset root,
there's no config-driven override for this path; it's just wherever you choose to
create it. The commands below use `D:\Diljith_A_K\.venvs\plant_disease` as this
machine's example -- substitute any local path on yours (e.g.
`~/.venvs/plant_disease` on Linux/Mac).

```powershell
python -m venv D:\Diljith_A_K\.venvs\plant_disease
D:\Diljith_A_K\.venvs\plant_disease\Scripts\python.exe -m pip install --upgrade pip

cd "path/to/Plant_Disease_Classification_Project"
D:\Diljith_A_K\.venvs\plant_disease\Scripts\pip.exe install -r requirements.txt
D:\Diljith_A_K\.venvs\plant_disease\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cpu
D:\Diljith_A_K\.venvs\plant_disease\Scripts\python.exe -m src.data.fetch_data
```

To open notebooks locally (VSCode's Jupyter extension or classic Jupyter), register
the venv as a kernel once:

```powershell
D:\Diljith_A_K\.venvs\plant_disease\Scripts\python.exe -m ipykernel install --user --name plant_disease --display-name "Plant Disease (venv)"
```

Then select **"Plant Disease (venv)"** as the notebook's kernel. Already done for
this project -- if you open a notebook and it's not selected, pick it from the
kernel picker in the top right.

Local is CPU-only, so it's meant for pipeline/debugging checks on a small data slice --
not for real training.

## Setup -- Colab

```python
from google.colab import drive
drive.mount('/content/drive')

%cd "/content/drive/MyDrive/Colab Notebooks/Plant_Disease_Classification_Project"
!pip install -r requirements.txt   # torch/torchvision already preinstalled with CUDA -- don't reinstall
!python -m src.data.fetch_data
```

Real training happens here, using the GPU runtime.

## Verify the setup

```bash
python -m src.utils.config
```

Should print the resolved config as JSON, including `env` (`local`/`colab`) and the
resolved dataset `path`. After fetching data, that path should contain 38 class folders.

## Roadmap

See PHASES.md for the full plan, current status, and findings for every phase.

## Contributing / working from another machine or agent

See CONTRIBUTING.md for coding conventions and how to add a new phase. See
CLAUDE.md for settled design decisions and current project status, written for
picking this repo back up cold (on another machine, or with a different AI agent).

## Version control

This project moved off Google Drive sync and onto git/GitHub as the code-distribution
mechanism (Drive sync corrupted files more than once during development -- a renamed
`.gitignore`, a silently-dropped character in a notebook cell -- neither of which git
would allow). `.gitignore` already excludes large/regenerable artifacts (dataset,
checkpoints, TensorBoard logs) while keeping small evidence files like
`experiments/logs/**/metrics.csv` tracked.

**Workflow:** edits happen locally and get pushed to GitHub (`git add` / `commit` /
`push`) as normal. Colab is **pull-only** -- each session starts with `git clone` (or
`git pull` if already cloned in that session) to fetch the latest pushed code, runs
from there, and does not push anything back. This keeps GitHub credentials out of
Colab entirely; results/findings get carried back into commits manually rather than
Colab committing on its own. Each notebook's former `drive.mount()` bootstrap cell is
replaced with a `git clone`/`git pull` cell once the GitHub repo exists.
