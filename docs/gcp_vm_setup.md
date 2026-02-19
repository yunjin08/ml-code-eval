# Create a GPU VM on GCP (Console only)

Use these steps in the **Google Cloud Console** in your browser. No `gcloud` CLI required.

---

## 1. Open Compute Engine

1. Go to [console.cloud.google.com](https://console.cloud.google.com).
2. Select your **project** (or create one) from the top project dropdown.
3. Open the **☰** menu (top left) → **Compute Engine** → **VM instances**.

---

## 2. Create the VM

1. Click **"+ CREATE INSTANCE"** at the top.

2. **Name**  
   - Example: `phase2-train` (any name is fine).

3. **Region and zone**  
   - You **must** pick a zone where **N1+T4** (or your chosen GPU) is available.  
   - **Suggested zones for NVIDIA T4** (from [GCP GPU locations](https://cloud.google.com/compute/docs/gpus/gpu-regions-zones)):  
     - **us-central1-a**, **us-central1-c**, **us-central1-f** (Iowa)  
     - **us-east1-c**, **us-east1-d** (South Carolina)  
     - **us-east4-a**, **us-east4-b**, **us-east4-c** (Virginia)  
     - **us-west1-a**, **us-west1-b** (Oregon)  
     - **europe-west1-b**, **europe-west1-c**, **europe-west4-b**, **europe-west4-c**  
   - In the **Zone** dropdown, choose one of the above (e.g. **us-central1-a**).  
   - If you don’t see the GPU option in the next step, try another zone from the list.

4. **Machine configuration**  
   - **Machine type:** **General-purpose** → **N2** or **N1** → choose **n2-standard-8** (8 vCPUs, 32 GB memory) or **n1-standard-8**.  
   - Click **GPU** (under Machine configuration).  
     - **GPU type:** **NVIDIA T4**.  
     - **Number of GPUs:** **1**.  
   - If T4 is not available in your zone, try another zone or **NVIDIA L4** if listed.

5. **Boot disk**  
   - Click **CHANGE** under Boot disk.  
   - **Operating system:** **Ubuntu**.  
   - **Version:** **Ubuntu 22.04 LTS**.  
   - **Disk size:** **50 GB** (or 100 GB if you prefer).  
   - Click **SELECT**.

6. **Firewall**  
   - Leave **Allow HTTP traffic** and **Allow HTTPS traffic** as you prefer (not required for training).  
   - You only need **SSH** (port 22), which GCP allows by default for “SSH” from the browser.

7. **Advanced options** (optional)  
   - Under **Availability, disks, networking, sole tenancy** you can leave defaults.  
   - If you want the VM to **stop (not delete)** when you’re done: after creation you can stop it from the VM list.

8. Click **CREATE**.

Wait until the VM appears in the list with a green checkmark and an **External IP**.

---

## 3. Connect to the VM (browser SSH)

1. In **VM instances**, find your VM.
2. In the **Connect** column, click **SSH** (dropdown) → **Open in browser window** (or the icon that opens a terminal in the browser).
3. A terminal opens in a new window/tab. You’re now in the VM.

---

## 4. Upload your project and data

You need the repo (or at least `src/`, `requirements.txt`) and Phase 1 outputs: `data/curated_cpp.csv`, `data/splits.json`.

**Option A – Upload from your computer (Console)**

1. In the **browser SSH** window, click the **⚙️ (Settings)** icon or **⋮** menu → **Upload file**.
2. Upload in one go (or in parts):
   - A **ZIP** of your project (including `data/`, `src/`, `requirements.txt`), **or**
   - Individual files: `requirements.txt`, then create `src/` and `data/` and upload the contents (e.g. `curated_cpp.csv`, `splits.json`, and all of `src/`).
3. If you uploaded a ZIP, in the SSH terminal run:
   ```bash
   unzip your-project.zip -d ~/project && cd ~/project
   ```
   If you uploaded folders, move them so you have:
   ```
   ~/project/
     src/
     data/
       curated_cpp.csv
       splits.json
     requirements.txt
   ```

**Option B – Clone from GitHub (if repo is on GitHub)**

1. In the SSH terminal:
   ```bash
   sudo apt-get update && sudo apt-get install -y git unzip
   git clone https://github.com/YOUR_USERNAME/code-reviewer-thesis.git ~/project
   cd ~/project
   ```
2. Then upload **only** the `data/` folder (with `curated_cpp.csv` and `splits.json`) via **Upload file** in the SSH window, into `~/project/data/`, because `data/` is usually not in the repo.

---

## 5. Install Python and dependencies on the VM

In the **browser SSH** terminal (with `cd ~/project` or your project path):

```bash
sudo apt-get update
sudo apt-get install -y python3.10-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Then install **PyTorch with CUDA** (for NVIDIA GPU). Choose the right line for your CUDA version (GCP T4 is often CUDA 11.8 or 12.1; try 11.8 first):

```bash
# CUDA 11.8 (common on Ubuntu 22.04)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

If that fails or you prefer CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 6. Run Phase 2

Still in the same terminal, with venv active and from the project directory:

```bash
nohup python src/phase2_train.py --epochs 3 --batch_size 16 > phase2_train.log 2>&1 &
tail -f phase2_train.log
```

- Press **Ctrl+C** to stop following the log (training keeps running).
- To check progress later: `tail -100 phase2_train.log`.

When it finishes, you’ll have:

- `src/models/codebert/` and possibly `src/models/rf.pkl`
- `results/phase2_validation_report.json`

---

## 7. Download results back to your computer

**Option A – Download from browser SSH**

- In the SSH window: **⋮** menu → **Download file**.
- Enter path, e.g. `/home/YOUR_USERNAME/project/results/phase2_validation_report.json`, then `src/models/codebert` (you may need to zip it first: `zip -r codebert.zip src/models/codebert`).

**Option B – Use Console “Upload/Download”**

- Some SSH UIs have “Download file” where you type the path and download.  
- For folders, zip first in the VM:
  ```bash
  cd ~/project && zip -r phase2_results.zip results/ src/models/
  ```
  Then download `phase2_results.zip`.

---

## 8. When you’re done: stop the VM to save cost

1. Go to **Compute Engine** → **VM instances**.
2. Select your VM.
3. Click **STOP** at the top.  
   The VM stops; you are not charged for compute while it’s stopped. You can **START** it again later.  
   To avoid any charges, you can **DELETE** the VM when you no longer need it (you’ll lose anything not downloaded or in a persistent disk).

---

## Quick reference

| Step            | Where / What |
|-----------------|--------------|
| Create VM       | Compute Engine → VM instances → CREATE INSTANCE |
| GPU             | Machine configuration → GPU → 1× NVIDIA T4 |
| Boot disk       | Ubuntu 22.04 LTS, 50 GB |
| Connect         | VM list → SSH → Open in browser |
| Upload          | SSH window → Upload file (ZIP or data + code) |
| Install + run   | `python3 -m venv .venv` → `pip install -r requirements.txt` → `pip install torch --index-url ...cu118` → `python src/phase2_train.py ...` |
| Stop VM         | VM instances → Select VM → STOP |
