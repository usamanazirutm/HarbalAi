# HerbalSalt-AI (NHANES edition) - Setup

## Folder layout (create these inside D:\My research papers\Harbal Salt)
```
Harbal Salt\
  01_raw_data\      <- downloaded NHANES files (auto-filled by script 1)
  02_processed\     <- analytic dataset (auto-filled by script 2)
  03_scripts\       <- put 01_download_nhanes.py, 02_build_dataset.py, requirements.txt here
  04_results\       <- reports, tables, figures
  05_manuscript\    <- move the old .tex here
```

## Run
1. Install Python 3.11+ from python.org (tick "Add Python to PATH").
2. Open Command Prompt:
```
cd /d "D:\My research papers\Harbal Salt\03_scripts"
pip install -r requirements.txt
python 01_download_nhanes.py
python 02_build_dataset.py
```
3. Send back the contents of `04_results\feasibility_report.txt`.

If script 1 lists any file as MISSING, open the URL it prints in a browser,
download the .xpt file, and save it into the folder named in the message.
