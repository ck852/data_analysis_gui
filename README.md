[![Tests](https://github.com/ck852/patchbatch/actions/workflows/test.yml/badge.svg)](https://github.com/ck852/patchbatch/actions/workflows/test.yml)
[![PyPI version](https://badge.fury.io/py/patchbatch.svg)](https://badge.fury.io/py/patchbatch)
[![Python Version](https://img.shields.io/pypi/pyversions/patchbatch)](https://pypi.org/project/patchbatch/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub release](https://img.shields.io/github/v/release/ck852/patchbatch)](https://github.com/ck852/patchbatch/releases)

# PatchBatch - Electrophysiology Data Analysis Tool

## **Installation** 

## Option 1: Download Executable
1. Go to **[Releases](https://github.com/ck852/patchbatch/releases)**
    
    Windows:

    2. Download `PatchBatch-Windows.zip` 
    
    3. Extract and run `PatchBatch.exe`

    Mac:

    2. Download `PatchBatch-macOS.dmg`
    
    3. Double-click to mount
    
    4. First run: Right-click > Open (to bypass Gatekeeper)

## Option 2: Install from PyPI (Recommended if you have any installation issues with the executables)

### To update:

To update your version if you installed from PyPI, open a terminal and enter `pip install --upgrade patchbatch`

Alternatively, you may need to enter `python3 -m pip install --upgrade patchbatch`

### Prerequisites: Installing Python

If you don't have Python installed:

1. **Download Python**: Visit [python.org](https://www.python.org/downloads/) and download Python 3.9 or newer
2. **Install Python**: Run the installer and **check "Add Python to PATH"** during installation (Windows) or use the default settings (Mac/Linux)
3. **Verify Installation**: Open a terminal/command prompt and type `python --version` or `python3 --version` to confirm installation



### Opening a Terminal/Command Prompt

**Windows:**
- Press `Win + R`, type `cmd`, press Enter
- OR: Right-click in any folder while holding Shift, select "Open PowerShell window here"

**Mac:**
- Press `Cmd + Space`, type `Terminal`, press Enter
- OR: Finder → Applications → Utilities → Terminal

**Linux:**
- Press `Ctrl + Alt + T`
- OR: Right-click desktop, select "Open Terminal"

In your terminal, run these commands:

```bash
pip install patchbatch
patchbatch
```

If pip isn't recognized, instead try

`python -m pip install patchbatch`

**If python is not a recognized command, try replacing `python` with `python3` in all instances.**

If you install from PyPI, you will always start the program by opening a terminal in the same directory where you installed it and simply enter `patchbatch`. 



## Option 3: Install from Source (For Development)
```bash
git clone https://github.com/ck852/patchbatch.git
cd patchbatch
pip install -r requirements.txt
python src/data_analysis_gui/main.py
```

## Contributing
Found a bug or want to contribute? Please open an issue at:
https://github.com/ck852/patchbatch/issues


## Introduction

Welcome to PatchBatch! The purpose of this program is to streamline electrophysiology data analysis workflows by enabling batch-analysis of data files that share the same analysis parameters. I developed this after growing impatient with the long, tedious workflows that require defining the same parameters repeatedly for every file, followed by further rote transformations that could be defined algorithmically. This is typically a repetitive process that, while not technically complicated, requires extended periods of focus to do reproducibly without errors. I developed this program because I wanted to conserve this cognitive effort for data interpretation and further experiments. 

## How to Use


Start by clicking "Open" in the top left corner. Select a single file to analyze. The sweeps should appear in the plot. The right and left arrows next to the "Open" button adjust which sweep is displayed. You can drag the green cursors to desired positions to define your analysis time range, very similar to WinWCP. You can also define them in the "Range 1 Start (ms)" and similar fields under "Analysis Settings". Note that you can check "Use Dual Analysis" to extract data from two regions in one output. Below, adjust "Plot Settings" for your desired analysis. This program includes the same four peak analysis modes (absolute, positive, negative, and peak-peak) available in WinWCP. The peak mode can be adjusted in the corresponding drop-down menu in the main window.


<img src="images/mainwindow.PNG" alt="main_window" width="1000"/>


 If you'd like to preview the output plot, click "Generate Analysis Plot". From there, you can export a CSV file with the analyzed data. If you only want the data without seeing the plot first, just click "Export Analysis Data" at the bottom of the window. 


This program makes it possible to analyze several electrophysiology files with the same analysis parameters. To use this feature, start by setting all of the desired parameters as you would do for a single file. The "Batch Analyze" button is under the "Analysis" menu at the top. A new window will appear which will prompt you to select files for analysis. Click "Start Analysis", then "View Results". A new window will appear that plots the analysis results. From this window, you can export individual CSV files for each analyzed file. These can be directly imported into Graphpad Prism. 


<img src="images/batch_analysis.PNG" alt="batch_analysis" width="300"/>


<img src="images/ba_results.PNG" alt="batchresultswindow" width="750"/>

### IV Analysis

If you are doing I-V analyses, the program allows you to create summary IV curves from batch analyses. When Current and Voltage are chosen as the Plot Settings, the Batch Analysis window will have an option to "Export IV Summary". This will output a single CSV that contains the voltage set from your first analyzed file, rounded to the nearest integer, in the first column. All subsequent columns will contain the analyzed current data from all sweeps from all input files. 

**IMPORTANT FOR SUMMARY IV: the user is responsible for their own data inputs; input files that use different voltage sets will yield erroneous results. Also note that voltages are rounded to the nearest integer. If the voltage sweeps are leaky, unstable, or otherwise inconsistent, the Summary CSV will likely be missing data points.**

You also have the option to generate a current density IV curve. Click the "Current Density IV" button in the Batch Analysis window. You will be prompted to enter Cslow values for all files. Then, a new window will appear that plots the current densities against voltages. 


<img src="images/cd_cslow.PNG" alt="Cslows" width="500"/>


<img src="images/cd_results.PNG" alt="cdresultswindow" width="800"/>

Similarly to the initial batch analysis, you have the option to export individual CSV files for every analyzed file, as well as a single Summary IV that follows the same format described above. The only difference is that these outputs contain current densities, rather than raw currents. All output CSVs are designed to be easily imported into Graphpad Prism. 


<img src="images/prism_import.PNG" alt="prismimport" width="300"/>


<img src="images/prism_cd_summary.PNG" alt="prism_cd_import" width="1100"/>

### Time-course Analysis

The workflow for other analyses, such as Time vs Current, proceeds in a very similar manner. For such time-course analyses, it is sometimes desirable to extract data from more than one analysis range per sweep. To this end, the "Use Dual Analysis" box enables the user to define a second analysis range.


<img src="images/dual_analysis_main.PNG" alt="dual_range_main" width="1100"/>


This enables the user to quickly plot both analysis ranges against the sweep times. The user can also output a CSV containing this data, ready for import into downstream analysis procedures.


<img src="images/dual_analysis_plot.PNG" alt="dual_analysis_plot" width="800"/>

### Background Subtraction

You can define a region from any current trace as the background region. The average current in this region will be calculated and subtracted from all current measurements in the sweep. This process repeats for all sweeps in the file. Batch execution is available for this as well. The same background region will be used for all files in a batch (new background value calculated for each sweep in each file).

<img src="images/bg_sub_dialog.PNG" alt="bg_sub" width="550"/>

### Ramp IV

If your data uses ramp voltage protocols to measure IV relationships, you can use the "Ramp IV" option in the "Analysis" menu. First, set the cursors around the ramp.

<img src="images/ramp_iv_setup.PNG" alt="ramp_iv_setup"/>

Then you can define the voltage range you are analyzing. The script will find the closest measured voltages within your analysis range and extract the current measurements at those time points. It will do this for all sweeps or a selection of sweeps. 

<img src="images/ramp_analysis.PNG" alt="ramp_analysis"/>

### Conductance

You can also plot conductance values in the Y Axis. Conductance calculations follow the formula G = I / (V-Vrev). Calculations are skipped if (V-Vrev) < 0.1 mV. Vrev is input by the user in mV.

### Sweep Extraction

Access from the Analysis menu. Load a file in the main window first. Users can also extract the currently displayed sweep from main window. Batch extraction is available as well to extract one or multiple of the same sweeps from several files at the same time. This makes it much easier to quantify differences between sweep phenotypes in different conditions.

### Leak Subtraction

Performs P/N leak subtraction on WCP files recorded with WinWCP's leak subtraction protocol. This operation follows the same processes performed by WinWCP's native leak subtraction analysis tool. The user specifies the VHold and VTest positions, in which VHold is the set to the holding potential and VTest is positioned in the test range of the sweep. The algorithm removes passive membrane currents using voltage-scaled subtraction:

1. Baseline Correction: All voltage and current traces are zeroed by subtracting the value at the holding potential (VHold cursor position averaged over the following 20 time indices).

2. Sweep Averaging: If multiple LEAK or TEST sweeps exist per group, they are averaged after baseline correction to improve signal-to-noise ratio. A pair consisting of one TEST sweep and one LEAK sweep is the typical output from WinWCP leak-subtraction voltage protocols (one pair per leak-subtracted data point). 

3. Voltage-Based Scaling: The LEAK current is scaled by the ratio of voltage steps:
   - Measure voltage step in TEST sweep: ΔV_test = V(VTest) - V(VHold)
   - Measure voltage step in LEAK sweep: ΔV_leak = V(VTest) - V(VHold)
   - Calculate scaling factor: scale = ΔV_test / ΔV_leak

4. Subtraction: The final leak-subtracted current is calculated as:
   I_subtracted = I_test - (scale × I_leak)
   Sweep groups for which TEST voltage is equal to LEAK voltage (LEAK voltage step < 0.001 mV) will be excluded from the leak-subtracted output.

Requirements:
- WCP file with sweeps classified as LEAK/TEST in WinWCP
- At least one LEAK and one TEST sweep per group

It has been noted that some leak-subtracted outputs by WinWCP do not match the outputs of this program. This is likely due to WinWCP version differences in which a wcp file was recorded in an earlier version but analyzed in a later version. Outputs from this program match manually-performed leak subtractions in Excel using complete raw TEST and LEAK traces following the algorithm described above.

## Validation

To validate the data processing modules of this program, analyses were performed on sets of real electrophysiology data files. The outputs by this program were compared with outputs by WinWCP, both analyses using identical parameters. 

### IV Analysis

This program's data processing methods have been validated by comparing PatchBatch outputs to those of WinWCP. Both analyses used the same dataset of 12 patch-clamp recordings. Each analysis used an analysis range of 150.1 ms - 649.2 ms, with the X-axis plotting Average Voltage and the Y-axis plotting Average Current. The comparison found excellent agreement between both analysis methods. Each recording contained 11 sweeps, thus 132 data points were compared. The maximum discrepancy in the analyzed current values was 0.049694 pA. Similarly, the distinction in the measured voltage was 0.011475 mV. The distinction is due to differences in floating point arithmetic in data averaging operations in WinWCP (written in Pascal) versus Python. WinWCP uses 32-bit floating-point precision (~7 significant digits), while PatchBatch uses Python formulas with 64-bit precision (~15-16 significant digits). Because the raw data is stored as integers before being scaled to practical units, the only real difference is that 64-bit precision actually produces more accurate calculations at the expense of increased computation time, which is negligible for the intended applications of this program. These results are summarized as follows:

<img src="images/wcp_data_comparison.png" alt="data_comparison"/>

To assess the ABF functionality, the same WCP files were converted to ABF. File format conversions were performed in WinWCP. The ABF dataset was analyzed with the same parameters as the WCP dataset. The results were functionally identical to those of the WCP dataset.


<img src="images/discrepancy.png" alt="discrepancy"/>


For files analyzed in WinWCP, current densities were calculated in Graphpad Prism. A direct comparison of a Current Density vs. Voltage relationship plot produced by WinWCP vs. by PatchBatch shows that the WinWCP results are accurately reproduced by PatchBatch. In this example, the ABF dataset was used:


<img src="images/abf_data_comparison.png" alt="data_comparison"/>

For current density analysis, the following Cslow values were used:

| File ID     | Cslow |
|-------------|-------|
| 250514_001  | 34.4  |
| 250514_002  | 14.5  |
| 250514_003  | 20.5  |
| 250514_004  | 16.3  |
| 250514_005  | 18.4  |
| 250514_006  | 17.3  |
| 250514_007  | 14.4  |
| 250514_008  | 14.1  |
| 250514_009  | 18.4  |
| 250514_010  | 21.0  |
| 250514_011  | 22.2  |
| 250514_012  | 23.2  |

### Time Course and Dual Range

To validate the functionality of time course analyses, three WCP files were analyzed in WinWCP and PatchBatch. For these files, there were two analysis ranges of interest. Thus, the dual analysis range functionality was also validated. The first analysis range was 50.45 - 249.8 ms and the second range was 250.45 - 449.5 ms. The files were batch-analyzed in PatchBatch using Time (Voltage) on the X-Axis and Average (Current) on the Y-Axis. There was excellent agreement between the analysis of both programs, with a maximum discrepancy of 0.005005 pA found across all data points (n = 1278). Similarly, the Time values extracted from the WCP files in PatchBatch showed very good agreement with WinWCP, with a maximum discrepancy of 0.005 s in this data set (n = 639). 

<img src="images/time_course_validation.png" alt="time-course-validation"/>

<img src="images/time_diff_plot.png" alt="time_diff_plot_time-course" width="750"/>

Users performing time course analyses of ABF files are encouraged to ensure that accurate sweep times are returned. Some discrepancies in the sweep time array were observed during testing in .abf files that were converted from .wcp via WinWCP; this is believed to be related to the file conversion and not an inherent issue with this program's data processing or pyABF, based on similar observations of the same files loaded in pClamp. Users with .wcp files are discouraged from converting them to .abf for use in this program.

### Peak Analysis

The four peak analysis modes (absolute, positive, negative, and peak-peak) were validated using one of the same WCP files from the time course validation. The analysis range was 50.2 - 164.9 ms. The analysis plotted Peak Voltage versus Peak Current. A similar maximum discrepancy of 0.00497 pA and 0.000485 mV was found across all four peak modes (n = 1278). The discrepancies of the peak current values for all four peak modes are summarized in the following figure.

<img src="images/peak_validation.png" alt="peak-validation"/>

## License

### PatchBatch License
PatchBatch is released under the MIT License. See [LICENSE.md](LICENSE.md) for details.

### Third-Party Components
PatchBatch uses PySide6 (Qt for Python), which is licensed under LGPLv3. 
This means:
- PatchBatch itself remains MIT-licensed
- PySide6 components remain LGPLv3-licensed
- You can modify and redistribute both, subject to their respective licenses

See [LICENSES/THIRD-PARTY-NOTICES.txt](LICENSES/THIRD-PARTY-NOTICES.txt) for complete attribution and [LICENSES/LICENSE-LGPLv3.txt](LICENSES/LICENSE-LGPLv3.txt) for the full LGPLv3 terms.

### Rebuilding with Modified PySide6
Advanced users can replace or modify the PySide6 library. See [BUILD-INSTRUCTIONS.md](BUILD-INSTRUCTIONS.md) for details.