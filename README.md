Go to https://github.com/ck852/patchbatch?tab=readme-ov-file#patchbatch---electrophysiology-data-analysis-tool for the most up-to-date README

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

### Prerequisites: Installing Python

If you don't have Python installed:

1. **Download Python**: Visit [python.org](https://www.python.org/downloads/) and download Python 3.9 or newer
2. **Install Python**: Run the installer and **check "Add Python to PATH"** during installation (Windows) or use the default settings (Mac/Linux)
3. **Verify Installation**: Open a terminal/command prompt and type `python --version` to confirm installation

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

If python is not a recognized command, try replacing it with `python3`.

If you install from PyPI, you will always start the program by opening a terminal in the same directory where you installed it and simply enter `patchbatch`. 



## Option 3: Install from Source (Developers)
```bash
git clone https://github.com/ck852/patchbatch.git
cd patchbatch
pip install -r requirements.txt
python src/data_analysis_gui/main.py
```

## Contributing
Found a bug or want to contribute? Please open an issue at:
https://github.com/ck852/patchbatch/issues


**Note that this software is currently in beta. There may be bugs and irregularities that were not seen in development. You are encouraged to validate your initial results against a trusted data analysis program such as WinWCP to ensure that all data processing modules are functioning as expected. See the Validation section at the end of this README for more information.**


## Introduction

Welcome to PatchBatch! The purpose of this program is to streamline electrophysiology data analysis workflows by enabling batch-analysis of data files that share the same analysis parameters. I developed this after growing impatient with the long, tedious workflows that require defining the same parameters repeatedly for every file, followed by further rote transformations that could be defined algorithmically. This is typically a repetitive process that, while not technically complicated, requires extended periods of focus to do reproducibly without errors. I developed this program because I wanted to conserve this cognitive effort for data interpretation and further experiments. 

## How to Use


Start by clicking "Open" in the top left corner. Select a single file to analyze. The sweeps should appear in the plot. The right and left arrows next to the "Open" button adjust which sweep is displayed. You can drag the green cursors to desired positions to define your analysis time range, very similar to WinWCP. You can also define them in the "Range 1 Start (ms)" and similar fields under "Analysis Settings". Note that you can check "Use Dual Analysis" to extract data from two regions in one output. Below, adjust "Plot Settings" for your desired analysis. This program includes the same four peak analysis modes (absolute, positive, negative, and peak-peak) available in WinWCP. The peak mode can be adjusted in the corresponding drop-down menu in the main window. **All peak modes have aligned with WinWCP in preliminary tests, but final validation with figures is still in progress.**


<img src="images/mainwindow.PNG" alt="main_window" width="1000"/>


 If you'd like to preview the output plot, click "Generate Analysis Plot". From there, you can export a CSV file with the analyzed data. If you only want the data without seeing the plot first, just click "Export Analysis Data" at the bottom of the window. 


This program makes it possible to analyze several electrophysiology files with the same analysis parameters. To use this feature, start by setting all of the desired parameters as you would do for a single file. It is optional to open a single file in the main window first - if you know all of your parameters, you can skip loading the single file and go straight to batch analysis. The "Batch Analyze" button is under the "Analysis" menu at the top. A new window will appear which will prompt you to select files for analysis. Click "Start Analysis", then "View Results". A new window will appear that plots the analysis results. From this window, you can export individual CSV files for each analyzed file. These can be directly imported into Graphpad Prism. 


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

### Background Subtraction **BETA**

You can define a region from any current trace as the background region. The average current in this region will be calculated and subtracted from all current measurements in the sweep. This process repeats for all sweeps in the file.

<img src="images/bg_sub_dialog.PNG" alt="bg_sub" width="550"/>

### Ramp IV **BETA**

If your data uses ramp voltage protocols to measure IV relationships, you can use the "Ramp IV" option in the "Analysis" menu. First, set the cursors around the ramp.

<img src="images/ramp_iv_setup.PNG" alt="ramp_iv_setup"/>

Then you can define the voltage range you are analyzing. The script will find the closest measured voltages within your analysis range and extract the current measurements at those time points. It will do this for all sweeps or a selection of sweeps. **This has not been quantitatively tested yet and I highly advise validating against a known data set first.**

<img src="images/ramp_analysis.PNG" alt="ramp_analysis"/>

Better screenshot pending

### Dose-Response **BETA**

Click the "Add Range" buttons then click the plot to add them. This is meant for time vs current output files by the main window. The multi-file analysis lets you make a Prism-ready summary dataset for several files that have the same concentration range. Has not been extensively validated but matches excel-calculated results in tests.

### Sweep Extraction **BETA**

Access from the Analysis menu. Load a file in the main window first. Has not been extensively validated but matches Clampfit results in tests.

## Validation

To validate the data processing modules of this program, analyses were performed on sets of real electrophysiology data files. The outputs by this program were compared with outputs by WinWCP, both analyses using identical parameters. 

### IV Analysis

This program's data processing methods have been validated by comparing PatchBatch outputs to those of WinWCP. Both analyses used the same dataset of 12 patch-clamp recordings. Each analysis used an analysis range of 150.1 ms - 649.2 ms, with the X-axis plotting Average Voltage and the Y-axis plotting Average Current. The input abf files are available in the file repository. The comparison found excellent agreement between both analysis methods. Each recording contained 11 sweeps, thus 132 data points were compared. The maximum discrepancy in the analyzed current values was 0.049694 pA. Similarly, the distinction in the measured voltage was 0.011475 mV. The distinction is due to differences in floating point arithmetic in data averaging operations in WinWCP (written in Pascal) versus Python. WinWCP uses 32-bit floating-point precision (~7 significant digits), while PatchBatch uses Python formulas with 64-bit precision (~15-16 significant digits). Because the raw data is stored losslessly as integers before being scaled to practical units, the only real difference is that 64-bit precision actually produces more accurate calculations at the expense of increased computation time, which is negligible for the intended applications of this program. These results are summarized as follows:

<img src="images/wcp_data_comparison.png" alt="data_comparison" width="1000"/>

To assess the ABF functionality, the same WCP files were converted to ABF. File format conversions were performed in WinWCP. The ABF dataset was analyzed with the same parameters as the WCP dataset. The results were nearly identical to those of the WCP dataset.


<img src="images/discrepancy.png" alt="discrepancy" width="1000"/>

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


For files analyzed in WinWCP, current densities were calculated in Graphpad Prism. A direct comparison of a Current Density vs. Voltage relationship plot produced by WinWCP vs. by PatchBatch shows that the WinWCP results are accurately reproduced by PatchBatch. In this example, the ABF dataset was used:


<img src="images/abf_data_comparison.png" alt="data_comparison" width="750"/>