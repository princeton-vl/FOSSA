
<h1 align="center">Zero-Shot Depth from Defocus</h1>
  <p align="center">
    <a href="https://zuoym15.github.io/"><strong>Yiming Zuo*</strong></a>
    ·
    <a href="https://hermera.github.io/"><strong>Hongyu Wen*</strong></a>
    ·
    <a href="https://www.linkedin.com/in/venkat-subramanian5/"><strong>Venkat Subramanian*</strong></a>
    ·
    <a href="https://patrickchen.me/"><strong>Patrick Chen</strong></a>
    ·
    <a href="https://kkayan.com/"><strong>Karhan Kayan</strong></a>
    ·
    <a href="http://mariobijelic.de/wordpress/"><strong>Mario Bijelic</strong></a>
    ·
    <a href="https://www.cs.princeton.edu/~fheide/"><strong>Felix Heide</strong></a>
    ·
    <a href="https://www.cs.princeton.edu/~jiadeng/"><strong>Jia Deng</strong></a> 
  </p>
  <p align="center">
    (*Equal Contribution)
  </p>
  <p align="center">
    <a href="https://pvl.cs.princeton.edu/">Princeton Vision & Learning Lab (PVL)</a>    
  </p>
</p>

<h3 align="center"><a href="https://arxiv.org/abs/2603.26658">Paper</a>  · </a><a href="https://zedd.cs.princeton.edu/">Project</a> </a></h3>

<p align="center">
  <a href="https://arxiv.org/abs/2603.26658">
    <img src="assets/teaser.png" alt="FOSSA Teaser" width="100%">
  </a>
</p>

---

## ZEDD Benchmark
Released under CC BY 4.0 License at 
- Website and test server: [https://zedd.cs.princeton.edu/](https://zedd.cs.princeton.edu/).
- Huggingface download link: [https://huggingface.co/datasets/venkatsubra/ZEDD](https://huggingface.co/datasets/venkatsubra/ZEDD).

## Roadmap
- ✅ Release FOSSA training code
- ✅ Release FOSSA evaluation code
- ✅ Release ZEDD dataset and test server


## Installation & Setup

### Step 1: Create and activate conda environment

```bash
conda create -n fossa python=3.8
conda activate fossa
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Build PowerExpPSF CUDA Extension

This is **required** for training and evaluation with synthetic defocus effects.

<details>
<summary><b>Build steps</b></summary>

```bash
cd power_exp_psf

# Build and install the extension
python setup.py build_ext --inplace

# Verify successful installation
python - <<'PY'
import torch
try:
    import power_exp_psf_cuda
    import os
    path = power_exp_psf_cuda.__file__
    if os.path.exists(path):
        print(f"SUCCESS: power_exp_psf_cuda loaded from {path}")
    else:
        print(f"ERROR: module loaded but file does not exist at {path}")
except Exception as e:
    print(f"IMPORT FAILED: {e}")
PY

cd ..

# Add power_exp_psf as a search directory for imports
export PYTHONPATH=$PWD/power_exp_psf:$PYTHONPATH
```
</details>

### Step 4: Load datasets into `dataset/datasets`

---

<details>

<summary><b>Datasets download instructions</b></summary>



##### 📦 HAMMER
Download: [HAMMER Dataset](https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation/blob/main/HAMMER.zip) prepared by [MoGe2](https://github.com/microsoft/moge).

```bash
cd dataset/datasets
wget https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation/resolve/main/HAMMER.zip
unzip HAMMER.zip
rm -f HAMMER.zip
cd ../..
```

##### 📦 DDFF-12
###### Data split

```bash
cd dataset/datasets
mkdir ddff12_val_generation
cd ddff12_val_generation
mkdir third_part
```

Then, in your browser, navigate to the [DFV Split (MS Sharepoint)](https://pennstateoffice365-my.sharepoint.com/personal/fuy34_psu_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Ffuy34%5Fpsu%5Fedu%2FDocuments%2FCVPR%5F2022%5FDFVDFF%2Fmy%5Fddff%5FtrainVal%2Eh5&parent=%2Fpersonal%2Ffuy34%5Fpsu%5Fedu%2FDocuments%2FCVPR%5F2022%5FDFVDFF&ga=1) prepared by [DFF-DFV](https://github.com/fuy34/DFV).

Click the download button. Then, copy the downloaded "my_ddff_trainVal.h5" file into dataset/datasets/ddff12_val_generation and rename it to "dfv_trainVal.h5".

##### Intrinsics matrix:

The intrinsics matrix is also [provided by DFV(.mat file)](https://github.com/fuy34/DFV/blob/main/data_preprocess/third_part/IntParamLF.mat). 

Download the "raw file" in the GitHub UI and place the downloaded IntParamLF.mat at "dataset/datasets/ddff_val_generation/third_part/".

At the end, the "dataset" directory should look like this (of which only ddff12_val_generation and HAMMER you need to create).

##### Expected format:

```text
dataset/
├── datasets/
│   ├── ddff12_val_generation/
│   │   ├── dfv_trainVal.h5
│   │   └── third_part/
│   │       └── IntParamLF.mat
│   ├── HAMMER/
│   │   └── scene2_traj1_1/
│   │   │   └── 000000/
│   │   │   │   └── depth.png
│   │   │   │   └── intrinsics.json
│   │   │   │   └── meta.json
│   │   │   └── ...
│   │   └── ...
│   │   └── .index.txt
│   └── splits/
│       └── infinigen_defocus/
│           └── val.json
├── __init__.py
├── base.py
├── ddff12_val.py
├── hammer.py
├── infinigen_defocus.py
├── uniformat.py
└── zedd.py
```


#### Datasets that are loaded from HuggingFace (no user downloading necessary)

Note: the first time that evaluation is done on these datasets will take some time for the zip file to download and get unpacked. If you are downloading the zip file manually, note that you will have to delete the outer folder created by the unzipped file to achieve the above file structure (deleting of the outer folder is done automatically in the provided code).

##### Final expected format:
```text
dataset/
├── datasets/
│   ├── ddff12_val_generation/
│   │   ├── dfv_trainVal.h5
│   │   └── third_part/
│   │       └── IntParamLF.mat
│   ├── defocus_uniformat/
│   │   ├── diode/
│   │   │   ├── diode_indoor_v2/
│   │   │   │   ├── 000000.npy
│   │   │   │   ├── 000001.npy
│   │   │   │   └── ...
│   │   │   └── diode_outdoor_v2/
│   │   │       ├── 000000.npy
│   │   │       ├── 000001.npy
│   │   │       └── ...
│   │   └── ibims/
│   │       ├── 000000.npy
│   │       ├── 000001.npy
│   │       └── ...
│   ├── HAMMER/
│   │   ├── scene2_traj1_1/
│   │   │   ├── 000000/
│   │   │   │   ├── depth.png
│   │   │   │   ├── intrinsics.json
│   │   │   │   └── meta.json
│   │   │   └── ...
│   │   ├── ...
│   │   └── .index.txt
│   ├── infinigen_defocus/
│   │   ├── 1a4897de_1/
│   │   │   ├── cam_all_in_focus.npz
│   │   │   ├── cam_ap_1.40_fd_0.80.npz
│   │   │   ├── ...
│   │   │   ├── depth.npy
│   │   │   ├── image_all_in_focus.png
│   │   │   └── image_ap_1.40_fd_0.80.png
│   │   └── ...
│   ├── ZEDD/
│   │   ├── test/
│   │   │   ├── test_0001/
│   │   │   │   ├── focus_stack/
│   │   │   │   │   ├── img_run_1_motor_6D3E_aperture_F1.4.jpg
│   │   │   │   │   ├── img_run_1_motor_6D3E_aperture_F2.0.jpg
│   │   │   │   │   └── ...
│   │   │   │   └── gt/
│   │   │   │       └── K.txt
│   │   │   └── ...
│   │   └── val/
│   │       ├── val_0001/
│   │       │   ├── focus_stack/
│   │       │   │   ├── img_run_1_motor_6D3E_aperture_F1.4.jpg
│   │       │   │   ├── img_run_1_motor_6D3E_aperture_F2.0.jpg
│   │       │   │   └── ...
│   │       │   └── gt/
│   │       │       ├── depth_vis.jpg
│   │       │       ├── depth.npy
│   │       │       ├── K.txt
│   │       │       └── overlay.jpg
│   │       └── ...
│   └── splits/
│       └── infinigen_defocus/
│           └── val.json
├── __init__.py
├── base.py
├── ddff12_val.py
├── hammer.py
├── infinigen_defocus.py
├── uniformat.py
└── zedd.py
```

#### 📦 ZEDD
Dataset: [ZEDD on Hugging Face](https://huggingface.co/datasets/venkatsubra/ZEDD)

---

#### 📦 Infinigen Defocus
Dataset: [Infinigen Defocus on Hugging Face](https://huggingface.co/datasets/venkatsubra/InfinigenDefocus)

---

#### 📦 iBims-1 and DIODE
Dataset: [Preprocessed (depth holes filled) on Hugging Face](https://huggingface.co/datasets/venkatsubra/Uniformat)


---

</details>


## Validation Quickstart

### Running Validation

The easiest way to validate is using the distributed validation script:

```bash
bash dist_val.sh --encoder [VITS/VITB] --resumed_from [NAME OF PARAMETERS] --val_loader_config_choice [VAL_CONFIG_CHOICE]
```

### Available Validation Configurations

See `config/validation_configs.py` for all predefined validation setups:

### Model Loading Options

**Option 1: Load from HuggingFace Hub** (recommended)
```python
resumed_from='model_name'  # automatically pull from venkatsubra/model_name
```

**Option 2: Load from local path**
```python
resumed_from='/path/to/model.pth'
```
link: [ViT-B](https://huggingface.co/venkatsubra/fossa-vitb), [ViT-S](https://huggingface.co/venkatsubra/fossa-vits)

## Reproducing Numbers in the Paper

---


<details>

<summary><b>🔹 ViT-S</b></summary>

#### Table 2

##### ZEDD 
**Note: The results below are on the validation split, so do not match the numbers in Table 2 on the test split**
```bash
bash dist_val.sh --encoder vits --resumed_from fossa-vits \
  --val_loader_config_choice zedd_F2_8_fixed_fd_0_2_4_6_8
```

| D1.05 | D1.15 | D1.25 | abs_rel |
|------:|------:|------:|--------:|
| 0.4450 | 0.7866 | 0.8858 | 0.0985 |

##### Infinigen
```bash
bash dist_val.sh --encoder vits --resumed_from fossa-vits \
  --val_loader_config_choice infinigen_defocus_F1_4_fixed_fd_0_8,1_7,3_0,4_7,8_0
```

| D1.05 | D1.15 | D1.25 | abs_rel |
|------:|------:|------:|--------:|
| 0.5201 | 0.8635 | 0.9400 | 0.0847 |

---

#### Table 3

##### iBims-1
```bash
bash dist_val.sh --encoder vits --resumed_from fossa-vits \
  --val_loader_config_choice ibims_F1_4_adaptive_fd
```

| D1.05 | D1.15 | D1.25 | abs_rel |
|------:|------:|------:|--------:|
| 0.5193 | 0.8502 | 0.9540 | 0.0745 |

##### DIODE
```bash
bash dist_val.sh --encoder vits --resumed_from fossa-vits \
  --val_loader_config_choice diode_F1_4_adaptive_fd
```

| D1.05 | D1.15 | D1.25 | abs_rel |
|------:|------:|------:|--------:|
| 0.4105 | 0.6649 | 0.7661 | 0.1778 |

##### HAMMER
```bash
bash dist_val.sh --encoder vits --resumed_from fossa-vits \
  --val_loader_config_choice hammer_F1_4_adaptive_fd
```

| D1.05 | D1.15 | D1.25 | abs_rel |
|------:|------:|------:|--------:|
| 0.6006 | 0.9889 | 0.9987 | 0.0440 |

---

#### Table 4

##### DDFF12 (Base Model)
```bash
bash dist_val.sh --encoder vits --resumed_from fossa-vits \
  --val_loader_config_choice ddff12_val
```

|    MSE |   RMSE | AbsRel |  SqRel |     D1 |     D2 |     D3 |
| -----: | -----: | -----: | -----: | -----: | -----: | -----: |
| 0.0015 | 0.0352 | 0.2676 | 0.0119 | 0.3462 | 0.8119 | 0.9544 |


##### DDFF12 (Finetuned)
```bash
bash dist_val.sh --encoder vits --resumed_from fossa-vits-ddff-finetuned \
  --val_loader_config_choice ddff12_val
```

|    MSE |   RMSE | AbsRel |  SqRel |     D1 |     D2 |     D3 |
| -----: | -----: | -----: | -----: | -----: | -----: | -----: |
| 0.0004 | 0.0183 | 0.1076 | 0.0045 | 0.9363 | 0.9829 | 0.9908 |


---

</details>

<details>

<summary><b>🔹 ViT-B</b></summary>

#### Table 2

##### ZEDD
**Note: The results below are on the validation split, so do not match the numbers in Table 2 on the test split**
```bash
bash dist_val.sh --encoder vitb --resumed_from fossa-vitb \
  --val_loader_config_choice zedd_F2_8_fixed_fd_0_2_4_6_8
```

| D1.05 | D1.15 | D1.25 | abs_rel |
|------:|------:|------:|--------:|
| 0.4317 | 0.8101 | 0.9194 | 0.0957 |

##### Infinigen
```bash
bash dist_val.sh --encoder vitb --resumed_from fossa-vitb \
  --val_loader_config_choice infinigen_defocus_F1_4_fixed_fd_0_8,1_7,3_0,4_7,8_0
```
readme
| D1.05 | D1.15 | D1.25 | abs_rel |
|------:|------:|------:|--------:|
| 0.4199 | 0.8199 | 0.9355 | 0.0908 |

---

#### Table 3

##### iBims-1
```bash
bash dist_val.sh --encoder vitb --resumed_from fossa-vitb \
  --val_loader_config_choice ibims_F1_4_adaptive_fd
```

| D1.05 | D1.15 | D1.25 | abs_rel |
|------:|------:|------:|--------:|
| 0.5548 | 0.8719 | 0.9633 | 0.0701 |

##### DIODE 
```bash
bash dist_val.sh --encoder vitb --resumed_from fossa-vitb \
  --val_loader_config_choice diode_F1_4_adaptive_fd
```

| D1.05 | D1.15 | D1.25 | abs_rel |
|------:|------:|------:|--------:|
| 0.4127 | 0.6692 | 0.7786 | 0.1601 |

##### HAMMER
```bash
bash dist_val.sh --encoder vitb --resumed_from fossa-vitb \
  --val_loader_config_choice hammer_F1_4_adaptive_fd
```

| D1.05 | D1.15 | D1.25 | abs_rel |
|------:|------:|------:|--------:|
| 0.9377 | 0.9974 | 0.9993 | 0.0172 |

---

#### Table 4

##### DDFF12 (Base Model)
```bash
bash dist_val.sh --encoder vitb --resumed_from fossa-vitb \
  --val_loader_config_choice ddff12_val
```


|    MSE |   RMSE | AbsRel |  SqRel |     D1 |     D2 |     D3 |
| -----: | -----: | -----: | -----: | -----: | -----: | -----: |
| 0.0013 | 0.0324 | 0.2105 | 0.0107 | 0.6075 | 0.9206 | 0.9679 |


##### DDFF12 (Finetuned)
```bash
bash dist_val.sh --encoder vitb --resumed_from fossa-vitb-ddff-finetuned \
  --val_loader_config_choice ddff12_val
```

|    MSE |   RMSE | AbsRel |  SqRel |     D1 |     D2 |     D3 |
| -----: | -----: | -----: | -----: | -----: | -----: | -----: |
| 0.0003 | 0.0148 | 0.1088 | 0.0025 | 0.9322 | 0.9866 | 0.9939 |

</details>

</details>

## Submitting to ZEDD Test Server

For ZEDD test set, save model outputs in the following format:

- A single `.zip` file containing exactly **50** `.npy` files at the root level (no subdirectories)
- Files must be named `zedd_output_0001.npy` through `zedd_output_0050.npy`
- Each `.npy` file must be a **2-D float array** of shape **(H=1216, W=1824)** — no channel dimension
- All values must be **finite** (no NaN or Inf)

Please run the following command to check the file format before submitting to the server:
```bash
python zedd_test/zedd_check_format.py --zip [YOUR_ZIP_FILE]
```

Here is an example to compile the zip file for FOSSA ViT-S:

```bash
bash dist_test.sh --encoder=vits --resumed_from fossa-vits --val_loader_config_choice zedd_test_F2_8_fixed_fd_0_2_4_6_8 --experiment_name=FOSSA --zedd_test_output_dir=zedd_outputs
```

Finally, submit your zip file to [the ZEDD test server](https://zedd.cs.princeton.edu/submissions/new/).

## Training from Scratch & Finetuning on DDFF
See [Training.md](./Training.md) for details.

## Troubleshooting

<details>

<summary><b>PowerExpPSF building</b></summary>

#### ❌ Error: `nvcc` not found / CUDA extension build fails

If you see an error like: "error: [Errno 2] No such file or directory: '/usr/local/cuda-12.1/bin/nvcc'" or "nvcc not found", this means your environment does **not have a CUDA toolkit with `nvcc` available**.

#### ✅ Fix: Load a valid CUDA toolkit and set environment variables

On cluster environments, load an available CUDA module:

```bash
module avail cuda
module load cudatoolkit/12.6   # or closest version to your PyTorch CUDA
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

Then verify:
```bash
which nvcc
nvcc --version
```

Then retry:
```bash
python setup.py build_ext --inplace
```

#### ❌ Error: ModuleNotFoundError: No module named 'power_exp_psf_cuda'

If you see an error like: "ModuleNotFoundError: No module named 'power_exp_psf_cuda'", this means your environment does not know where to search for the power_exp_psf_cuda module. 

#### ✅ Fix: Add the module to PYTHONPATH

From your project root, run:

```bash 
export PYTHONPATH=$PWD/power_exp_psf:$PYTHONPATH
```
Then retry your script.

</details>

## Citation
```bibtex
@article{ZeroShotDepthFromDefocus,
  author  = {Zuo, Yiming and Wen, Hongyu and Subramanian, Venkat and Chen, Patrick and Kayan, Karhan and Bijelic, Mario and Heide, Felix and Deng, Jia},
  title   = {Zero-Shot Depth from Defocus},
  journal = {arXiv preprint arXiv:2603.26658},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.26658}
}
```

## Acknowledgments
This codebase is partially based on [Depth Anything v2](https://github.com/DepthAnything/Depth-Anything-V2), [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything), [DFF-DFV](https://github.com/fuy34/DFV), and [Unsupervised Depth from Focus](https://github.com/shirgur/UnsupervisedDepthFromFocus).
