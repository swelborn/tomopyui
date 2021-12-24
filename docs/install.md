# Installation

:::{note}

At the moment, this package only supports [tomopy](https://tomopy.readthedocs.io/en/latest/) and [astra-toolbox](http://www.astra-toolbox.com/docs/install.html) installed with CUDA. In the future, we will relax this requirement.  

:::

## Prerequisites

You will have to have the following set of hardware/software to run all of the current features of tomopyui:

- anaconda (to create a python environment)
- NVIDIA graphics card capable of CUDA 10.2+
- CUDA 10.2+

If you just want to check out the UI, there is also a way to run that if you do not have a CUDA-enabled graphics card. See [this section](#installing-tomopyui-without-cuda).

Below are the installation instructions given you do not have any of this installed on your computer.

### Installing conda

First, you'll need to install [anaconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) if you want conda to take up less space on your hard drive). If you are not familiar with conda, read the [Getting Started](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) page.

Open up an anaconda prompt. You should see the following:

```
(base)
```

This is your base environment. You generally don't want to mess with your base environment. However, we will install mamba and git here. Run the following:

```
conda install -c conda-forge mamba
conda install -c anaconda git
```
and type 'y' when prompted. 

### Installing [CUDA](https://en.wikipedia.org/wiki/CUDA)

This installation can be very confusing, and I hope to not confuse you further with this guide. You _might_ be able to run this software with an old GPU, but you'll have to check whether or not this GPU is compatible with CUDA 10.2 or higher. 

:::{note}

I have only tested this on Windows machines. If someone would like to write up a "for Dummys" install instructions for Linux or Mac, be my guest.

:::

To check compatibility, follow this list of instructions:
1. Find information on your GPU on [Windows 10](https://www.windowscentral.com/how-determine-graphics-card-windows-10), [linux](https://itsfoss.com/check-graphics-card-linux/), or [Mac](https://www.howtogeek.com/706679/how-to-check-which-graphics-card-gpu-your-mac-has/).
2. Check out whether or not your GPU is supported on [this page](https://developer.nvidia.com/cuda-gpus). Obviously this doesn't tell you what _version_ of CUDA you should install, because that would be convenient. Make note of your compute capability. If it is lower than 3.0 compute capability, don't bother continuing this installation. We need at least that to install [cupy](https://docs.cupy.dev/en/stable/install.html#).
3. Check to see what your latest graphics card driver version is on the [NVIDIA driver page](https://www.nvidia.com/Download/index.aspx?lang=en-us)(e.g., Version: 391.35).
4. See [Table 3](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) on the [CUDA Toolkit Docs](https://docs.nvidia.com/cuda/index.html). 
5. Check under "Toolkit Driver Version". We need at least CUDA 10.2 for this installation (as of this documentation, [cupy](https://docs.cupy.dev/en/stable/install.html#) supports drivers at or above CUDA Toolkit version 10.2). If your driver number is above the number under "Toolkit Driver Version", you should be good to forge on with this installation. 

**On Windows, you should install Microsoft Visual Studio (VS).** 
Check compatibility using the following instructions:
1. Go to the [CUDA Toolkit Download Archive](https://developer.nvidia.com/cuda-toolkit-archive)
2. Select the versioned documentation that matches what you read in [Table 3](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) and choose the installation guide of your choice (Windows, linux, Mac)
3. For example, for Windows and CUDA v10.2 you will end up at [this page](https://docs.nvidia.com/cuda/archive/10.2/cuda-installation-guide-microsoft-windows/index.html).
4. Go to the [Windows Compiler Support table](https://docs.nvidia.com/cuda/archive/10.2/cuda-installation-guide-microsoft-windows/index.html#system-requirements) under System Requirements, and find which MSVC Version you need.

If you already have these versions of VS installed on your computer, you can probably proceed without installing it. If you don't have Visual Studio installed, you will need to download and install a version which corresponds to the Compiler Support table associated with the version of CUDA you want to install.

Microsoft releases the latest version of VS (i.e., VS2022) for public download. In that version, you have the option of downloading and installing old versions of VS for compatibility reasons. Unfortunately, because of where these older versions are installed on your computer, the CUDA installer will likely not recognize VS. 

To avoid this, you should install an old version using a direct download on the [archived versions page](https://visualstudio.microsoft.com/vs/older-downloads/). You'll have to make an account to do this. Once you do, select your version and download. When you have the option, you should only need the "Build Tools" component of the download.

**After installing VS, continue to downloading and installing CUDA:**
1. Go back to the [CUDA Toolkit Download Archive](https://developer.nvidia.com/cuda-toolkit-archive) 
2. Select the CUDA version that matches what you read in [Table 3](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions).
3. Follow the install instructions.
4. If all goes well, you should not receive any warning from the installer about finding Visual Studio. If it did warn you, shut down the installer and you will likely have to edit your PATH environment variable to include the VS executable and reinstall. 
    - [DuckDuckGo](https://duckduckgo.com/) editing path variables if you don't know how.
    - VS MSVC path should be something like this C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64
    - Find that path on your computer, and put it in your PATH environment variable.

If all went well, you should be able to open an anaconda prompt and type the following command:

```
(base) nvcc
```

Which will output something like this:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Mon_Oct_11_22:11:21_Pacific_Daylight_Time_2021
Cuda compilation tools, release 11.4, V11.4.152
Build cuda_11.4.r11.4/compiler.30521435_0
```

If this does not work, you need to set your PATH to include nvcc. That look something like this:
- C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\bin\

Retry:

```
(base) nvcc
```

to see if your computer can recognize the nvcc command. 

## Installing tomopyui

First, navigate to where you want to install tomopyui:

```
cd your-install-directory-name
```

Clone the github repository:

```
git clone https://github.com/samwelborn/tomopyui.git
```

Navigate on into the tomopyui directory:

```
cd tomopyui
```

Run the following command:

```
conda env create -f environment.yml
```

This will install a new environment called tomopyui. To activate this environment:

```
conda activate tomopyui
```

Once you do that, you should see (tomopyui) instead of (base) in your anaconda prompt. Your last step is to install tomopyui. From the main directory (the one that has setup.py in it), run:

```
pip install .
```

## Installing tomopyui without CUDA

Without CUDA, this program is useless for aligning/reconstructing tomography data. 

If you don't have CUDA and just want to check out the ipywidgets, you can still do that using the environment.yml in the docs folder:

```
cd tomopyui
cd docs
conda env create -f environment.yml
```

Then, activate the environment:

```
conda activate tomopyui-docs
```

## Installing tomopyui for development

First create your own fork of <https://github.com/samwelborn/tomopyui>. If you are familiar with command-line git, you can do it that way. Otherwise, download [GitHub Desktop](https://desktop.github.com/) and download the tomopyui repository from there. Follow the install instructions above, then run:

```
pip install -e .
```

The {command}`-e .` flag installs the `tomopyui` folder in ["editable" mode](https://pip.pypa.io/en/stable/cli/pip_install/#editable-installs). This will let you make changes to the code in your install folder, and send them to your own `tomopyui` fork. 

:::{note}

A nice set of basic instructions for development is on [Development Guide - tomopy](https://tomopy.readthedocs.io/en/latest/devguide.html) or on this [GitHub page](https://github.com/firstcontributions/first-contributions#first-contributions).

:::
