# Installation

## Short version

### Docker


### Installing with CUDA

Install CUDA (different operating systems have different instructions, see below for more information), navigate to the directory in which you would like to install tomopyui:

```bash
cd /place/where/you/install/github/repos
git clone https://github.com/samwelborn/tomopyui.git
```

Navigate on into the tomopyui directory:

```bash
cd tomopyui 
conda env create -f environment.yml # create without cudatoolkit
conda activate tomopyui # activate
pip install . # install
```

### Installing without CUDA

Without CUDA, you will miss out on some of the features in TomoPyUI. You can still install it by doing the following

```bash
cd tomopyui 
conda env create -f environment-nocuda.yml # create without cudatoolkit
conda activate tomopyui # activate
pip install . # install
```

### Installing dev version

Replace the `pip install .` command above with

```bash
pip install -e .
```

## Long Version

:::{warning}
The instructions below are probably out of date. They are here, in case there is useful information.
:::
You will have to have the following set of hardware/software to run all of the features of TomoPyUI:

- anaconda (to create a python environment)
- NVIDIA graphics card capable of CUDA 10.2+
- CUDA 10.2+

Below are the installation instructions given you do not have any of this installed on your computer.

### Installing conda

First, you'll need to install [anaconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) if you want conda to take up less space on your hard drive. If you are not familiar with conda, read the [Getting Started](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) page.

Open up an anaconda prompt. You should see the following:

```bash
(base)
```

This is your base environment. You generally don't want to mess with your base environment. However, we will install mamba and git here. Run the following:

```bash
conda install -c conda-forge mamba
conda install -c anaconda git
```

and type 'y' when prompted.

### Installing [CUDA](https://en.wikipedia.org/wiki/CUDA)

This installation can be very confusing, and I hope to not confuse you further with this guide. You _might_ be able to run this software with an old GPU, but you'll have to check whether or not this GPU is compatible with CUDA 10.2 or higher.

:::{note}

I have only tested this on Windows machines. If someone would like to write up install instructions for Linux or Mac, be my guest.

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

```bash
(base) nvcc
```

Which will output something like this:

```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Mon_Oct_11_22:11:21_Pacific_Daylight_Time_2021
Cuda compilation tools, release 11.4, V11.4.152
Build cuda_11.4.r11.4/compiler.30521435_0
```

If this does not work, you need to set your PATH to include nvcc. That look something like this:

- C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\bin\

Retry:

```bash
(base) nvcc
```

to see if your computer can recognize the nvcc command.

## Installing TomoPyUI

First, navigate to where you want to install TomoPyUI:

```bash
cd your-install-directory-name
```

Clone the github repository:

```bash
git clone https://github.com/samwelborn/tomopyui.git
```

:::{note}

If you don't want to download the entire repository, you can just download the environment.yml file [here](https://github.com/samwelborn/tomopyui/blob/main/environment.yml) for CUDA or [here](https://github.com/samwelborn/tomopyui/blob/main/environment-nocuda.yml) for non-CUDA. Along with setting up your environment, this should install the latest stable release of tomopyui from [PyPI](https://pypi.org/project/tomopyui/).

:::

Navigate on into the TomoPyUI directory:

```bash
cd tomopyui
```

Run the following command:

```bash
conda env create -f environment.yml
```

This will install a new environment called tomopyui. To activate this environment:

```bash
conda activate tomopyui
```

Once you do that, you should see (tomopyui) instead of (base) in your anaconda prompt. This should have installed the latest release of tomopyui from PyPI. If you want to install the latest version from the master branch, you can run:

```bash
pip install .
```

in the tomopyui directory (the one with setup.py).

## Installing TomoPyUI without CUDA

If you don't have CUDA and just want to check out the ipywidgets, you can still do that using the [environment-nocuda.yml](https://github.com/samwelborn/tomopyui/blob/main/environment-nocuda.yml) file:

```bash
cd tomopyui
conda env create -f environment-nocuda.yml
```

Then, activate the environment:

```bash
conda activate tomopyui-nocuda
```

## Installing TomoPyUI for development

First create your own fork of <https://github.com/samwelborn/tomopyui>. If you are familiar with command-line git, you can do it that way. Otherwise, download [GitHub Desktop](https://desktop.github.com/) and download the TomoPyUI repository from there. Follow the install instructions above, then run:

```bash
pip install -e .
```

The {command}`-e .` flag installs the `tomopyui` folder in ["editable" mode](https://pip.pypa.io/en/stable/cli/pip_install/#editable-installs). This will let you make changes to the code in your install folder, and send them to your own `tomopyui` fork.

:::{note}

A nice set of basic instructions for development is on [Development Guide - tomopy](https://tomopy.readthedocs.io/en/latest/devguide.html) or on this [GitHub page](https://github.com/firstcontributions/first-contributions#first-contributions).

:::
