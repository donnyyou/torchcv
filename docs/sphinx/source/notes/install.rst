PyTorchCV Installation
================

Requirements
^^^^^^^^^^^^^^^^^^^^^
This framework is developed with PyTorch, keeping up with the newest version. All the environments you need to install are listed below.
::

    python3
    cython
    numpy
    cffi
    opencv-python
    scipy
    easydict
    matplotlib
    Pillow
    pyyaml
    visdom
    bs4
    html5lib
    ninja
    torch
    torchvision


Simple Installation
^^^^^^^^^^^^^^^^^^^^^
Cuda and Nvidia
----------------
Before installing PyTorch, we need to set up a proper environment with the following steps:

1. Check if your computer has an NVIDIA graphics card and install the GPU version of PyTorch in order to take
   advantages of its powerful capability of computation acceleration, or, just install CPU version if not so.
   To be more specific, the CUDA Computing Capability of your graphics card that you can
   check on `NVIDIA official website <https://developer.nvidia.com/cuda-gpus/>`_ should not be less than 3.0.

2. Install the Python environment. Anaconda is recommended. It is an open-source release version of Python that
   provides a full environment for scientific computation including common libraries such as NumPy and SciPy, or
   you can choose your favorite ones of course.

   * You can choose to add the directory of Anaconda into the PATH (though not recommended by the installation
     wizard). It enables you to call all Anaconda commands under command line or Powershell directly. You can
     always call them under the Anaconda Prompt started in the Start Menu.

3. (For GPU version installation) Install the NVIDIA graphics driver, `CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`_ and `cuDNN <https://developer.nvidia.com/cudnn>`_. You should note that:

   * We recommend you install it through the following order: 1) latest NVIDIA graphics driver 2) CUDA (without
     selecting the built-in driver when installing since the built-in ones may be out-of-date) 3) cuDNN;
   * There is a quite simple way to install drivers in Ubuntu. First click "Software & Updates" in "System Setting",
     then toggle on "Using NVIDIA binary driver" option in "Additional Drivers" and click "Apply Changes" for system
     to install NVIDIA drivers automatically, otherwise, it won't be peaceful for NVIDIA installation on Linux. You
     should disable the built-in graphics driver Nouveau and Secure Boot function of the motherboard. You can seek a
     more detailed guidance `here <https://www.linkedin.com/pulse/installing-nvidia-cuda-80-ubuntu-1604-linux-gpu-new-victor/>`_;
   * The version of CUDA Toolkit and cuDNN must agree with the requirements on TensorFlow official website which does not always require the latest version.
   * You have to copy the downloaded files of cuDNN to the installation directory of CUDA to complete cuDNN installation.


PyTorch Installation
-------------------
::

    pip install -r requirements.txt


