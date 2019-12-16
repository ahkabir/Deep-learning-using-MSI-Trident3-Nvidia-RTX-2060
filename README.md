# Deep-learning-using-MSI-Trident3-Nvidia-RTX-2060
Setting up MSI Trident3 Gaming comupter (NVIDIA GPU RTX 2060) for Deep learning 

This guides describes how you can configure you MSI Trident 3 Gaming Desktop for performing Deep/Machine Learning work. This README is particularly useful to the users who have similar hardware and software platforms like I use in my setup.

- Hardware
  - MSI Trident 3 8SC-439US Gaming Desktop PC
  - 8th generation i7 cores
  - 16GB RAM
  - 512GB SSD
  - NVIDIA GeForce RTX 2060

- Software
  - Ubuntu 18.03

## 1. Unpacking the desktop and connecting Monitor
There are two options for connecting a monitor to the Desktop
- Use the HDMI port of the i7 Motherboard or
- Use the HDMI port of the GPU card

Talking to the customer I learnt that HDMI port of the Motherboard can not be used until you remove the NVIDIA GeForce RTX 2060 PCI Express card. But in my case that was not an option as I needed the GPU card for deep learning. Therefore, the only option left to me was to connect the monitor to the HDMI port of the GPU card.

I also researched to figure out whether or not I can use the same GPU card for the following two purposes:
- Display
- Deep Learning
It seems there is no issues doing that.


## 2. Starting the Desktop for the first time
When the Desktop was fired up it showed everything is Chinese language. I have ordered this machine from TigerDirect. Calling the customer service I found the solution to change language to my preferred language. The customer service at TigerDirect was very prompt. The solution was:
To press F3 button continuously when the system boots and then follow through the instructions for changing language

Once language problem was taken care of the Desktop booted Windows 10 home (came with the machine) successfully.

## 3. Dual Partition - adding Ubuntu in second partition
With Windows 10 home in one partition I thought of installing Ubuntu in a second partition. Although primarily I planned on using Ubuntu but being able to keep the Windows 10 home that came for price looked to me as a more practical option. So I decided to install Ubuntu 18.03 in the second partition.
I created a USB Key with Ubuntu 18.03 Desktop installer and inserted that to one of the USB ports. Changing the boot priority to USB Key as highest priority I was able to boot from USB Key and ran the Ubuntu installer. Installation of Ubuntu is well documented and is therefore beyond the scope of this README.
Once Ubuntu was installed I verified that I could boot either Windows 10 home or Ubuntu 18.03.
Also I installed NVIDIA GPU drivers for my GPU card.

## 4. Requiring larger disk space for Ubuntu
After creating the dual partition I realized half of the SSD disk space (512/2=256GB) for Ubuntu where I would do my Deep Learning was not sufficient. So I decided to wipe out the Windows 10 partition and expand the Ubuntu disk space.
To make the uninstallation easy from Ubuntu partition I installed an application called OS Uninstaller (https://help.ubuntu.com/community/OS-Uninstaller). I used this application and selected Windows 10 for uninstallation. I ended up getting a blackscreen. So I wouldn't recommend using OS-Uninstaller.

## 5. Single partition of Ubuntu 18.03
Since I got black screen I decided that I would do a fresh install of Ubuntu 18.03 and tried to boot from USB key. But due to some incompatibilities of NVIDIA driver from previous installation, I was not able to see the GUI of Ubuntu 18.03 installer. In order to solve this problem I did the following:
- Keep hitting CTRL+ALT+F3 while the system was booting
  - With this I was able to see the Ubuntu login prompt (CLI) from previous Ubuntu installation
- From the CLI promp cleaned up the NVIDIA driver installation
  - `sudo apt-get purge nvidia*`
- Remove USB Key
- Reboot
- Keep pressing DEL key to enter BIOS to check the following settings:
- - Security should be disabled
- - Fastboot should be disabled
- - Windows related things should be disabled

Save these settings and exit BIOS. Note that these BIOS settings are very important.

Finally, I was able to see the Ubuntu 18.03 installer GUI and ended up successfully installing Ubuntu 18.03

## 6. Installing NVIDIA GPU driver
First I found out the NVIDIA GPU driver version that I need to install from NVIDIA Drivers Download site (https://www.nvidia.com/Download/index.aspx?lang=en-us). After I filled the form with the make and model of my GPU card I got the driver version number. It was 440.36 at the time when I installed.
- `sudo apt-apt-repository ppa:graphics-drivers/ppa`
- `sudo apt update`
- `sudo apt install nvidia-driver-440`

The driver installation also installed some tools in addition to installing the GPU driver. Once such tool is nvidia-smi. I used this tool the following way to get to know the driver details:
- `nvidia-smi`
  - Driver Version: 440.26
  - CUDA Version: 10.2

###### 6.1 Removing the default NVIDIA driver
I checked  to see whether the default NVIDIA driver - nouveau - is loaded.
- `lscpi | grep nouveau`

In your system if it is loaded get rid of it now before rebooting the machine. You can do that in the following way:
- Edit /etc/default/grub file
- In the line where it shows GRUB_CMDLINE_LINUX_DEFAULT, add modprobe.blacklist=nouveau after the word splash. The updated line should look like the following:
- - `GRUB_CMDLINE_LINUX_DEFAULT="quiet splash modprobe.blacklist=nouveau"`
- Then update the grub:
- - `sudo update-grub`
- - - This in turn updates the /boot/grub/grub.cfg file

At this point the system was rebooted. Ubuntu successfully booted and there was no issues with GUI.

## 7. Installing CUDA
The CUDA installation in Linux is described in details in this NVIDIA Guide : https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

This REAMDE briefly describes the installation steps which are all captured from the NVIDIA Guide.

#### 7.1 `nvidia-smi`
When the NVIDIA driver was installed it also installed some tools including `nvidia-smi`. This tool was run to check to see whether NVIDIA driver is properly installed, the version of the driver etc.
- `nvidia-smi`
- - Driver Version: 440.26
- - CUDA Version: 10.2

#### 7.2 Installing CUDA toolkit
CUDA tools were installed using the following command:

- `sudo apt install nvidia-cuda-toolkit`
- - The above command also installed a NVIDIA CUDA compiler
- - `nvcc --version`
- - This command showed CUDA version of 9.1

In section 7.1 the CUDA version shown was 10.2 and nvcc showed 9.1. There is a discrepancy with the two versions. This requires some explanations.

The nvidia-smi command's CUDA version is the CUDA driver's viewpoint. Whereas, nvcc command's CUDA version is the runtime viewpoint of the CUDA toolkit.

There is a way to make sure that both versions will be the same. It is dicussed in the next section.

#### 7.3 CUDA installation

The site (https://developer.nvidia.com/cuda-downloads) allows user to answers couple of questions and provides details instructions on how to download and install CUDA. For my system it looked like the following:

- `wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb`
- `sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb`
- `sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub`
- `sudo apt-get update`
- `sudo apt-get -y install cuda`

I updated my .bashrc file to include some CUDA specific PATH and EXPORT definitions:

- `export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:${LD_LIBRARY_PATH}`
- `export PATH=/usr/local/cuda-10.2/bin:${PATH}`

Once .bashrc was updated like above my `nvidia-smi` and `nvcc --version` were showing the same CUDA version - 10.2

## 8. Installing CuDNN
In order to get CuDNN from NVIDIA website (https://developer.nvidia.com/rdp/form/cudnn-download-survey) an account is required. I had created an account for myself. With my login credentials I was able to download the following files:
- `libcudnn7-doc_7.6.5.32-1+cuda10.2_amd64.deb`
- `libcudnn7-dev_7.6.5.32-1+cuda10.2_amd64.deb`
- `libcudnn7_7.6.5.32-1+cuda10.2_amd64.deb`

After downloading these files I ran the following command to install CuDNN
- `sudo dpkg -i libcudnn7*`


#### 8.1 Verifying CuDNN installation
There are some CuDNN examples that comes with the installation. I ran the `mnistCUDNN` example to make sure my CuDNN installation is working. I did the following to run the example:
- `cp -r /usr/src/cudnn_samples_v7 ${HOME}`
- `cd ${HOME}/cudnn_samples_v7/mnistCUDNN`
- `make clean && make`

With this the build failed. Then I had to make the following changes to make the build successful.

- Edit `/usr/include/cudnn.h` file
- - change this line:
- - - `#include "driver-types.h"`
- - to:
- - - `#include <driver-types.h>`
- Rebuild

With the changes made above when I ran ./mnistCUDNN I could see this line at the end of the output log:
`Test passed!`

## 8. Building and installing Tensorflow from source for NVIDIA GPU GeForce RTX 2060
In this section I will describe how I built Tensorflow from source. The procedures involved is alreay very well document here: https://www.tensorflow.org/install/source

My guide contains the gist of what is described in the above link. If you are building Tensorflow from source then you need the build tool bazel. And bazel, in turn could be built from source. When building Tensorflow from source it is important to find out which version of bazel will work for building which version of Tensorflow. The tensorflow.org website contains some tested configurations: https://www.tensorflow.org/install/source#gpu

For my GPU card I decided to build and install the followings:
- Bazel
- - version 0.24.1
- Tensorflow
- - 1.0
#### 8.1 Building bazel from source

I followed these steps to install bazel:

- Download the zip file : Download bazel zip file, bazel-0.28.1-dist.zip, from here : https://github.com/bazelbuild/bazel/releases
- Install the prerequisites:
- - `sudo apt-get install build-essential openjdk-8-jdk python zip unzip`
- Extract zip file in some ${BAZEL_SRC} directory
- Build bazel with the following command
- - `env EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" bash ./compile.sh`
- The binary is placed in `output/bazel`
- Copy  `output/bazel` to some system directory, like, `/usr/local/bin` and make sure this is in the `PATH` variable
- Check to see whether bazel is installed properly by running the following command:
- - `bazel version`

#### 8.2 Building Tensorflow from source (GPU)
The details steps for building and installing Tensorflow are described below:

###### 8.2.1 Install pre-requisite packages
Install all the prerequisite packages:
- `sudo apt-get update`
- `sudo apt-get install python-pip python-dev`
- `sudo apt-get install python3-pip python3-dev`
- `sudo apt-get install build-essential cmake git unzip pkg-config`
- `sudo apt-get install python-numpy python-scipy python-matplotlib python-yaml`
- `sudo apt install python-dev python-pip`
- `sudo apt install python3-dev python3-pip`
- `pip install -U --user pip six numpy wheel setuptools mock 'future>=0.17.1'`
- `pip install -U --user keras_applications --no-deps`
- `pip install -U --user keras_preprocessing --no-deps`

Notes: that when you use the `-U` option with the `pip` command it may end up upgrading the pip version and the next `pip` command may fail with an error saying `pip can't import main`. To solve this problem do the following:
- `sudo python -m pip uninstall && sudo apt install python-pip --reinstall`

###### 8.2.2 Obtain Tensorflow source
- `cd ${TENSORFLOW_SRC}`
- `git clone https://github.com/tensorflow/tensorflow.git`
- `cd tensorflow`
- `git checkout -b r2.0 remotes/origin/r2.0`

###### 8.2.3 Configure Tensorflow build
- `cd ${TENSORFLOW_SRC}/tensorflow`
- `./configure`
- - Make sure you have the following settings. For brevity full option names are not provided. Please checkout the Tensorflow site for details
- - - cuda = yes
- - - jit = no
- - - RT = no
- - - ROCm = no

###### 8.2.4 Build Tensorflow pip package
- `bazel build //tensorflow/tools/pip-package:build-pip-package`
Once I executed the above command I got the following error:
- Error:
- - fatbinary fatal   : Unknown option '-bin2c-path'
- Fix: (https://github.com/tensorflow/tensorflow/issues/34429#issuecomment-557408498)
- - remove the line "--bin2c-path=%s" % bin2c.dirname, from the file `third_party/nccl/build_defs.bzl.tpl`

With the above fix I was able to build Tensorflow pip package

###### 8.2.5 Build Tensorflow .whl package under /tmp/tensorflow-pkg
- `./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg`

###### 8.2.6 Build the package (final step)
- `pip install /tmp/tensorflow_pkg/tensorflow-version-tags.whl`

This completes the Tensorflow build.


## 9. Keras
- Clone the Keras repository in ${KERAS_SRC}
- - `cd ${KERAS_SRC}`
- - `git clone https://github.com/fchollet/keras

Note that we cloned the keras repository so that we can work on the
examples under examples/ folder. There is no need to install separate
Keras package from package management system. The reason for this is
that Keras is already integrated in Tensorflow 2.0

The keras_example/ directory of this github project contains working
examples that use Keras APIs and uses Tensorflow in the backend.

The mnist_cnn.py under keras_example/ is a working application that
uses my NVIDIA GeForce RTX 2060 for training MNIST dataset. See the
keras_example/README.txt to find out the successful log from running
this application which trains MNIST dataset using GPU.














