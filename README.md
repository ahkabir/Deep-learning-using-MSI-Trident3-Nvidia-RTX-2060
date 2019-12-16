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
  - $sudo apt-get purge nvidia*
- Remove USB Key
- Reboot

Finally, I was able to see the Ubuntu 18.03 installer GUI and ended up successfully installing Ubuntu 18.03

## 6. Installing NVIDIA GPU driv
First I found out the NVIDIA GPU driver version that I need to install from NVIDIA Drivers Download site (https://www.nvidia.com/Download/index.aspx?lang=en-us). After I filled the form with the make and model of my GPU card I got the driver version number. It was 440.36 at the time when I installed.
- $ sudo apt-apt-repository ppa:graphics-drivers/ppa
- $ sudo apt update
- $ sudo apt install nvidia-driver-440
The driver installation also installed some tools in addition to installing the GPU driver. Once such tool is nvidia-smi. I used this tool the following way to get to know the driver details:
- $ nvidia-smi
  - Driver Version: 440.26
  - CUDA Version: 10.2
At this point the system was rebooted. Ubuntu successfully booted and there was no issues with GUI.










