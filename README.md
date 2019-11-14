# Deep-RL-Course-1
Projects and files created while learning from the github page here:

# Space Invaders
## Rom Download: http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html
unzip and put the roms folder under the space invaders folder
execute this command within the space invaders folder:
!python3 -m retro.import ./roms

## Installing necessary packages:
!pip install -r requirements.txt
### Mac Users:
You may receive an error related to tensorflow-gpu. Ignore it, you do not need it.

### PC Users:
tensorflow-gpu 1.14.0: if you have an Nvidia GPU you may receive an error to install CUDA.
I recommend installing 10.0 and its corresponding cnn.dll(you will receive instructions on how to download this after installing CUDA and trying to run this project again).
