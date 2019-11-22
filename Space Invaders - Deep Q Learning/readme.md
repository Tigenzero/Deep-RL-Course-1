# Space Invaders - Complete
## Rom Download: http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html
unzip and put the roms folder under the space invaders folder
execute this command within the space invaders folder:
!python3 -m retro.import ./roms

## Installing necessary packages:
### Mac and Linux Users:
!pip install -r requirements.txt

### PC Users:
!pip install -r windows.txt
If you have an Nvidia GPU you may receive an error to install CUDA.
I recommend installing 10.0 and its corresponding cnn.dll(you will receive instructions on how to download this after installing CUDA and trying to run this project again).

## Memory Leak:
Due to code pulled from the tutorial, there is a memory leak during training. Be advised, training for 30 episodes requires 32GB of RAM. Training for 60 episodes requires 64GB of RAM.