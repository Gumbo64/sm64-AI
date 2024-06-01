
# sm64 AI environment

This fork provides an interface between Python and the game Mario 64 in order to train AI, with many additional features to make it as painless and practical as possible. 
This functionality includes
* Support for multiple game instances
* Support for up to 254 players within the same game (can touch and see each other etc)
* Can load into any level directly
* Images and several other numerical features (position, velocity, height, health etc) accessible
* Custom resolution images
* Python tool to stitch all images together in one window
* Extremely high speed, since the game executes in native C code and not an emulator
* Variety of environment types provided
* Pytorch (CleanRL) examples provided
* Supports the pettingzoo API
* Curiosity based learning

# Results

My most successful approach has been using curiosity-based learning with [EA's technique](http://arxiv.org/pdf/2103.13798), where the AI is rewarded for reaching areas that have not been visited yet. 

Below is a video of it running
[![Alt text](https://img.youtube.com/vi/UNaf_jnOZrA/0.jpg)](https://www.youtube.com/watch?v=UNaf_jnOZrA)


Comparing the coverage of the random player to the AI's displays its successful learning.
|Original map for reference|Randomly moving players|Curiosity-based AI learning|
|--|--|--|
|![enter image description here](https://github.com/Gumbo64/sm64-AI/blob/coop/map_BOB.png?raw=true =400x)|![enter image description here](https://github.com/Gumbo64/sm64-AI/blob/coop/coverage_random_agent.png?raw=true)| ![enter image description here](https://github.com/Gumbo64/sm64-AI/blob/coop/coverage_curiosityAI_agent.png?raw=true) |

# Setup 
1. Put a sm64 US rom in the env folder and rename it baserom.us.z64

2. If on windows, [set up MSYS2 and install dependencies](https://github.com/djoslin0/sm64ex-coop/wiki/Compiling-on-Windows) otherwise for linux just [install build dependencies](https://github.com/djoslin0/sm64ex-coop/wiki/Compiling-on-Linux)

3. Open MSYS2 (windows) or terminal (linux) from the repository folder and then run this command.

```

cd env && make -j16 MAX_PLAYERS=20

```

MAX_PLAYERS is the number of marios playing per environment. Even numbers are recommended otherwise hide and seek will not work. 255 is the maximum, 20 is recommended. If you want to change MAX_PLAYERS, delete the env/build/us_pc folder first (my temporary solution)

  

2. install python requirements

```

pip install -r requirements.txt

```

3. Run sm64_test_tag_model.py to test a pre-trained AI or sm64_env_random_action_text.py to test the environment

  

to change the map, go to env/mods/compass/main.lua and change gLevelValues.entryLevel to whatever level
