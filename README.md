# sm64 AI environment
1. Put a sm64 US rom in the env folder and rename it baserom.us.z64
2. If on windows, [set up MSYS2 and install dependencies](https://github.com/djoslin0/sm64ex-coop/wiki/Compiling-on-Windows) otherwise for linux just [install build dependencies](https://github.com/djoslin0/sm64ex-coop/wiki/Compiling-on-Linux)
3. Open MSYS2 (windows) or terminal (linux) from the repository folder and then run this command.
```
cd env && make -j16 MAX_PLAYERS=20
```
MAX_PLAYERS is the number of marios playing per environment. Even numbers are recommended otherwise hide and seek will not work. 255 is the maximum, 20 is recommended.

2. install python requirements
```
pip install -r requirements.txt
``` 
3. Run sm64_test_tag_model.py to test a pre-trained AI or sm64_env_random_action_text.py to test the environment