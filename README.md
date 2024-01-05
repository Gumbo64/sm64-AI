# sm64 AI environment
1. Compile in the same way as the normal sm64ex-coop repo
(Put a sm64 US rom in the env folder, name it baserom.us.z64, then cd into /env and then run "make -j16")
```
cd env && make -j16
```
2. install python requirements
```
pip install -r requirements.txt
``` 
3. Run sm64_test_tag_model.py to test a pre-trained AI or sm64_env_random_action_text.py to test the environment