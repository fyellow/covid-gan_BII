import os
cmd = 'pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html'
cmd = 'python3 -m ensurepip --upgrade'
os.system(cmd)