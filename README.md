# **Livereload Dockerize PyTorch on MacOS**
Ref. https://pytorch.org/get-started/locally/#macos-version

## **Quickstart**

Normally, PyTorch on MacOS doesn't support GPU yet. In case you have `AMD Radeon` in MacOS, you can use [RadeonOpenComputing(ROCm)](https://github.com/RadeonOpenCompute) to interact with your GPU but you need to ensure that you have `/dev/kfd` drive on your machine :

```bash
$ ls -lF /dev/kfd
```

If you have GPU, you can run PyTorch via `docker-compose` below:

```bash
$ docker-compose -f docker-compose.amd-gpu.yml up --build -d
```

But in case you have not :

```bash
$ docker-compose -f docker-compose.mac.yml up --build -d
```

To verify PyTorch in docker-compose service :

```bash
$ docker exec -it pytorch-mac python tests/test.py

# You should see the result looks like below:
tensor([[0.9933, 0.1517, 0.5074],
        [0.8188, 0.7017, 0.3469],
        [0.0937, 0.7780, 0.8762],
        [0.9044, 0.9377, 0.0779],
        [0.2791, 0.0870, 0.4692]])
```

Now you are ready to develop ML in `./mac/ml` or  `./mac/tests` upon your define in `./mac/server.py` :

```python
import falcon

# directory name
import tests
import ml

# ...
```

**Note** : I'm not support **RadeonOpenComputing(ROCm)** for now, you can learn more detail from https://github.com/RadeonOpenCompute.

## **Dockerize PyTorch**

This container uses for serving `MLaaS` for `PyTorch` by `Falcon`, I wrapped it with `gunicorn` to support live reloading on port number `8000` (container's port).

### Requirements
Ref. [requirements.txt](./mac/requirements.txt)

```text
requests
gunicorn
falcon==2.0
```

### **Project Directory Structure**

```
pytorch
|- mac
|  |- tests
|  |  `- YOUR_TEST_FILE.py
|  |- ml
|  |  `- YOUR_ML_FILE.py
|  |- docker-entrypoint.sh
|  |- mac.Dockerfile
|  |- requirements.txt
|  `- server.py
`- docker-compose.mac.yml
```

### **Usage**

Start docker-compose service :

```bash
# Start service
$ docker-compose -f docker-compose.mac.yml up --build -d
# Check logs
$ docker logs -f pytorch-mac
[...] [6] [INFO] Starting gunicorn 20.0.4
[...] [6] [INFO] Listening at: http://0.0.0.0:8000 (6)
[...] [6] [INFO] Using worker: sync
[...] [9] [INFO] Booting worker with pid: 9
```

Now you can `revise`/`add`/`delete` file in `./mac/`, it will re-compile any files which are modified.

Example original content in `./mac/tests/test.py` :

```python
from __future__ import print_function
import torch
x = torch.rand(5, 3)
print(x)
```

Run script :

```bash
$ docker exec -it pytorch-mac python tests/test.py
# Output:
#
# tensor([[0.5643, 0.0480, 0.4504],
#         [0.4556, 0.0260, 0.6441],
#         [0.6649, 0.0913, 0.8559],
#         [0.1680, 0.3213, 0.6010],
#         [0.2019, 0.7981, 0.8575]])
```

Let's transpose `x` :

```python
from __future__ import print_function
import torch
x = torch.rand(5, 3)
print(x)
print(x.T) # <--------- transpose
```

then press save :

```bash
# Console updated
.
.
.
[...] [27] [INFO] Worker reloading: /app/server.py modified
[...] [27] [INFO] Worker exiting (pid: 27)
[...] [29] [INFO] Booting worker with pid: 29
```

Run script again :

```bash
$ docker exec -it pytorch-mac python tests/test.py
# Output:
#
# tensor([[0.0489, 0.5667, 0.4823],
#         [0.5269, 0.9376, 0.9321],
#         [0.9060, 0.8768, 0.9934],
#         [0.7177, 0.8271, 0.9015],
#         [0.6396, 0.8067, 0.3173]])
# tensor([[0.0489, 0.5269, 0.9060, 0.7177, 0.6396],
#         [0.5667, 0.9376, 0.8768, 0.8271, 0.8067],
#         [0.4823, 0.9321, 0.9934, 0.9015, 0.3173]])
```

## **Example**

### **Text Sentiment with NGRAMS**

Ref. https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

From the PyTorch's tutorial, I created `./ml/text-classification` directory to learn/try the tutorial. I split it out to 2-parts, train and test(predict) script. 

For train model :

```bash
$ docker exec -it pytorch-mac python ml/text-classification/tc_train.py
ag_news_csv.tar.gz: 11.8MB [00:01, 5.95MB/s]
120000lines [00:08, 14785.21lines/s]
120000lines [00:14, 8357.50lines/s]
7600lines [00:00, 8653.12lines/s]
Epoch: 1  | time in 0 minutes, 19 seconds
	Loss: 0.0259(train)	|	Acc: 85.0%(train)
	Loss: 0.0001(valid)	|	Acc: 88.9%(valid)
Epoch: 2  | time in 0 minutes, 21 seconds
	Loss: 0.0118(train)	|	Acc: 93.7%(train)
	Loss: 0.0001(valid)	|	Acc: 90.7%(valid)
Epoch: 3  | time in 0 minutes, 20 seconds
	Loss: 0.0068(train)	|	Acc: 96.4%(train)
	Loss: 0.0001(valid)	|	Acc: 90.9%(valid)
Epoch: 4  | time in 0 minutes, 19 seconds
	Loss: 0.0038(train)	|	Acc: 98.1%(train)
	Loss: 0.0001(valid)	|	Acc: 90.8%(valid)
Epoch: 5  | time in 0 minutes, 20 seconds
	Loss: 0.0022(train)	|	Acc: 99.1%(train)
	Loss: 0.0001(valid)	|	Acc: 90.7%(valid)
Checking the results of test dataset...
	Loss: 0.0003(test)	|	Acc: 89.2%(test)
Saved model to -> ./ml/text-classification/tc-model.pt
```

test prediction :

```bash
$ docker exec -it pytorch-mac python ml/text-classification/tc_test.py
120000lines [00:08, 13920.01lines/s]
120000lines [00:15, 7926.40lines/s]
7600lines [00:01, 7348.76lines/s]
This is a Sports news
```

## **License**

MIT