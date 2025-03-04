# RL_pytorch_shared2

## Env Setup

```
C:/Users/Shawn/AppData/Local/Programs/Python/Python310/python.exe -m pip install --upgrade pip

pip install -U autopep8

C:/Users/Shawn/AppData/Local/Programs/Python/Python310/python.exe -m venv .env

pip install matplotlib numpy pandas
```

```
NVIDIA-smi
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install gymnasium
```

## Lear RL by AI

1. DQN
2. LSTM
3. Transform(Encoder)

## TODO

1. n-step 回报的逻辑类似于我之前研究的，between-action-reward/cost. 用于降低 action 切换逻辑。
2. 基于 td_errors 优化 ReplayBuffer.update_priorities。
