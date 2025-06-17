
# The official repository of PVPF

PVPF: Privacy Preserving and-Verifiable Personalized Federated Learning with Parameter Decoupling



## File structure description

- `main.py`: run code
- `fedselect.py`: The main implementation file contains the **core** federated learning training logic and evaluation functions
- `broadcast.py`: Realize the **weight broadcasting** and **aggregation functions** between the server and the client
- `lottery_ticket.py`: Realize the **model pruning** and **mask initialization functions** related to the lottery hypothesis
- `pflopt/`: The implementation directory related to the optimizer
  - It includes custom optimizers such as MaskLocalAltSGD
- `utils/`: Tool function Directory
  - It includes auxiliary functions such as data loading and parameter parsing

- `models.py`: Realize model initialization and initialize the models (**CNN**,  **MLP**, Resnet18) used for local training of federated learning based on system parameters.

## Parameter transmission process

- The client uses local data to train the model (use `train_personalized` function)）
- The client uploads the model parameters and mask to the server (by `add_server_weights` and `add_masks` function)
- The server aggregates the received parameters (in `div_server_weights` function)
- The server broadcasts the aggregated parameters back to the client (by `broadcast_server_to_client_initialization` function)

This implementation approach supports personalized federated learning, allowing some parameters to remain local (personalized parameters), while others are shared and aggregated globally (shared parameters).

## Environmental requirements
- Python 3.12
- PyTorch
- NumPy
- tqdm
- CUDA 12.4

## run record

- **CNN MNIST**
  ```bash
  python main.py --dataset mnist --model cnn --frac 1.0 --num_users 100  --com_rounds 100 --la_epochs 15  --prune_target 30
  ```

- **MLP MNIST**
  ```bash
  python main.py --dataset mnist --model mlp --frac 1.0 --num_users 100  --com_rounds 100 --la_epochs 15  --prune_target 30
  ```