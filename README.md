Here is an example to perform FedCos and other baselines:

//FedAvg

python  main_fed.py --model mlp --epochs 250 --dataset fmnist --iidpart 0 --step_in_round 100 --local_bs 64 --frac 0.1 --all_clients=False --num_users 100 --num_classes 10 --lr 0.01 --gpu 0 --method 0 --summary_path 'logs/fmnist/mlp/noniid_frac(100worker-0.1frac-250e-64bs-100r-0.01lr-fedavg)



//FedAvgM (FedAvg+server momentum)

python  main_fed.py --model mlp --epochs 250 --dataset fmnist --iidpart 0 --step_in_round 100 --local_bs 64 --frac 0.1 --all_clients=False --num_users 100 --num_classes 10 --lr 0.01 --gpu 1 --method 0 --server_mom=0.5 --summary_path 'logs/fmnist/mlp/noniid_frac(100worker-0.1frac-250e-64bs-100r-0.01lr-fedavgm0.5)     



//FedOpt 

python  main_fed.py --model mlp --epochs 250 --dataset fmnist --iidpart 0 --step_in_round 100 --local_bs 64 --frac 0.1 --all_clients=False --num_users 100 --num_classes 10 --lr 0.01 --gpu 2 --method 0 --optrate=1.5 --summary_path 'logs/fmnist/mlp/noniid_frac(100worker-0.1frac-250e-64bs-100r-0.01lr-fedopt1.5)'

//FedProx 

python  main_fed.py --model mlp --epochs 250 --dataset fmnist --iidpart 0 --step_in_round 100 --local_bs 64 --frac 0.1 --all_clients=False --num_users 100 --num_classes 10 --lr 0.01 --gpu 3 --method 1  --imp0=0.1 --summary_path 'logs/fmnist/mlp/noniid_frac(100worker-0.1frac-250e-64bs-100r-0.01lr-fedProx0.1)'

//FedCos

python  main_fed.py --model mlp --epochs 250 --dataset fmnist --iidpart 0 --step_in_round 100 --local_bs 64 --frac 0.1 --all_clients=False --num_users 100 --num_classes 10 --lr 0.01 --gpu 4 --method 7  --imp0=0.01 --summary_path 'logs/fmnist/mlp/noniid_frac(100worker-0.1frac-250e-64bs-100r-0.01lr-fedcos0.01)'



//fedopt+fedcos
python main_fed.py main_fed.py --model resnet18 --epochs 300 --dataset cifar100 --iidpart 0.1 --step_in_round 200 --local_bs 64 --frac 1 --num_users 5 --num_classes 100 --lr 0.1 --gpu 7 --method 7 --imp0=0.02 --optrate=1.5 --summary_path 'logs/cifar100/resnet18-2/noniid0.1(5worker-1frac-300e-64bs-200r-0.1lr-fedcos0.02+fedopt1.5)'



cite information:

```
@ARTICLE{9933784,
  author={Zhang, Hao and Wu, Tingting and Cheng, Siyao and Liu, Jie},
  journal={IEEE Internet of Things Journal}, 
  title={FedCos: A Scene-adaptive Enhancement for Federated Learning}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/JIOT.2022.3218315}}
```
