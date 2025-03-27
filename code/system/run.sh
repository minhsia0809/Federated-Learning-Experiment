#!/bin/bash
for pr in 0.0 0.4; do
    python main.py -data Cifar10_alpha05_200 -nc 200 -jr 0.5 -algo FedUCBN -sca UCB -gr 499 -pr $pr -nb 10
done
for pr in 0.0 0.4; do
    python main.py -data Cifar10_alpha05_200 -nc 200 -jr 0.5 -algo FedUCBN -sca Random -gr 499 -pr $pr -nb 10
done
for pr in 0.0 0.4; do
    python main.py -data Cifar10_alpha05_200 -nc 200 -jr 0.5 -algo FedUCBN -sca UCB_cs -gr 499 -pr $pr -nb 10
done

for pr in 0.0 0.4; do
    python main.py -data Cifar10_alpha05_300 -nc 300 -jr 0.1 -algo FedUCBN -sca UCB -gr 499 -pr $pr -nb 10
done
for pr in 0.0 0.4; do
    python main.py -data Cifar10_alpha05_300 -nc 300 -jr 0.1 -algo FedUCBN -sca Random -gr 499 -pr $pr -nb 10
done
for pr in 0.0 0.4; do
    python main.py -data Cifar10_alpha05_300 -nc 300 -jr 0.1 -algo FedUCBN -sca UCB_cs -gr 499 -pr $pr -nb 10
done

for pr in 0.0 0.4; do
    python main.py -data Cifar10_alpha05_300 -nc 300 -jr 0.5 -algo FedUCBN -sca UCB -gr 499 -pr $pr -nb 10
done
for pr in 0.0 0.4; do
    python main.py -data Cifar10_alpha05_300 -nc 300 -jr 0.5 -algo FedUCBN -sca Random -gr 499 -pr $pr -nb 10
done
for pr in 0.0 0.4; do
    python main.py -data Cifar10_alpha05_300 -nc 300 -jr 0.5 -algo FedUCBN -sca UCB_cs -gr 499 -pr $pr -nb 10
done

for pr in 0.0 0.4; do
    python main.py -data Cifar10_alpha05_400 -nc 400 -jr 0.1 -algo FedUCBN -sca UCB -gr 499 -pr $pr -nb 10
done
for pr in 0.0 0.4; do
    python main.py -data Cifar10_alpha05_400 -nc 400 -jr 0.1 -algo FedUCBN -sca Random -gr 499 -pr $pr -nb 10
done
for pr in 0.0 0.4; do
    python main.py -data Cifar10_alpha05_400 -nc 400 -jr 0.1 -algo FedUCBN -sca UCB_cs -gr 499 -pr $pr -nb 10
done

for pr in 0.0 0.4; do
    python main.py -data Cifar10_alpha05_400 -nc 400 -jr 0.5 -algo FedUCBN -sca UCB -gr 499 -pr $pr -nb 10
done
for pr in 0.0 0.4; do
    python main.py -data Cifar10_alpha05_400 -nc 400 -jr 0.5 -algo FedUCBN -sca Random -gr 499 -pr $pr -nb 10
done
for pr in 0.0 0.4; do
    python main.py -data Cifar10_alpha05_400 -nc 400 -jr 0.5 -algo FedUCBN -sca UCB_cs -gr 499 -pr $pr -nb 10
done

: '
for pr in 0.0 0.4; do
    python main.py -data mnist_alpha05_100 -nc 100 -jr 0.1 -algo FedAvg -sca Random -gr 99 -pr $pr -nb 10
done
for pr in 0.0 0.4; do
    python main.py -data mnist_alpha05_100 -nc 100 -jr 0.1 -algo FedAvg -sca UCB -gr 99 -pr $pr -nb 10
done

for pr in 0.0 0.4; do
    python main.py -data mnist_alpha05_100 -nc 100 -jr 0.5 -algo FedAvg -sca Random -gr 99 -pr $pr -nb 10
done
for pr in 0.0 0.4; do
    python main.py -data mnist_alpha05_100 -nc 100 -jr 0.5 -algo FedAvg -sca UCB -gr 99 -pr $pr -nb 10
done


for pr in 0.0 0.4; do
    python main.py -data fmnist_alpha01_100 -nc 100 -jr 0.1 -algo FedAvg -sca Random -gr 99 -pr $pr -nb 10
done
for pr in 0.0 0.4; do
    python main.py -data fmnist_alpha01_100 -nc 100 -jr 0.1 -algo FedAvg -sca UCB -gr 99 -pr $pr -nb 10
done

for pr in 0.0 0.4; do
    python main.py -data fmnist_alpha01_100 -nc 100 -jr 0.5 -algo FedAvg -sca Random -gr 99 -pr $pr -nb 10
done
for pr in 0.0 0.4; do
    python main.py -data fmnist_alpha01_100 -nc 100 -jr 0.5 -algo FedAvg -sca UCB -gr 99 -pr $pr -nb 10
done


for pr in 0.0 0.4; do
    python main.py -data fmnist_alpha05_100 -nc 100 -jr 0.1 -algo FedAvg -sca Random -gr 99 -pr $pr -nb 10
done
for pr in 0.0 0.4; do
    python main.py -data fmnist_alpha05_100 -nc 100 -jr 0.1 -algo FedAvg -sca UCB -gr 99 -pr $pr -nb 10
done


for pr in 0.0 0.4; do
    python main.py -data fmnist_alpha05_100 -nc 100 -jr 0.5 -algo FedAvg -sca Random -gr 99 -pr $pr -nb 10
done
for pr in 0.0 0.4; do
    python main.py -data fmnist_alpha05_100 -nc 100 -jr 0.5 -algo FedAvg -sca UCB -gr 99 -pr $pr -nb 10
done
'''



#for jr in 0.1 0.5 0.9; do
#    python main.py -data Cifar100_alpha01_10 -nc 10 -jr $jr -algo FedUCBN -sca UCB -gr 799 -pr 0.4 -nb 100
#done
#for jr in 0.1 0.5 0.9; do
#    python main.py -data Cifar100_alpha01_10 -nc 10 -jr $jr -algo FedUCBN -sca UCB_cs -gr 799 -pr 0.0 -nb 100
#done
#for jr in 0.1 0.5 0.9; do
#    python main.py -data Cifar100_alpha01_10 -nc 10 -jr $jr -algo FedUCBN -sca UCB_cs -gr 799 -pr 0.4 -nb 100
#done
#for jr in 0.1 0.5 0.9; do
#    python main.py -data Cifar100_alpha01_10 -nc 10 -jr $jr -algo FedUCBN -sca Random -gr 799 -pr 0.0 -nb 100
#done
#for jr in 0.1 0.5 0.9; do
#    python main.py -data Cifar100_alpha01_10 -nc 10 -jr $jr -algo FedUCBN -sca Random -gr 799 -pr 0.4 -nb 100
#done
#for jr in 0.1 0.5 0.9; do
#    python main.py -data Cifar100_alpha10_10 -nc 10 -jr $jr -algo FedUCBN -sca UCB -gr 799 -pr 0.0 -nb 100
#done
#for jr in 0.1 0.5 0.9; do
#    python main.py -data Cifar100_alpha10_10 -nc 10 -jr $jr -algo FedUCBN -sca UCB -gr 799 -pr 0.4 -nb 100
#done

#for jr in 0.1 0.5 0.9; do
#    python main.py -data Cifar100_alpha10_10 -nc 10 -jr $jr -algo FedUCBN -sca UCB_cs -gr 799 -pr 0.0 -nb 100
#done
#for jr in 0.1 0.5 0.9; do
#    python main.py -data Cifar100_alpha10_10 -nc 10 -jr $jr -algo FedUCBN -sca UCB_cs -gr 799 -pr 0.4 -nb 100
#done
#for jr in 0.1 0.5 0.9; do
#    python main.py -data Cifar100_alpha10_10 -nc 10 -jr $jr -algo FedUCBN -sca Random -gr 799 -pr 0.0 -nb 100
#done
#for jr in 0.1 0.5 0.9; do
#    python main.py -data Cifar100_alpha10_10 -nc 10 -jr $jr -algo FedUCBN -sca Random -gr 799 -pr 0.4 -nb 100
#done


# for jr in 0.1 0.5 0.9; do
#     python main.py -data mnist_alpha05_100 -nc 100 -jr $jr -algo FedUCBN -sca UCB -gr 499 -pr 0.0 -nb 10
# done
#for jr in 0.1 0.5 0.9; do
#    python main.py -data mnist_alpha05_100 -nc 100 -jr $jr -algo FedUCBN -sca UCB_cs -gr 499 -pr 0.0 -nb 10
#done
# for jr in 0.1 0.5 0.9; do
#     python main.py -data mnist_alpha05_100 -nc 100 -jr $jr -algo FedUCBN -sca Random -gr 499 -pr 0.0 -nb 10
# done

# for jr in 0.1 0.5 0.9; do
#     python main.py -data mnist_alpha05_100 -nc 100 -jr $jr -algo FedUCBN -sca UCB -gr 499 -pr 0.4 -nb 10
# done
#for jr in 0.1 0.5 0.9; do
#    python main.py -data mnist_alpha05_100 -nc 100 -jr $jr -algo FedUCBN -sca UCB_cs -gr 499 -pr 0.4 -nb 10
#done
# for jr in 0.1 0.5 0.9; do
#     python main.py -data mnist_alpha05_100 -nc 100 -jr $jr -algo FedUCBN -sca Random -gr 499 -pr 0.4 -nb 10
# done

#for jr in 0.1 0.5 0.9; do
#    python main.py -data fmnist_alpha05_100 -nc 100 -jr $jr -algo FedUCBN -sca UCB -gr 499 -pr 0.0 -nb 10
#done
#for jr in 0.1 0.5 0.9; do
#    python main.py -data fmnist_alpha01_100 -nc 100 -jr $jr -algo FedUCBN -sca UCB_cs -gr 499 -pr 0.0 -nb 10
#done
#for jr in 0.1 0.5 0.9; do
#    python main.py -data fmnist_alpha05_100 -nc 100 -jr $jr -algo FedUCBN -sca Random -gr 499 -pr 0.0 -nb 10
#done

#for jr in 0.1 0.5 0.9; do
#    python main.py -data fmnist_alpha05_100 -nc 100 -jr $jr -algo FedUCBN -sca UCB -gr 499 -pr 0.4 -nb 10
#done
#for jr in 0.1 0.5 0.9; do
#    python main.py -data fmnist_alpha01_100 -nc 100 -jr $jr -algo FedUCBN -sca UCB_cs -gr 499 -pr 0.4 -nb 10
#done
#for jr in 0.1 0.5 0.9; do
#    python main.py -data fmnist_alpha05_100 -nc 100 -jr $jr -algo FedUCBN -sca Random -gr 499 -pr 0.4 -nb 10
#done
