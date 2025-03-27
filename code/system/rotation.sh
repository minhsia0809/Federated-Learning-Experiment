#!/bin/bash
for alpha in 01_100_rotation_15angle 05_100_rotation_15angle; do
    python main.py -data Cifar100_alpha$alpha -nc 100 -jr 0.5 -algo FedUCBN -sca UCB_cs -gr 799 -pr 0.0 -nb 100 -lbs 64 -lr 0.1
done