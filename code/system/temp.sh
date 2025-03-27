#!/bin/bash
python main.py -data Cifar100_alpha01_10 -nc 10 -jr 0.1 -algo FedUCBN -sca UCB_cs -gr 799 -pr 0.0 -nb 100
python main.py -data Cifar100_alpha10_10 -nc 10 -jr 0.1 -algo FedUCBN -sca UCB_cs -gr 799 -pr 0.0 -nb 100
