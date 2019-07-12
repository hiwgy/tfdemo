#! /usr/bin/python
import sys
import argparse

SOCIAL_BASE_LOW=3387
SOCIAL_BASE_HIGH=25401
PAY_BASE=5000
L1=3000
L2=12000
L3=25000
L4=35000
L5=55000
L6=80000
P1=0.03
P2=0.1
P3=0.2
P4=0.25
P5=0.3
P6=0.35
P7=0.45

parser = argparse.ArgumentParser()
parser.add_argument('--salary', default=10000, type=int, help='salary before tax')

def get_salary(salary):
    if salary >= SOCIAL_BASE_HIGH:
        SOCIAL_BASE=SOCIAL_BASE_HIGH
    elif salary <= SOCIAL_BASE_LOW:
        SOCIAL_BASE=SOCIAL_BASE_LOW
    else:
        SOCIAL_BASE=salary

    BASIC_OUT=SOCIAL_BASE*0.12+SOCIAL_BASE*0.08+SOCIAL_BASE*0.02+3+SOCIAL_BASE*0.002
    PS=salary-BASIC_OUT-PAY_BASE

    if PS <= 0:
        PAY=0
    elif PS <= L1:
        PAY=PS*P1
    elif PS <= L2:
        PAY=(PS-L1)*P2+L1*P1
    elif PS <= L3:
        PAY=(PS-L2)*P3+(L2-L1)*P2+L1*P1
    elif PS <= L4:
        PAY=(PS-L3)*P4+(L3-L2)*P3+(L2-L1)*P2+L1*P1
    elif PS <= L5:
        PAY=(PS-L4)*P5+(L4-L3)*P4+(L3-L2)*P3+(L2-L1)*P2+L1*P1
    elif PS <= L6:
        PAY=(PS-L5)*P6+(L5-L4)*P5+(L4-L3)*P4+(L3-L2)*P3+(L2-L1)*P2+L1*P1
    else:
        PAY=(PS-L6)*P7+(L6-L5)*P6+(L5-L4)*P5+(L4-L3)*P4+(L3-L2)*P3+(L2-L1)*P2+L1*P1
    return salary-BASIC_OUT-PAY

if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    print("salary in hand:{}".format(get_salary(args.salary)))

