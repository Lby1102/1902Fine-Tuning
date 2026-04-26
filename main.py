import os
import config
from train import train

def main():
    os.makedirs(config.output_dir,exist_ok=True)
    print('-'*20)
    print('fine tuning starts')
    train()
    print("fine tuning has been finished")

if __name__=='__main__':
    main()

