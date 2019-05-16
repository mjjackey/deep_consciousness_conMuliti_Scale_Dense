```bash
usage: main.py [-h] [--model M] [--growth GROWTH RATE] [--reduction R]
               [--savedir PATH] [--resume] [--pretrained] [--no-save-model]
               [--evaluate] [--convert-from PATH] [--evaluate-from PATH]
               [-j N] [--epochs N] [--start-epoch N] [-b N] [--lr LR]
               [--lr-type T] [--momentum M] [--weight-decay W]
               [--print-freq N] [--manual-seed N] [--gpu GPU] [--debug]
               [--msd-base B] [--msd-blocks nB] [--msd-stepmode nB]
               [--msd-step S] [--msd-bottleneck]
               [--msd-bottleneck-factor bottleneck rate factor of each sacle]
               [--msd-growth GROWTH RATE]
               [--msd-growth-factor growth factor of each sacle]
               [--msd-prune MSD_PRUNE] [--msd-join-type MSD_JOIN_TYPE]
               [--msd-all-gcn] [--msd-gcn] [--msd-share-weights]
               [--msd-gcn-kernel KERNEL_SIZE] [--msd-kernel KERNEL_SIZE]
               DIR

PyTorch MSDNet implementation

positional arguments:
  DIR                   Path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --model M             Model to train the dataset
  --growth GROWTH RATE  Per layer growth
  --reduction R         Transition reduction (default: 0.5)
  --savedir PATH        Path to save result and checkpoint (default:
                        results/savedir)
  --resume              Use latest checkpoint if have any (default: none)
  --pretrained          Use pre-trained model (default: false)
  --no-save-model       Only save best model (default: false)
  --evaluate            Evaluate model on validation set (default: false)
  --convert-from PATH   Path to saved checkpoint (default: none)
  --evaluate-from PATH  Path to saved checkpoint (default: none)
  -j N, --workers N     Number of data loading workers (default: 4)
  --epochs N            Number of total epochs to run (default: 300)
  --start-epoch N       Manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256)
  --lr LR, --learning-rate LR
                        initial learning rate (default: 0.1)
  --lr-type T           Learning rate strategy (default: cosine)
  --momentum M          Momentum (default: 0.9)
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)
  --print-freq N, -p N  print frequency (default: 10)
  --manual-seed N       Manual seed (default: 0)
  --gpu GPU             gpu available
  --debug               enable debugging
  --msd-base B          The layer to attach the first classifier (default: 4)
  --msd-blocks nB       Number of blocks/classifiers (default: 1)
  --msd-stepmode nB     Pattern of span between two adjacent classifers
                        [even|lin_grow] (default: even)
  --msd-step S          Span between two adjacent classifers (default: 1)
  --msd-bottleneck      Use 1x1 conv layer or not (default: True)
  --msd-bottleneck-factor bottleneck rate factor of each sacle
                        Per scale bottleneck
  --msd-growth GROWTH RATE
                        Per layer growth
  --msd-growth-factor growth factor of each sacle
                        Per scale growth
  --msd-prune MSD_PRUNE
                        Specify how to prune the network
  --msd-join-type MSD_JOIN_TYPE
                        Add or concat for features from different paths
  --msd-all-gcn         Use GCN blocks for all MSDNet layers
  --msd-gcn             Use GCN block for the first MSDNet layer
  --msd-share-weights   Use GCN blocks for MSDNet
  --msd-gcn-kernel KERNEL_SIZE
                        GCN Conv2d kernel size
  --msd-kernel KERNEL_SIZE
                        MSD Conv2d kernel size
```

