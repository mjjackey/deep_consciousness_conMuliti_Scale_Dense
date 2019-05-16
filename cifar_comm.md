```bash
echo "Evaluating Cifar10 results"
wget https://www.dropbox.com/s/kjxh1oaxkhtpbmw/anytime_cifar_10.h.tar
python3 main.py --model msdnet -b 64 -j 2 cifar10 --msd-blocks 10 --msd-base 4 --msd-step 2 --msd-stepmode even --growth 6-12-24 --gpu 0 --resume --evaluate-from anytime_cifar_10.pth.tar
read -p "Press [Enter] key to proceed to Cifar 10 test of MSDNet-GCN with base 2..."

echo "Evaluating Cifar10 results with GCN kernel and base 2"
wget https://www.dropbox.com/s/ifu6efuuhzn9mwi/all_shared_gcn_cifar10_b2_k5_s1.tar
python3 main.py --model msdnet -b 64 -j 2 cifar10 --msd-blocks 10 --msd-base 2 --msd-step 1 --msd-stepmode even --growth 6-12-24 --gpu 0 --msd-gcn --msd-gcn-kernel 5 --msd-share-weights --msd-all-gcn --resume --evaluate-from all_shared_gcn_cifar10_b2_k5_s1.tar
read -p "Press [Enter] key to proceed to Cifar 10 test of MSDNet-GCN with base 3..."

echo "Evaluating Cifar100 results with GCN kernel and base 3"
wget https://www.dropbox.com/s/6rt2qhfajoq3bte/all_shared_gcn_cifar100_b3_k5_s1.tar
python3 main.py --model msdnet -b 64 -j 2 cifar100 --msd-blocks 10 --msd-base 3 --msd-step 1 --msd-stepmode even --growth 6-12-24 --gpu 0 --msd-gcn --msd-gcn-kernel 5 --msd-share-weights --msd-all-gcn --resume --evaluate-from all_shared_gcn_cifar100_b3_k5_s1.tar
read -p "Example tests are done! please look at script.sh for more examples for training MSDNet-GCN."

###### train anytime train on cifar10 
python main.py --model msdnet -b 64  cifar10 --epochs 30 --msd-blocks 10 --msd-base 4 --msd-step 2 --msd-stepmode even --growth 6-12-24 --gpu 0  --debug --savedir anytime_cifar10/

###### train anytime train on cifar10 
python main.py --model msdnet -b 64  cifar100 --epochs 30 --msd-blocks 10 --msd-base 4 --msd-step 2 --msd-stepmode even --growth 6-12-24 --gpu 0  --debug --savedir anytime_cifar100/

###### train budgeted batch classification train on cifar10 
python main.py --model msdnet -b 64  -j 2 cifar10  --epochs 30 --msd-blocks 7 --msd-base 1 --msd-step 1 --msd-stepmode lin_grow  --growth 6-12-24 --gpu 0 --savedir budg_cifar10/

# budgeted batch classification evaluate on cifar10
python main.py --model msdnet -b 64 -j 2 cifar10 --msd-blocks 7 --msd-base 1 --msd-step 1 --msd-stepmode lin_grow  --growth 6-12-24 --gpu 0 --evalmode dynamic --resume --evaluate-from budg_cifar_10.pth.tar 

###### train budgeted batch classification train on cifar100
python main.py --model msdnet -b 64  -j 2 cifar100  --epochs 30  --msd-blocks 7 --msd-base 1 --msd-step 1 --msd-stepmode lin_grow  --growth 6-12-24 --gpu 0 --savedir budg_cifar100/

# budgeted batch classification evaluate on cifar100
python main.py --model msdnet -b 64 -j 2 cifar100 --msd-blocks 7 --msd-base 1 --msd-step 1 --msd-stepmode lin_grow  --growth 6-12-24 --gpu 0 --evalmode dynamic --resume --evaluate-from budg_cifar_10.pth.tar 

###### train to apply GCN on all layers anytime train on cifar10, base=2
python main.py --model msdnet -b 64  -j 2 cifar10 --epochs 30 --msd-blocks 10 --msd-base 2 
--msd-step 1 --msd-stepmode even --growth 6-12-24 --gpu 0  --msd-gcn --msd-gcn-kernel 5 
--msd-share-weights --msd-all-gcn --savedir gcn_cifar10/

# To apply GCN on all layers anytime evaluate on cifar10, base=2
python main.py --model msdnet -b 64 -j 2 cifar10 --msd-blocks 10 --msd-base 2 --msd-step 1 --msd-stepmode even --growth 6-12-24 --gpu 0 --msd-gcn --msd-gcn-kernel 5 --msd-share-weights --msd-all-gcn --resume --evaluate-from all_shared_gcn_cifar10_b2_k5_s1.tar

###### train to apply GCN on all layers anytime train on cifar100, base=2
python main.py --model msdnet -b 64  -j 2 cifar100 --epochs 30 --msd-blocks 10 --msd-base 2 
--msd-step 1 --msd-stepmode even --growth 6-12-24 --gpu 0  --msd-gcn --msd-gcn-kernel 5 
--msd-share-weights --msd-all-gcn --savedir gcn_cifar100/

# To apply GCN on all layers anytime evaluate on cifar10, base=2
python main.py --model msdnet -b 64 -j 2 cifar100 --msd-blocks 10 --msd-base 2 --msd-step 1 --msd-stepmode even --growth 6-12-24 --gpu 0 --msd-gcn --msd-gcn-kernel 5 --msd-share-weights --msd-all-gcn --resume --evaluate-from all_shared_gcn_cifar100_b2_k5_s1.tar


```