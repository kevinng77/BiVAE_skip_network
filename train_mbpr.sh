conda activate python37
for n in 2 5 10
do
    for t in 3 4
    do
    echo "${count}/06"
    python train_mbpr.py --lr 0.1 --k 128 --lambda_reg 0.003 --epochs 20 --work_path "." --train_data_path "/data/Modify_train_${t}_${n}.csv"
    done
done
