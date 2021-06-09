conda activate python37
declare -i count
for n in 10 15 20 25
do
    for t in 3 4 5
    do
    echo "$count/15"
    python gen_data.py --num_neg $n --true_threshold $t --work_path "." --verbose --dtype "merge" --raw_data_path '/data/cs608_ip_merge_v3.csv'
    done
done
