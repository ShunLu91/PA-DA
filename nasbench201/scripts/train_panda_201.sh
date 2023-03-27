IDX=3
SeedList=(0 1 2)
Epochs=250
BatchSize=256
PythonScript=train_panda_201.py

for((s=0; s<${#SeedList[*]}; s++)); do

    # GPU
    GPU=$((${IDX} % 8))
    let IDX+=1

    # run
    Seed=${SeedList[s]}
    EXPName=201_panda_s${Seed}
    LogName=log_${EXPName}
    CUDA_VISIBLE_DEVICES=${GPU} nohup python -u ${PythonScript}  \
        --gpu_id ${GPU} --exp_name ${EXPName} --seed ${Seed} --epochs ${Epochs} \
        --train_batch_size ${BatchSize} \
     > ./logs/${LogName} 2>&1 &

    # display
    echo "GPU:$GPU EXP:$EXPName"
    if [ $GPU = 7 ] ; then
        echo "sleep 30s"
        sleep 30s
    fi


done

tail -f logs/${LogName}
