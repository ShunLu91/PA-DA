IDX=0
SeedList=(0 1 2)
Method=spos  # spos/fairnas/sumnas
PythonScript=train_baselines_201.py

for((s=0; s<${#SeedList[*]}; s++)); do
    Seed=${SeedList[s]}

    # GPU
    GPU=$((${IDX} % 8))
    let IDX+=1

    # run
    EXPName=201_baselines_${Method}_s${Seed}
    LogName=log_${EXPName}
    CUDA_VISIBLE_DEVICES=${GPU} nohup python -u ${PythonScript} --gpu_id ${GPU} \
     --method ${Method} --exp_name ${EXPName} --seed ${Seed} \
     > ./logs/${LogName} 2>&1 &

    # display
    echo "GPU:$GPU EXP:$EXPName"
    if [ $GPU = 7 ] ; then
        echo "sleep 30s"
        sleep 30s
    fi


done

tail -f logs/${LogName}
