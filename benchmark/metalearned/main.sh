for pth in $(ls project/shared/); do
    echo "Running ".$pth
    python ./main.py run --path=project/shared/$pth
done;
