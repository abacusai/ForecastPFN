Create a project directory and copy the content of this reporsitory in `<project-dir>/source`
It is important that the code is in `source` directory otherwise the make targets will fail!

1. Build images
```bash
make build
```
   
2. Download datasets
Get the FRED API key, see https://research.stlouisfed.org/docs/api/api_key.html
Store the key in ../storage/resources/fred/key.txt file in plain text, for example `abcdefghijklmnopqrstuvwxyz123456`.
    
```bash
make init-resources
```    
Note: Downloading FRED dataset may take days.

3. Create experiments

```bash
make make create-experiments experiment=experiments/tl name=shared
```

Names of experiments are defined defined in `experiments/tl/parameters.py`

4. Execute an experiment

After point 3. you should see `../experiments/tl/<name>/<instance>/experiment.cmd`

Execute this command in a container where this directory (parent of the source) is mounted as `/project`.
To run locally try:

```
make exec bash
```

Then

```
`cat /project/experiments/tl/<name>/<instance>/experiment.cmd`
```

The paper results can be found in `experiments/tl/*.ipynb`