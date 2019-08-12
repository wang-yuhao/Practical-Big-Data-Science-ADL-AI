**1. preparations**:
To use this poject in development mode:
```bash
$ python setup.py develop
```
The scripts are then copied to your sys.path and can be directly invocated


<br/><br/>

**2. process data**:  
*in /src/data run:*
```bash
$ python make_dataset.py ../../data/raw/FB15k-237 ../../data/processed/FB15k-237
```
optional: shorten dataset (for test purposes):
```bash
$ python shorten_dataset.py --input ../../data/processed/FB15k-237 --output ../../data/processed/FB15k-237/shortened --portion 20
```
create mid2name dictionary: this generates dictionary to freebase mid to human readable words. It fetches data from google api and wikidata. It requires google api key. It will not retrieve for all mids, since some of them (around 200~300) doesn't exist anymore on both databases.
```bash
$ python mid2name.py --files "path to preprocessed data" --out "output path" --apikey "google api key"
```


*alternative way to get all processed data and dictionaries: dvc pull (see more here: [here](https://gitlab.lrz.de/adl-ai/practical-big-data-science-adl-ai/wikis/Set-up-the-Infrastuctre/How-to-use-DVC-for-datasets))*


<br/><br/>


**3. train models**:  
*in /scripts run:*
- **RotatE**: run_rotate.py
```bash
$ CUDA_VISIBLE_DEVICES=0 python -u run_rotate.py --do_train --cuda --do_valid --do_test --data_path <path to data folder> --model <model name> -n <negative sample size> -b <batch size> -d <number of hidden dimensions> -g <gamma> -a <adversarial_temperature> -adv -lr <learning rate> --max_steps <max number of steps> -save <path to model output folder> --test_batch_size <integer> -de
```
with default parameters:  
```bash
$ CUDA_VISIBLE_DEVICES=0 python -u run_rotate.py --do_train --cuda --do_valid --do_test --data_path ../data/processed/FB15k-237 --model RotatE -n 256 -b 1024 -d 1000 -g 9.0 -a 1.0 -adv -lr 0.00005 --max_steps 100000 -save ../models/RotatE_FB15k-237_3 --test_batch_size 16 -de
```
- **DistMult**: run_distmult.py
```bash
$ python run_distmult.py --data_path <path to data folder> --output <output path to store model>
```
The hyperparamters must be set in script. 

- **ConvE**: run_conve.py

The results will be logged to `10.195.1.54` under the experiment `ConvE`.
You can also change the experiment name using the `--mlflow_experiment_name` option.
If you want to store the results locally set the `MLFLOW_TRACKING_URI` environment variable
to a local path, e.g `./mlruns`.

```bash
$ python run_conve.py --device cuda --tracking_uri http://10.195.1.54
```

For hyperparameter defaults as well as model defaults please refer to the `--help` option.


- **TransE**: run_transe.py
```bash
$ python run_transe.py -e < number of epochs> -lr < learning rate > -b < batch_size > -p < preferred_device, only 'gpu' or 'cpu'> -r < random seed> -dim < embedding dimensions> -g < margin > -n < normalization os entities > -opt < optimizer,only 'SGD' or 'adam' > -d < datasets path > -o < output path > -h < help information >
```
1. Easy run:
```bash
$ python run_transe.py
```
2. Set hyperparameter:
```bash
$ python run_transe.py -e 3000 -lr 0.0005 -b 100 -p 'gpu' -r 123 -dim 100 -g 1 -n 1 -opt 'SGD' -d ../data/processed/FB15k-237 -o ../models
```

- **RESCAL**: run_rescal.py
```bash
$ python run_rescal.py -e < number of epochs> -lr < learning rate > -b < batch_size > -p < preferred_device, only 'gpu' or 'cpu'> -r < random seed> -dim < embedding dimensions> -g < margin > -s < scoring function > -opt < optimizer,ADAGRAD > -d < datasets path > -o < output path > -h < help information >
```
1. Easy run (default parameters are the best parameters):
```bash
$ python run_rescal.py
```
2. Set hyperparameter:
```bash
$ python run_rescal.py -e 500 -lr 0.1 -b 100 -p 'gpu' -r 0 -dim 100 -g 1 -s 2 -opt 'ADAGRAD' -d ../data/processed/FB15k-237 -o ../models
```

- **Ensemble**:

The results will be logged to `10.195.1.54` under the experiment `ensemble-model`.
```bash
$ run_ensemble --mlflow_run_name my-ensemble --preferred_device cpu
```

<br/><br/>
**4. create model output for error analysis**:  
*in /scripts run:*  
- **RotatE**: use_evaluation_model_rotate.py
```bash
$ python use_evaluation_model_rotate.py -m ../models/<model folder name> -d ../data/processed/FB15k-237 -dim <models dimensions> -o ../models/<model folder name>/output
```
with default parameters:  
```bash
$ python use_evaluation_model_rotate.py -m ../models/RotatE_FB15k-237_3 -d ../data/processed/FB15k-237 -dim 1000 -o ../models/RotatE_FB15k-237_3/output
```

- **DistMult**: use_evaluation_model_distmult.py

The model must contain weights of DistMult as entity_embedding.npy and relation_embedding.npy. They can be generated with generate_distmult_embedding_npys.py with saved model from training.
```bash
$ python use_evaluation_model_distmult.py -m ../models/<model folder name> -d ../data/processed/FB15k-237 -o ../models/<model folder name>/output
```

- **ConvE**: use_evaluation_model_conve.py

The artifacts will be logged to `10.195.1.54` under the experiment `conve-error-analysis`.
```bash
$ python use_evaluation_model_conve.py --device cuda --artifact_path sftp://sftpuser@10.195.1.54/sftpuser/mlruns/2/0278ec00cc7b47eda553db7c4f66120e/artifacts/models/conve-model-43
```

- **TransE**: use_evaluation_model_transe.py

```bash
$ python use_evaluation_model_transe.py -m ../models/< model folder name> -d ../data/processed/FB15k-237 -g < margin> -o ../models/< model folder name>/output
```
```bash
$ python use_evaluation_model_transe.py -m ../models/TransE_FB15k-237_Filt -d ../data/processed/FB15k-237 -g 1 -o ../models/TransE_FB15k-237_Filt/output
```

- **RESCAL**: use_evaluation_model_rescal.py
```bash
$ python use_evaluation_model_rescal.py -m ../models/<model folder name> -d ../data/processed/FB15k-237 -o ../models/<model folder name>/output
```
```bash
$ python use_evaluation_model_rescal.py -m ../models/RESCAL -d ../data/processed/FB15k-237 -o ../models/RESCAL/output
```

<br/><br/>

**5. use model selector**:

The trained models must have trained RotatE, TransE, DistMult and RESCAL. Prediction only can be set, if generating lookup can be skipped.
```bash
$ python use_model_selector.py -m '../models/trained_models/' -d '../data/processed/FB15k-237' -o 'model_selector_output' --prediction_only True
```

<br/><br/>

**other**  
*in /scripts run:*  

- **RotatE**: tune_hyperparameter_rotate.py
```bash
$ python tune_hyperparameter_rotate.py -d ../data/processed/FB15k-237 -s ../models/tmp
```