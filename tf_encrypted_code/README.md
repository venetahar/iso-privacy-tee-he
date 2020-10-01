### Getting started
1. Install the required dependencies using either conda or venv. ```pip install -r requirements.txt```
2. Generate the server configs needed by tf_encrypted by running 
```python3 create_tf_encrypted_configs.py```. This generates a '/tmp/tfe.config' file. 
3. Open three terminal tabs. In each one, run the following command required by tf_encrypted:
    ```
    python3 -m tf_encrypted.player --config /tmp/tfe.config server0
    python3 -m tf_encrypted.player --config /tmp/tfe.config server1
    python3 -m tf_encrypted.player --config /tmp/tfe.config server2
    ```
4. Open another two tabs. From the root of the repository run the following in the first one start the server
by choosing the appropriate experiment name and batch size:
    ```bash
    export CURRENT_PACKAGE_ROOT=`pwd`
    export PYTHONPATH="${PYTHONPATH}:$CURRENT_PACKAGE_ROOT"
    python3 tf_encrypted_code/private_inference_server.py --experiment_name=<EXPERIMENT_NAME> --batch_size=<BATCH_SIZE>
    ```
5. Finally, launch the client. It's important to set the same batch size as the server because tf_encrypted precomputes the 
multiplication triplets. Use the benchmark option if you want to benchmark the framework in terms of runtime:
    ```bash
    export CURRENT_PACKAGE_ROOT=`pwd`
    export PYTHONPATH="${PYTHONPATH}:$CURRENT_PACKAGE_ROOT"
    python3 tf_encrypted_code/private_inference_client.py --experiment_name <EXPERIMENT_NAME> --batch_size <BATCH_SIZE> --benchmark
    ```


