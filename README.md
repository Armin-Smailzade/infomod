# infomod
The input dataset must have the following schema:
- id: INT (a monotonically increasing sequence to uniquely identify each row.)
- feature_name_1: INT (feature of the model.)
- ...
- feature_name_n: INT (feature of the model.)
- target: INT (the target outcome that is used for pattern extraction such as false-positive, false-negative, model prediction, etc.)

The input dataset can be defined in the INPUT_DATA variable and you can run the program with the following command:
spark-submit --master yarn --driver-memory 15G --num-executors 50 --executor-cores 5 --executor-memory 20G infomod_pyspark.py
