# ======================================================================
# MANY - stations with many results, each of which train their own model
# ======================================================================

## From scratch:
### lr=0.01 (default): 2025-11-19 (failed?)
python phasenet/train_stations.py --mode train_valid --epochs 100 --batch_size 10 --format numpy --data_dir dataset/involcan/involcan_data --train_list dataset/involcan/involcan_train_many.csv --valid_list dataset/involcan/involcan_valid_many.csv --test_list dataset/involcan/involcan_test_many.csv

## Fine-tuning (based on PhaseNet model 190703-214543):
### lr=0.01: 2025-12-01
python phasenet/train_stations.py --mode train_valid --epochs 100 --batch_size 10 --format numpy --data_dir dataset/involcan/involcan_data --train_list dataset/involcan/involcan_train_many.csv --valid_list dataset/involcan/involcan_valid_many.csv --test_list dataset/involcan/involcan_test_many.csv --model_dir model/190703-214543
### lr=0.001: 2025-12-02
python phasenet/train_stations.py --mode train_valid --epochs 100 --batch_size 10 --format numpy --data_dir dataset/involcan/involcan_data --train_list dataset/involcan/involcan_train_many.csv --valid_list dataset/involcan/involcan_valid_many.csv --test_list dataset/involcan/involcan_test_many.csv --model_dir model/190703-214543 --learning_rate 0.001




# ===================================================================
# FEW - stations with few picks that are used to train a unique model
# ===================================================================

## From scratch:
### lr=0.01: 2025-12-01
python phasenet/train.py --mode train_valid --epochs 100 --batch_size 10 --format numpy --train_dir dataset/involcan/involcan_data --train_list dataset/involcan/involcan_train_few.csv --valid_dir dataset/involcan/involcan_data --valid_list dataset/involcan/involcan_valid_few.csv --test_list dataset/involcan/involcan_test_few.csv

## Fine-tuning (based on PhaseNet model 190703-214543):
### lr=0.01: 2025-12-02
python phasenet/train.py --mode train_valid --epochs 100 --batch_size 10 --format numpy --train_dir dataset/involcan/involcan_data --train_list dataset/involcan/involcan_train_few.csv --valid_dir dataset/involcan/involcan_data --valid_list dataset/involcan/involcan_valid_few.csv --test_list dataset/involcan/involcan_test_few.csv --model_dir model/190703-214543
### lr=0.003: 2025-12-02
python phasenet/train.py --mode train_valid --epochs 100 --batch_size 10 --format numpy --train_dir dataset/involcan/involcan_data --train_list dataset/involcan/involcan_train_few.csv --valid_dir dataset/involcan/involcan_data --valid_list dataset/involcan/involcan_valid_few.csv --test_list dataset/involcan/involcan_test_few.csv --model_dir model/190703-214543 --learning_rate 0.003
### lr=0.001: 2025-12-02
python phasenet/train.py --mode train_valid --epochs 100 --batch_size 10 --format numpy --train_dir dataset/involcan/involcan_data --train_list dataset/involcan/involcan_train_few.csv --valid_dir dataset/involcan/involcan_data --valid_list dataset/involcan/involcan_valid_few.csv --test_list dataset/involcan/involcan_test_few.csv --model_dir model/190703-214543 --learning_rate 0.001
### lr=0.0003: 2025-12-02
python phasenet/train.py --mode train_valid --epochs 100 --batch_size 10 --format numpy --train_dir dataset/involcan/involcan_data --train_list dataset/involcan/involcan_train_few.csv --valid_dir dataset/involcan/involcan_data --valid_list dataset/involcan/involcan_valid_few.csv --test_list dataset/involcan/involcan_test_few.csv --model_dir model/190703-214543 --learning_rate 0.0003
### lr=0.0001: 2025-12-02
python phasenet/train.py --mode train_valid --epochs 100 --batch_size 10 --format numpy --train_dir dataset/involcan/involcan_data --train_list dataset/involcan/involcan_train_few.csv --valid_dir dataset/involcan/involcan_data --valid_list dataset/involcan/involcan_valid_few.csv --test_list dataset/involcan/involcan_test_few.csv --model_dir model/190703-214543 --learning_rate 0.0001




# =============================================================
# ALL - data from all stations are used to train a unique model
# =============================================================

## From scratch: 
### lr=0.0001: 2025-12-04
python phasenet/train.py --mode train_valid --epochs 100 --batch_size 128 --format numpy --train_dir dataset/involcan/involcan_data --train_list dataset/involcan/involcan_train_split.csv --valid_dir dataset/involcan/involcan_data --valid_list dataset/involcan/involcan_valid_split.csv --learning_rate 0.0001

## Fine-tuning (based on PhaseNet model 190703-214543):
### lr=0.0001: 2025-12-03
python phasenet/train.py --mode train_valid --epochs 100 --batch_size 128 --format numpy --train_dir dataset/involcan/involcan_data --train_list dataset/involcan/involcan_train_split.csv --valid_dir dataset/involcan/involcan_data --valid_list dataset/involcan/involcan_valid_split.csv --model_dir model/190703-214543 --learning_rate 0.0001
### lr=0.001: 2025-12-04
python phasenet/train.py --mode train_valid --epochs 100 --batch_size 128 --format numpy --train_dir dataset/involcan/involcan_data --train_list dataset/involcan/involcan_train_split.csv --valid_dir dataset/involcan/involcan_data --valid_list dataset/involcan/involcan_valid_split.csv --model_dir model/190703-214543 --learning_rate 0.001
### lr=0.01: 2025-12-05
python phasenet/train.py --mode train_valid --epochs 100 --batch_size 128 --format numpy --train_dir dataset/involcan/involcan_data --train_list dataset/involcan/involcan_train_split.csv --valid_dir dataset/involcan/involcan_data --valid_list dataset/involcan/involcan_valid_split.csv --model_dir model/190703-214543 --learning_rate 0.01
