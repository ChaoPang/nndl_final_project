# nndl_final_project

## Getting Started
### Pre-requisite
The project is built in python 3.7, and project dependency needs to be installed 

`pip3 install -r requirements.txt`
### Train Conv AutoEncoder to increase the resolution  
`
PYTHONPATH=./:$PYTHONPATH python3 recover_resolution_trainer.py --checkpoint_path checkpoint_recover_imgs_v3 --epochs 100 --cifar_data_path data --lr 0.01
`

### Train the model for competition
`
PYTHONPATH=./:$PYTHONPATH python3 multitask_ensemble_trainer.py --training_data_path data/project_data/train_shuffle/ --training_label_path data/project_data/train_data.csv --test_data_path data/project_data/test_shuffle/ --checkpoint_path checkpoint_multitask_essemble_img_32_val_acc --early_stopping_patience 10 --epochs 50 --lr 0.05 --img_size 32 --batch_size 64 --dropout_rate 0.2 --num_of_classifiers 5
`

