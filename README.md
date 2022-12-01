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
PYTHONPATH=./:$PYTHONPATH python3 basic_trainer.py --training_data_path data/project_data/train_shuffle/ --training_label_path data/project_data/train_data.csv --test_data_path data/project_data/test_shuffle/ --checkpoint_path checkpoint_regnet_32 --early_stopping_patience 10 --epochs 100 --test_label --lr 0.008 --cifar_data_path data/ --img_size 8 --external_validation --up_sampler_path checkpoint_recover_imgs_v2/final_model.pt 
`