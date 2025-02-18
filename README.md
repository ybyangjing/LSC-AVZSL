# Learning Semantic Consistency for Audio-Visual Zero-shot Learning

This is the official PyTorch code for the paper:

**Learning Semantic Consistency for Audio-Visual Zero-shot Learning**

***Artificial Intelligence Review***

**Xiaoyng Li**, **Jing Yang***, **Yuling Chen**, **Wei Zhang**,  **Xiaoli Ruan**, **Chengjiang Li and Zhidong Su**.

<p align="center">
  <img src="img/introduction.jpg"alt="" align=center />
</p>

## Requirements
Install all required dependencies into a new virtual environment via conda.
```shell
conda env create -f lsc.yml
conda activate lsc
```


## Download Features

Download the datasets following the project [```ClipClap-GZSL```](https://github.com/dkurzend/ClipClap-GZSL).


# Training
In order to train the model run the following command:
```python3 main.py --cfg CFG_FILE  --root_dir ROOT_DIR --log_dir LOG_DIR --dataset_name DATASET_NAME --run all```

```
arguments:
--cfg CFG_FILE is the file containing all the hyperparameters for the experiments. To replicate our results, use ```--cfg config/clipclap.yaml``` for all three datasets.
--root_dir ROOT_DIR indicates the location where the dataset is stored.
--dataset_name {VGGSound, UCF, ActivityNet} indicate the name of the dataset.
--log_dir LOG_DIR indicates where to save the experiments.
--run {'all', 'stage-1', 'stage-2'}. 'all' indicates to run both training stages + evaluation, whereas 'stage-1', 'stage-2' indicates to run only those particular training stages
```


# Evaluation

Evaluation can be done in two ways. Either you train with ```--run all``` which means that after training the evaluation will be done automatically, or you can do it manually.

For manual evaluation run the following command:

```python3 get_evaluation.py --cfg CFG_FILE --load_path_stage_A PATH_STAGE_A --load_path_stage_B PATH_STAGE_B --dataset_name DATASET_NAME --root_dir ROOT_DIR```

```
arguments:
--cfg CFG_FILE is the file containing all the hyperparameters for the experiments. To replicate our results, use ```--cfg config/clipclap.yaml``` for all three datasets.
--load_path_stage_A will indicate to the path that contains the network for stage 1
--load_path_stage_B will indicate to the path that contains the network for stage 2
--dataset_name {VGGSound, UCF, ActivityNet} will indicate the name of the dataset
--root_dir points to the location where the dataset is stored
```

**The complete code will be uploaded after the paper is accepted.**

**If you have any questions you can contact us : gs.xiaoyongli22@gzu.edu.cn or jyang23@gzu.edu.cn**