# Learning to Fuse Asymmetric Feature Maps in Siamese Trackers

This paper has been accepted by CVPR2021

> paper: https://arxiv.org/abs/2012.02776

```
@article{han2020learning,
  title={Learning to Fuse Asymmetric Feature Maps in Siamese Trackers},
  author={Han, Wencheng and Dong, Xingping and Khan, Fahad Shahbaz and Shao, Ling and Shen, Jianbing},
  journal={arXiv preprint arXiv:2012.02776},
  year={2020}
}
```

## weights and raw results
(Please remove the blank after https: by hand. It is used for anti-spider)


raw results https:// iiai-wencheng2.oss-cn-hongkong.aliyuncs.com/acm_raw_results.zip

config and weights for LaSOT https:// iiai-wencheng2.oss-cn-hongkong.aliyuncs.com/LaSOT_weight_config.zip

config weights for VOT https:// iiai-wencheng2.oss-cn-hongkong.aliyuncs.com/weight_VOT2019.zip


## Installation

Please find installation instructions in [`INSTALL.md`](INSTALL.md).

## Quick Start: Using SiamBAN

### Add SiamBAN to your PYTHONPATH

```bash
export PYTHONPATH=/path/to/siamban:$PYTHONPATH
```

### Download models

Download models in [Model Zoo](MODEL_ZOO.md) and put the `model.pth` in the correct directory in experiments

### Download testing datasets

Download datasets and put them into `testing_dataset` directory. Jsons of commonly used datasets can be downloaded from [here](https://drive.google.com/drive/folders/10cfXjwQQBQeu48XMf2xc_W1LucpistPI) or [here](https://pan.baidu.com/s/1et_3n25ACXIkH063CCPOQQ), extraction code: `8fju`. If you want to test tracker on new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to setting `testing_dataset`. 

### Test tracker

```bash
cd experiments/siamban_r50_l234
python -u ../../tools/test.py 	\
	--snapshot model.pth 	\ # model path
	--dataset VOT2018 	\ # dataset name
	--config config.yaml	  # config file
```

The testing results will in the current directory(results/dataset/model_name/)

### Eval tracker

assume still in experiments/siamban_r50_l234

``` bash
python ../../tools/eval.py 	 \
	--tracker_path ./results \ # result path
	--dataset VOT2018        \ # dataset name
	--num 1 		 \ # number thread to eval
	--tracker_prefix 'model'   # tracker_name
```

## License

This project is released under the [Apache 2.0 license](LICENSE). 
