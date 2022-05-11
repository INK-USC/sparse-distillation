# Sparse Distillation

Code for [Sparse Distillation: Speeding Up Text Classification by Using Bigger Models](https://arxiv.org/abs/2110.08536) (to appear at NAACL 2022). The code is modified based on [fairseq](https://github.com/pytorch/fairseq).

## Quick Links
- [Configure Environment](#configure-environment)
- [Example Code](#example-code)
- [Download Checkpoints](#download-checkpoints)
- [Contact Us](#contact-us)

## Configure Environment
1. Basic installation
```bash
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
pip install requests sklearn

# enter the directory that you clone this repository
pip install --editable ./
git submodule update --init --recursive
```
2. We use [submitit](https://github.com/facebookincubator/submitit) for parallization. Altenatively you can modify the code to be directly using Python's multiprocessing.
3. The complete configuration for the conda environment can be found in `fairseq/user_dir/misc/environment.yml`.

## Code Overview
Code for the sparse distilllation project is in the `fairseq/user_dir` directory. Some important files:
* `criterions/distillation_loss.py`   
  * Distillation loss (KL divergence)
* `data/feature_dropout_dataset.py`
   * Randomly dropout some features (n-grams) during data loading. Also this will cut off the sequence if the exceeds a pre-defined max length. 
* `model/feature_based_linear_model.py`
  * This is the DAN model described in the paper.
  * Note that sparse parameters are marked with `p.param_group = "sparse"`, so they will be recognized by the optimizer.
* `optim/mixed_adam.py`
  * This is a hybrid optimizer. During KD, dense parameters will be optimized with regular adam; Sparse parameters will be optimized with [sparse adam](https://pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html).  
* `feature_based_sentence_prediction.py`
  * This is the fairseq task that handles data loading.

## Example code

In the following we will apply the pipeline of sparse distillation to IMDB (movie review sentiment classification). We will use amazon reviews as our unsupervised corpus.

### 1. Fine-tune a RoBERTa model on IMDB

Follow the steps in the [tutorial](https://github.com/pytorch/fairseq/blob/main/examples/roberta/README.custom_classification.md).


### 2. Gather unsupervised data
We will use amazon review data as unlabeled data. Download amazon review data from [here](https://nijianmo.github.io/amazon/index.html). In our experiments we use the files in the `"Small" subsets for experimentation` section, and we use reviews in all categories. If you only want to run small-scale experiments, please consider using `Movies and TV` category only.

Then we need to format these json files into a simpler format - text files. Please reference `demo/format_amazon_review_data.py`, which will format all reviews into a huge `.txt` file. You'll need to change the hard-coded paths in the python script.

Note: `[train|dev|unsup].input0` should be obtained in the previous steps. They are text files containing IMDB input data, and each line is one example.

### 3. Get a n-gram dictionary from the corpus.
Use `demo/get_dict.py`.

It takes in a large text file, finds the top 1 million n-grams, and save it to a pkl file. N is in the range of 1 to 4. This step will take a few hours.

### 4. Extract n-grams 
This step is to extract n-grams from the unsupervised corpus using the dictionary we just obtained. To speed up, this step is done with parallelization.

Use `demo/get_ngrams.py`.

You will need to run this three times: (1) for the unsupervised corpus; (2) for the training set; (3) for the dev set.

The output should be a text file. Each line is a list of indices, representing the n-grams extracted from the raw text.

### 5. Get distillation target
This step is to apply the RoBERTa model we trained in step 1 to the unlabeled distillation corpus to get the distillation target.

Use `demo/get_distillation_target.py`.

The output should be a `.npy` file, containing a numpy array of size `[num_examples, num_classes]`.

This step will take hours (depending on how many threads/gpus you're using).

### 6. Fairseq Pre-preprocess and Binarize
We need to collect all the preprocessing files we have and formulate a binarized dataset that will be used in fairseq.

See `demo/binarize.sh`.

You also need to move the `.npy` files (containing distillation target) to the binarized data directory. 

In the end, the file structure should look like this:

<details>
<summary>File structure</summary>

```
data-bin/
├── input0/
│   ├── train.bin
│   ├── train.idx
│   ├── valid.bin
│   ├── valid.idx
│   ├── valt.bin
│   ├── valt.idx
│   ├── dict.txt
├── label/
│   ├── train.bin
│   ├── train.idx
│   ├── valid.bin
│   ├── valid.idx
│   ├── valt.bin
│   ├── valt.idx
│   ├── dict.txt
├── distill/
│   ├── train.npy
│   ├── valid.npy
│   ├── valt.npy
```
</details>
  
<br>

### 7. Run Distillation
See `demo/train_distillation.sh`.

### 8. Get the predictions on dev set.

See `demo/get_feature_based_predictions.py`. This will run inference on a binarized dataset, and compute accuracy. You can modify the code to save prediction for each instance as well. 

## Download Checkpoints

Checkpoints and n-gram dictionary for IMDB can be downloaded [here](https://drive.google.com/drive/folders/1vVjDrBBQvmJMAb-Y5sG8Vtpl9gVjg8wn?usp=sharing).  
:smiley: Other checkpoints will be added soon.

## Contact Us
If you find bugs in our code, encounter problems when running the code, or have suggestions for the project, please submit an issue, or reach out to Qinyuan (qinyuany@usc.edu).

If you think our work is useful, please cite us with the following BibTeX:

<details>
<summary>BibTeX</summary>

```
@article{Ye2021SparseDS,
  title={Sparse Distillation: Speeding Up Text Classification by Using Bigger Models},
  author={Qinyuan Ye and Madian Khabsa and Mike Lewis and Sinong Wang and Xiang Ren and Aaron Jaech},
  journal={ArXiv},
  year={2021},
  volume={abs/2110.08536}
}
```
</details>