# Practice session 3: HPC

### 09.03.2022

In this practice session, we will learn how to use the University of Tartu's High Performance Computing Center servers. Later we will use GPUs there to train our big translation models.

## Login

To connect to the server, use SSH. If you only have a Windows system, you will have to use PuTTY instead.

Log in using your university username and password:

```
ssh your_username@rocket.hpc.ut.ee
```

## Finding your way around

If you are not comfortable with Linux commands, check out, for example, [this guide](https://maker.pro/linux/tutorial/basic-linux-commands-for-beginners).

Once you have logged in, you can create a directory where you will keep all the data for your experiments:

```
mkdir mtcourse
```

Move into the new directory:

```
cd mtcourse
```

Create directories in which to store your data and scripts:

```
mkdir data
mkdir scripts
```

## Fairseq

We will install Fairseq in a Conda virtual environment. By containing all the packages we need in a separate clean environment, we want to avoid version conflicts and [this situation](https://xkcd.com/1987/):

![](https://imgs.xkcd.com/comics/python_environment.png)

For additional information on managing Conda environments, see [this guide](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

First, load python:

```
module load any/python/3.8.3-conda
```

Then create a clean environment:

```
conda create -n mtcourse python=3.8
```

Activate the environment:

```
conda activate mtcourse
```

Install PyTorch:

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

In the earlier practice sessions, we installed Fairseq from `pip`.
Now we will install the newest version of Fairseq from source. We will also use
the flag `--editable`. With it, you can change something in Fairseq's code and
continue using it as a package.

```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

Install SentencePiece and TensorBoard as well:

```
pip install sentencepiece
pip install tensorboardX
```

After installing everything, reinstall NumPy:

```
pip uninstall numpy
pip install numpy
```

If you need to deactivate the environment:

```
conda deactivate
```

## SLURM

The Rocket cluster uses SLURM (a scheduling system for running jobs). We will cover all the basic things that you will need in this lab, but you can check out [HPC's guide on SLURM](https://hpc.ut.ee/en/guides/slurm/) as well. **DO NOT** execute commands that require considerable resources (preprocessing, model training, etc.) directly on the head node! **ALWAYS USE SLURM,** or your access to HPC may be suspended. You can do some small and quick stuff on the nead node, like managing your virtual environments, installing packages, copying or removing files, etc.

To submit your jobs to SLURM, you will need scripts like this one (you can find this script in `/gpfs/space/projects/nlpgroup/mt2022/scripts/01_example_script.sh`):

```
#!/bin/bash

# The name of the job is test_job
#SBATCH -J test_job

# Format of the output filename: slurm-jobname.jobid.out
#SBATCH --output=slurm-%x.%j.out

# The job requires 1 compute node
#SBATCH -N 1

# The job requires 1 task per node
#SBATCH --ntasks-per-node=1

# The maximum walltime of the job is 5 minutes
#SBATCH -t 00:05:00

#SBATCH --mem=5G

# If you keep the next two lines, you will get an e-mail notification
# whenever something happens to your job (it starts running, completes or fails)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email@here.com

# Keep this line if you need a GPU for your job
#SBATCH --partition=gpu

# Indicates that you need one GPU node
#SBATCH --gres=gpu:tesla:1

# Commands to execute go below

# Load Python
module load any/python/3.8.3-conda

# Activate your environment
conda activate mtcourse

# Display fairseq's help message
fairseq-train --help
```

Save the scripts that you use for every step of your work. This way, you can always go back and check what you did. Having all your scripts makes it easier to find mistakes.

**Do not** ask for GPU nodes when you perform preprocessing steps (e.g. cleaning, SentencePiece). It will not make them faster and you will block valuable resources. Only use GPUs for training models.

To send your script to the queue:

```
sbatch path/to/your/script.sh
```

You will see output like:

```
Submitted batch job XXX
```

`XXX` will be the ID of your job. If you want to cancel it:

```
scancel XXX
```

Once your job starts running, a file named `slurm-XXX.out` will be created in your current working directory (where you executed `sbatch`). Your log and output will be written into this file.

## Queue

You can view your jobs that are pending or running:

```
squeue -u your_username
```

Or all the GPU jobs on the cluster:

```
squeue -p gpu
```

Or simply all the jobs on the cluster (the list will be very long):

```
squeue
```

## Run a script

**Task.** In `/gpfs/space/projects/nlpgroup/mt2022/data/sequence-copy-testing`, you will find files with data similar to the reversed copy task from our first lab.

1. Copy the files to your personal directory.
2. Using a SLURM script, binarize the data (command `fairseq-preprocess`; don't forget to activate the `mtcourse` environment beforehand).
3. Then train a small model on these data. If the GPU queue is busy, don't use a GPU for now. Use the following parameters (note that you need to change the paths to the binarized data and `save-dir` to the actual locations of your data, either absolute or relative):

```
fairseq-train path/to/binarized/data --arch transformer \
                                     --lr 0.005 \
                                     --encoder-attention-heads 2 \
                                     --encoder-embed-dim 8 \
                                     --encoder-layers 1 \
                                     --encoder-ffn-embed-dim 32 \
                                     --decoder-attention-heads 2 \
                                     --decoder-embed-dim 8 \
                                     --decoder-layers 1 \
                                     --decoder-ffn-embed-dim 32 \
                                     --max-epoch 3 \
                                     --optimizer adam \
                                     --max-tokens 5000 \
                                     --save-dir your/checkpoint/directory \
                                     --log-format json \
                                     --eval-bleu \
                                     2>&1 | tee log.out
```

Your goal is not to get a good model, but just to check that training on a GPU works.

4. Make it run and check the contents of your output file to see if everything works correctly.
