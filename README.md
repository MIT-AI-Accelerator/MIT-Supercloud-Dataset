# MIT Supercloud Dataset
This dataset consists of the labelled parts of the data described in the paper [_The MIT Supercloud Dataset_](http://arxiv.org/abs/2108.02037). The archive contains compressed CSV files consisting of monitoring data from the MIT Supercloud system. For details on the capabilites offered by MIT Supercloud cluster see [_Reuther, et. al. IEEE HPEC 2018_](https://arxiv.org/abs/1807.07814).

# Citation
If you use this data in your work, please cite the following paper 

```
@INPROCEEDINGS{supercloud,
  author={Samsi, Siddharth and Weiss, Matthew L and Bestor, David and Li, Baolin and Jones, Michael and Reuther, Albert and Edelman, Daniel and Arcand, William and Byun, Chansup and Holodnack, John and Hubbell, Matthew and Kepner, Jeremy and Klein, Anna and McDonald, Joseph and Michaleas, Adam and Michaleas, Peter and Milechin, Lauren and Mullen, Julia and Yee, Charles and Price, Benjamin and Prout, Andrew and Rosa, Antonio and Vanterpool, Allan and McEvoy, Lindsey and Cheng, Anson and Tiwari, Devesh and Gadepally, Vijay},
  booktitle={2021 IEEE High Performance Extreme Computing Conference (HPEC)}, 
  title={The MIT Supercloud Dataset}, 
  year={2021},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/HPEC49654.2021.9622850}
}
```

Please see our website for additional information : https://dcc.mit.edu

Any questions about the dataset can be sent to mit-dcc@mit.edu

# Data Organization
The MIT Supercloud Dataset consists of anonymized scheduler logs, time series data from CPUs and GPUs, monitoring data from each compute node. The uncompressed dataset is of the order of a Terabyte and is made available in compressed files. The dataset includes traces from 460,497 jobs, out of which 98,177 are jobs that requested GPUs for a variety of compute workloads which include AI/ML training and inference. The data is organized in the following directories:

_Note:_ Slurm jobids, compute node hostnames, user ids and other identifiable entries have been anonymized prior to publication of this dataset. Please see the paper above for details.

### CPU utilization
The `cpu` folder contains time-series profiling data collected using the Slurm profiler plugin on each node assigned to a job included in this dataset. The data is organized in 100 subfolders. Each CSV file in the archive contains CPU usage, memory usage and file read/write data collected at 10 second intervals. Also included is a summary of the usage data for each job in the corresponding folder.

### GPU utilization
The GPU utilization data is organized in 100 directories in the `gpu` folder. Each CSV file in these folders contains GPU usage data collected on all GPUs across all nodes assigned to a job. GPU data is collected using the `nvidia-smi` utility at 100 ms intervals.

### Compute node utilization
We include monitoring data collected from each compute node at 5 minute intervals. This includes system load, number of users, number of processes per user, memory usage on the node and total number of Lustre remote procedure calls. This data is contained in the csv file `node-data.csv`. The data in this file provides a snapshot of the node utilization and is independent of the utilization data collected using Slurm.

### Slurm scheduler data
We also include the following data from the Slurm scheduler:

- **slurm-log.csv**: Slurm accounting information for the jobs included in this dataset.

- **tres-mapping.txt**: This file contains a mapping between trackable resources (tres) in Slurm requested by a job. These resources are listed in the `tres_req` column in the `labelled-slurm-log.csv` file. This CSV file maps the integer values to the corresponding resource (e.g.: GPU type, CPU, memory) as shown below:

| Tres ID   |      Resource      |   | Tres ID   |      Resource      | 
|:----------:|:-------------:| --------- |:----------:|:-------------:|
| 1 | cpu |     | 6 | fs |
| 2 | mem |     | 7 | vmem |
| 3 | energy |     | 8 | pages |
| 4 | node |     | 1001 | gpu:tesla |
| 5 | billing |     | 1002 | gpu:volta |

### Labeled DNN workloads
This dataset includes 3,425 known deep learning workloads. We provide a mapping between Slurm jobids and the actual model that was trained as part of the job. This relase includes compute workloads from a variety of published, open soure deep neural networks, trained on publicly available data. We provide the following additional files to help identify these jobs in the anonnymized Slurm logs and the corresponding time-series data:

- **labelled_jobids.csv**: This file contains the mapping between Slurm Job IDs and the type of neural network model trained in a given job. Jobs in this dataset consist of standard convolutional neural networks (ResNet, Inception, VGG), U-Nets, Natural Language models and Graph Neural Networks. We include a mix of Tensorflow and pytorch implementations of these models. Current distribution of jobs is listed in the table below: 

| Vision Networks   | Job Count || Language Models   | Job Count  || Graph Neural Networks   |  Job Count      | 
|----------|:-------------:|-|----------|:-------------:|-|----------|:-------------:|
| VGG | 560 | | Bert | 189 || DimeNet | 33 |
| ResNet | 463 | | DistillBert | 172 || SchNet | 39 |
| Inception | 484 | |||| PNA | 27 |
| U-Net | 1431 | |||| NNConv | 32 |

Data collection of the above jobs is ongoing and this labelled dataset will be further augmented with other types of machine learning and AI workloads over time. 

#### Code availability
The known workloads above used open source implementations and publicly available datasets as listed below:

- TensorFlow implementations of VGG, ResNet and Inception are available from https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks
These models were trained on the ImageNet dataset.

- U-Net models used here are described in the paper [_SEVIR : A Storm Event Imagery Dataset for Deep Learning Applications in Radar and Satellite Meteorology_](https://proceedings.neurips.cc/paper/2020/hash/fa78a16157fed00d7a80515818432169-Abstract.html), NeurIPS 2020. Implementations and dataset used for training are available from https://github.com/MIT-AI-Accelerator/neurips-2020-sevir.

- Pytorch implementations of Bert and DistillBert from the HuggingFace library were trained on fine tuning tasks. Available at https://huggingface.co.

- Pytorch implementations of DimeNet, SchNet, NNConv, and PNA are available in the `pytorch-geometric` package. For documentation see the following pages:  
  - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/dimenet.html
  - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/schnet.html
  - https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.NNConv 
  - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/pna_conv.html 

  These models were trained on the QM9 and ZINC datasets.

#### Acknowledgement
Research was sponsored by the United States Air Force Research Laboratory and the United States Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the United States Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

