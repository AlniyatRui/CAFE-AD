# CAFE-AD

**[ICRA 25] CAFE-AD: Cross-Scenario Adaptive Feature Enhancement for Trajectory Planning in Autonomous Driving**

Junrui Zhang, Chenjie Wang, Jie Peng, Haoyu Li, Jianmin Ji, Yu Zhang and Yanyong Zhang

<p align="left">
<a href='https://arxiv.org/abs/2504.06584' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/arXiv-PDF-red?style=flat&logo=arXiv&logoColor=wihte' alt='arXiv PDF'>
</a>
</p>

## Setup Environment & Dataset Preparation

Setup the nuPlan dataset and conda environment following the [PLUTO](https://github.com/jchengai/pluto).
All methods are trained on a randomly selected 100k-split dataset, the corresponding scenario tokens are saved in the [token_list.txt file](https://github.com/AlniyatRui/CAFE-AD/blob/master/token_list.txt).

## Training
(The training part it not fully tested)

Our method is implemented based on [PLUTO](https://github.com/jchengai/pluto).

Change the `nuplan` configuration in the [train.sh](https://github.com/AlniyatRui/CAFE-AD/blob/master/train.sh) and then execute the following command: 
   ```
   sh train.sh
   ```
Our method requires multiple forward passes during training, which increases the demand for computational resources, so we use a smaller training dataset than PLUTO.

## Inference

1. **Update the Configuration**  
   Change the `nuplan` configuration in the [script/run_pluto_planner](https://github.com/AlniyatRui/CAFE-AD/blob/master/script/run_pluto_planner.sh) file.

2. **Run the Simulation for `test14-hard` Benchmark**  
   Once the configuration is updated, execute the simulation for the `test14-hard` benchmark using the following command:
   ```
   sh simulation.sh
   ```
   
3. **Notes**
   
    This repo also includes simple implenmentation for running log replay and testing the IDM model.
    In the original PLUTO code, the rule-based post-processing would throw an error when no reference line was available in some frames, so we used pure learning results to prevent these errors in those frames.

## Acknowledgments
We thank the developers of [PLUTO](https://github.com/jchengai/pluto), [planTF](https://github.com/jchengai/planTF), for their public code release.

## Citation

If you find this repo useful, please consider giving us a star 🌟 and citing our related paper.

```bibtex
@article{zhang2025cafe,
  title={CAFE-AD: Cross-Scenario Adaptive Feature Enhancement for Trajectory Planning in Autonomous Driving},
  author={Zhang, Junrui and Wang, Chenjie and Peng, Jie and Li, Haoyu and Ji, Jianmin and Zhang, Yu and Zhang, Yanyong},
  journal={arXiv preprint arXiv:2504.06584},
  year={2025}
}
```
