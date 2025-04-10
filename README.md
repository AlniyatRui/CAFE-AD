# CAFE-AD

[ICRA 25] CAFE-AD: Cross-Scenario Adaptive Feature Enhancement for Trajectory Planning in Autonomous Driving

<p align="left">
<a href='https://arxiv.org/abs/2504.06584' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/arXiv-PDF-red?style=flat&logo=arXiv&logoColor=wihte' alt='arXiv PDF'>
</a>
</p>

## Setup Environment & Dataset Preparation

Setup the nuPlan dataset and conda environment following the https://github.com/jchengai/pluto.

## Training

## Inference

1. **Update the Configuration**  
   Change the `nuplan` configuration in the [script/run_pluto_planner](script/run_pluto_planner) file.

2. **Run the Simulation for `test14-hard` Benchmark**  
   Once the configuration is updated, execute the simulation for the `test14-hard` benchmark using the following command:

   ```
   sh simulation.sh
   ```

## Acknowledgments
We thank the developers of [PLUTO](https://github.com/jchengai/pluto), [planTF](https://github.com/jchengai/planTF), for their public code release.

## Citation

If you find this repo useful, please consider giving us a star 🌟 and citing our related paper.
