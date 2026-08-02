# VTO: Visual Tool Orchestration for Video Anomaly Detection

Rui Wang<sup>1</sup>, Yeteng Wu<sup>1</sup>, Xianlin Zhang<sup>2</sup>, Mengshi Qi<sup>1*</sup>

<sup>1</sup> State Key Laboratory of Networking and Switching Technology, Beijing University of Posts and Telecommunications, China  
<sup>2</sup> School of Digital Media & Design Arts, Beijing University of Posts and Telecommunications, China

---

## News

- Our paper is accepted to ACM Multimedia 2026 (MM '26).

---

## Overview
<p align="center">
    <img src="assets/vto_framework.png" width="95%">
  </p>
VTO is a process-supervised reinforcement learning framework for dynamic visual-tool orchestration in video anomaly detection (VAD). Instead of treating visual tools as isolated modules, VTO iteratively reasons over a video, invokes specialized experts, incorporates their observations, and constructs a complete causal chain before producing an explainable anomaly report.

- **VAD-Tool benchmark.** A hierarchical tool environment with 12 visual experts covering entity tracking, behavior and scene understanding, and high-stakes hazard detection.
- **Process-Supervised Cognitive Alignment.** Fine-grained step-wise supervision combines objective tool-execution signals with foundation-model-based assessments of logicality and causal completeness.
- **Dynamic policy optimization.** GRPO improves multi-step orchestration while explicitly discouraging premature termination after detecting only a primary anomaly.

VTO improves tool-scheduling accuracy by up to **10.2 absolute percentage points** over the evaluated baselines and enables compact models to outperform substantially larger models on complex multi-step anomaly reasoning.

---

## VAD-Tool

The 12 tools are organized according to a practical security-analysis pipeline:

| Category | Visual experts | Supported inputs |
| --- | --- | --- |
| Entity Identity & Tracking | Person Re-identification, Gait Recognition, License Plate Recognition, Vehicle Re-identification | Image / Video |
| Behavior & Scene Understanding | Pose Estimation, Human Fall Detection, Crowd Counting, Scene Recognition | Image / Video |
| Security Event & Hazard Detection | Violence Detection, Weapon Detection, Fire and Smoke Detection, Anomaly Detection | Image / Video / Text |

The benchmark contains both single-step tasks and interrelated multi-step tasks. Its processed training and evaluation annotations are under `clean/one_step_data/` and `clean/multi_step_data/`; tool-specific benchmark files are under `VIoT_tool/`.

---


## Acknowledgments

This project builds upon several open-source projects and pretrained models, including:

- [verl](https://github.com/volcengine/verl) — reinforcement learning framework used for GRPO training;
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — efficient SFT and LoRA training;
- [FastReID](https://github.com/JDAI-CV/fast-reid), [OpenGait](https://github.com/ShiqiYu/OpenGait), [RTMO/MMPose](https://github.com/open-mmlab/mmpose), [MMAction2](https://github.com/open-mmlab/mmaction2), and the other visual experts integrated into VAD-Tool.

We thank the authors and maintainers of these projects for making their work publicly available.

---

## Citation

If you find this project useful, please cite:

```bibtex
@inproceedings{wang2026vto,
  title     = {VTO: Visual Tool Orchestration for Video Anomaly Detection},
  author    = {Wang, Rui and Wu, Yeteng and Zhang, Xianlin and Qi, Mengshi},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year      = {2026},
  doi       = {10.1145/3767308.3836202}
}
```

---

## License

The paper is distributed under the Creative Commons Attribution 4.0 International License. Licensing terms for the source code and third-party models follow their respective files and upstream projects.
