# VTO: Visual Tool Orchestration for Video Anomaly Detection

Rui Wang<sup>1</sup>, Yeteng Wu<sup>1</sup>, Xianlin Zhang<sup>2</sup>, Mengshi Qi<sup>1*</sup>

<sup>1</sup> State Key Laboratory of Networking and Switching Technology, Beijing University of Posts and Telecommunications, China  
<sup>2</sup> School of Digital Media & Design Arts, Beijing University of Posts and Telecommunications, China

---

## News

- Our paper is accepted to ACM Multimedia 2026 (MM '26).

---

## Overview

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

## Environment Setup

The repository contains multiple visual experts built on different generations of OpenMMLab and related libraries. The original experiments therefore use two isolated Conda environments:

- `holmes`: hosts the video anomaly detection service;
- `agent-copy`: runs the VTO agent and the remaining visual tools.

A package snapshot from the original tool environment is provided in `requirements.txt`. Because several expert models require specific CUDA, PyTorch, MMCV, MMAction2, MMPose, and MMDetection combinations, create the environments for your CUDA/PyTorch platform and install each expert's dependencies before running the full tool suite.

```bash
git clone <repository-url>
cd VIoTGPT

# Create the two environments, then install the recorded dependencies
conda create -n holmes python=3.10
conda create -n agent-copy python=3.10
conda activate agent-copy
pip install -r requirements.txt
```

Download the required base LLM/VLM checkpoints, LoRA or GRPO checkpoints, and visual-expert weights. Before execution, replace the machine-specific paths in the selected shell script, including `WORK_DIR`, Conda initialization, `CUDA_HOME`, model paths, checkpoint paths, GPU IDs, and output paths.

---

## Repository Structure

```text
VIoTGPT/
├── clean/                         # Processed single-step and multi-step benchmark data
├── DataProcess/                   # Data filtering, splitting, mapping, and format conversion
├── VIoT_tool/                     # VAD-Tool annotations and tool-specific test sets
├── project/                       # Implementations of specialized visual experts
├── mmaction2/                     # Action-recognition / video-understanding dependency
├── mmcv/ and mmagic/              # OpenMMLab dependencies used by visual tools
├── holmes_server_portable.py      # Portable anomaly-detection service
├── remote_tool_service.py         # Remote visual-tool service utilities
├── ww_test_VIoTGPT_test.py        # LoRA/SFT inference entry point
├── ww_test_VIoTGPT_test_analysis.py # GRPO inference and trajectory analysis
├── evaluation/eval_tools.py       # Tool-call and response evaluation
├── batch_test_lora_parallel.sh    # Parallel LoRA evaluation
├── batch_test_grpo_parallel.sh    # Parallel GRPO evaluation
├── batch_test_grpo_ablation_parallel.sh # Reward ablations
└── VTO_MM_26_CR_version_Rui_Wang.pdf    # Paper
```

---

## Usage

### 1. Prepare VAD-Tool data

The released processed splits can be used directly:

```text
clean/one_step_data/split_output/All_final_eval_query.json
clean/multi_step_data/split_output/All_final_eval_query.json
```

Scripts in `DataProcess/Single_Tool/` and `DataProcess/Multiple_Tool/` reproduce filtering, splitting, Qwen/LLaMA-format conversion, merging, and system-prompt construction. Update their input/output paths for your local data layout before use.

### 2. Start the anomaly-detection service

```bash
conda activate holmes
CUDA_VISIBLE_DEVICES=0 python holmes_server_portable.py --host 127.0.0.1 --port 8000
```

### 3. Run VTO inference

In another terminal, point the agent to the same service port and launch the 12-tool pipeline:

```bash
conda activate agent-copy
export HOLMES_PORT=8000

CUDA_VISIBLE_DEVICES=1 python ww_test_VIoTGPT_test_analysis.py \
  --load PersonReid_cuda:0,FSDetect_cuda:0,HumanPoseEstimation_cuda:0,GroundingDINOWeaponDetector_cuda:0,VideoAnomalyDetection_cuda:0,CrowdCounting_cuda:0,HumanFallDetection_cuda:0,SceneRecognition_cuda:0,ViolenceDetection_cuda:0,VehicleReid_cuda:0,GaitRecognition_cuda:0,PlateRecognition_cuda:0 \
  --query_path clean/multi_step_data/split_output/All_final_eval_query.json \
  --query_data_path . \
  --model_path /path/to/vto-checkpoint \
  --output_path evaluation/vto_multistep_result.json \
  --stats_path evaluation/vto_multistep_stats.json \
  --limit 10000
```

For an SFT/LoRA checkpoint, use `ww_test_VIoTGPT_test.py` and additionally pass `--lora_path /path/to/lora-checkpoint`.

### 4. Reproduce batch evaluation

After adapting paths and GPU assignments in the scripts:

```bash
bash batch_test_lora_parallel.sh          # SFT/LoRA baselines
bash batch_test_grpo_parallel.sh          # VTO / GRPO models
bash batch_test_grpo_ablation_parallel.sh # reward ablations
```

These launch independent `tmux` sessions for the Holmes service and agent jobs. Use `--status` to list sessions and `--cleanup` to stop the sessions created by a batch script.

### 5. Evaluate predictions

```bash
python evaluation/eval_tools.py \
  --GT_file /path/to/ground_truth.json \
  --prediction_path /path/to/prediction_directory \
  --pred_json predictions.jsonl
```

The evaluation code parses the `Thought / Action / Action Input / Observation` trajectories and reports tool-decision, tool-name, parameter, and final-response statistics for single-step and multi-step settings.

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
