# Logical Closed Loop: Uncovering Object Hallucinations in Large Vision-Language Models


<!-- <div align="center"> -->
[![Arxiv](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/pdf/2402.11622.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
<!-- </div> -->

Source code for the paper "Logical Closed Loop: Uncovering Object Hallucinations in Large Vision-Language Models". We are the first to detect and mitigate object hallucinations in LVLMs by themselves through logical closed loops.

## Overview

<p align="center">
    <img src="./pics/LogicCheckGPT.jpg" width="90%" height="90%">
</p>

> Object hallucination has been an Achilles’ heel which hinders the broader applications of large vision-language models (LVLMs). Object hallucination refers to the phenomenon that the LVLMs claim non-existent objects in the image. To mitigate the object hallucinations, instruction tuning and external model-based detection methods have been proposed, which either require large-scare computational resources or depend on the detection result of external models. However, there remains an under-explored field to utilize the LVLM itself to alleviate object hallucinations. In this work, we adopt the intuition that the LVLM tends to respond logically consistently for existent objects but inconsistently for hallucinated objects. Therefore, we propose a Logical Closed Loop-based framework for Object Hallucination Detection and Mitigation, namely LogicCheckGPT. In specific, we devise logical consistency probing to raise questions with logical correlations, inquiring about attributes from objects and vice versa. Whether their responses can form a logical closed loop serves as an indicator of object hallucination. As a plug-and-play method, it can be seamlessly applied to all existing LVLMs. Comprehensive experiments conducted on three benchmarks across four LVLMs have demonstrated significant improvements brought by our method, indicating its effectiveness and generality.




## Citation
If you find this repo useful, please consider citing:
```
@article{wu2024logical,
      title={Logical Closed Loop: Uncovering Object Hallucinations in Large Vision-Language Models}, 
      author={Junfei Wu and Qiang Liu and Ding Wang and Jinghao Zhang and Shu Wu and Liang Wang and Tieniu Tan},
      journal={arXiv preprint arXiv:2402.11622},
      year={2024},
}
```

## Acknowledgment
- [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl)
- [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [QWEN-VL](https://github.com/QwenLM/Qwen-VL)
- [Woodpecker](https://github.com/BradyFU/Woodpecker)

We thank them for their great contribution to the research community of LVLMs.