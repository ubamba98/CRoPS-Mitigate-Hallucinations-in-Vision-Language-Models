# CRoPS

**CRoPS: A Training-Free Hallucination Mitigation Framework for Vision-Language Models**

CRoPS is an inference-time framework that mitigates hallucinations in large vision-language models (LVLMs) without requiring any retraining or fine-tuning. It leverages contrastive decoding across multiple hallucination-inducing model variants to generate more visually grounded and reliable outputs.

This repository contains the official implementation accompanying the paper:  
**“CRoPS: A Training-Free Hallucination Mitigation Framework for Vision-Language Models”**  
Paper link: https://openreview.net/forum?id=KQSoZDPVGX

---

## 📌 Motivation

Despite strong generative capabilities, vision-language models often hallucinate objects or attributes that are not present in the image. Such hallucinations reduce trust and limit deployment in real-world applications.

CRoPS tackles this problem without modifying model weights. Instead, it contrasts the base model against multiple deliberately degraded variants, each simulating a different hallucination source, and suppresses ungrounded generations during decoding.

---

## 🧠 Method Overview

### Generalized Contrastive Decoding

Instead of contrasting against a single hallucinated model, CRoPS jointly contrasts the base model with multiple hallucinated variants during decoding. Tokens that are likely under hallucinated settings but not well supported by the original model are penalized, leading to more faithful outputs.

---

## 📊 Results

CRoPS consistently improves hallucination metrics across models and datasets:

- Significant improvements in **CHAIR** and related hallucination scores  
- Evaluated across **six benchmarks**  
- Works across **multiple LVLM families**  
- Outperforms existing training-free hallucination mitigation methods  

Refer to the paper for full quantitative and qualitative results.

---

## 📄 Citation
```bibtex
@article{anand2026crops,
  title={CRoPS: A Training-Free Hallucination Mitigation Framework for Vision-Language Models},
  author={Anand, Neeraj and Jha, Samyak and Bamba, Udbhav and Rahaman, Rahul},
  journal={Transactions on Machine Learning Research},
  year={2026}
}
```
