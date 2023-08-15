# NeVA: NVIDIA's Visual Question Answering Transformer

[NeVA is a powerful and versatile Visual Question Answering model by NVIDIA](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/playground/models/neva/bias). Neva builds upon the open-source LLaMA model, integrating it with an NVIDIA-trained GPT model to offer state-of-the-art performance. 


[![GitHub issues](https://img.shields.io/github/issues/kyegomez/neva)](https://github.com/kyegomez/neva/issues) 
[![GitHub forks](https://img.shields.io/github/forks/kyegomez/neva)](https://github.com/kyegomez/neva/network) 
[![GitHub stars](https://img.shields.io/github/stars/kyegomez/neva)](https://github.com/kyegomez/neva/stargazers) [![GitHub license](https://img.shields.io/github/license/kyegomez/neva)](https://github.com/kyegomez/neva/blob/master/LICENSE)
[![Share on Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Share%20%40kyegomez/neva)](https://twitter.com/intent/tweet?text=Excited%20to%20introduce%20neva,%20the%20all-new%20Multi-Modal%20model%20with%20the%20potential%20to%20revolutionize%20automation.%20Join%20us%20on%20this%20journey%20towards%20a%20smarter%20future.%20%23Neva%20%23Multi-Modal&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fneva)
[![Share on Facebook](https://img.shields.io/badge/Share-%20facebook-blue)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fneva)
[![Share on LinkedIn](https://img.shields.io/badge/Share-%20linkedin-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fneva&title=Introducing%20neva%2C%20the%20All-New%20Multi-Modal%20Model&summary=neva%20is%20the%20next-generation%20Multi-Modal%20model%20that%20promises%20to%20transform%20industries%20with%20its%20intelligence%20and%20efficiency.%20Join%20us%20to%20be%20a%20part%20of%20this%20revolutionary%20journey%20%23Neva%20%23Multi-Modal&source=)
![Discord](https://img.shields.io/discord/999382051935506503)
[![Share on Reddit](https://img.shields.io/badge/-Share%20on%20Reddit-orange)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fneva&title=Exciting%20Times%20Ahead%20with%20neva%2C%20the%20All-New%20Multi-Modal%20Model%20%23Neva%20%23Multi-Modal) [![Share on Hacker News](https://img.shields.io/badge/-Share%20on%20Hacker%20News-orange)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fneva&t=Exciting%20Times%20Ahead%20with%20neva%2C%20the%20All-New%20Multi-Modal%20Model%20%23Neva%20%23Multi-Modal)
[![Share on Pinterest](https://img.shields.io/badge/-Share%20on%20Pinterest-red)](https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fneva&media=https%3A%2F%2Fexample.com%2Fimage.jpg&description=neva%2C%20the%20Revolutionary%20Multi-Modal%20Model%20that%20will%20Change%20the%20Way%20We%20Work%20%23Neva%20%23Multi-Modal)
[![Share on WhatsApp](https://img.shields.io/badge/-Share%20on%20WhatsApp-green)](https://api.whatsapp.com/send?text=I%20just%20discovered%20neva,%20the%20all-new%20Multi-Modal%20model%20that%20promises%20to%20revolutionize%20automation.%20Join%20me%20on%20this%20exciting%20journey%20towards%20a%20smarter%20future.%20%23Neva%20%23Multi-Modal%0A%0Ahttps%3A%2F%2Fgithub.com%2Fkyegomez%2Fneva)


## Appreciation
* All the creators in Agora, [Join Agora](https://discord.gg/qUtxnK2NMf) the community of AI engineers changing the world with their creations.
* LucidRains for inspiring me to devote myself to open source AI



---

## Installation

To integrate NeVA into your Python environment, you can install it via pip:

```bash
pip install nevax
```
---

## Usage

```python
import torch
from nevax import Neva

#usage
img = torch.randn(1, 3, 256, 256)
caption_tokens = torch.randint(0, 4)

model = Neva()
output = model(img, caption_tokens)
```

## Description

At a high level, NeVA utilizes a frozen Hugging Face CLIP model to encode images. These encoded images are projected to text embedding dimensions, concatenated with the embeddings of the given prompt, and subsequently passed through the language model. The training process comprises two main stages:

1. **Pretraining**: Only the projection layer is trained with the language model kept frozen. This stage uses image-caption pairs for training.
2. **Finetuning**: Both the language model and the projection layer are trained. This stage utilizes synthetic instruction data generated with GPT4.

## Model Specifications

- **Architecture Type**: Transformer
- **Network Architecture**: GPT + CLIP
- **Model versions**: 8B, 22B, 43B

## Input & Output

- **Input Format**: RGB Image + Text
- **Input Parameters**: temperature, max output tokens, quality, toxicity, humor, creativity, violence, helpfulness, not_appropriate
- **Output Format**: Text

## Integration and Compatibility

- **Supported Hardware Platforms**: Hopper, Ampere/Turing
- **Supported Operating Systems**: Linux
- **Runtime**: N/A

## Training & Fine-tuning Data

**Pretraining Dataset**:
- **Link**: [CC-3M](#)
- **Description**: The dataset comprises CC3M images and captions, refined to 595,000 samples.
- **License**: [COCO](#), [CC-3M](#), [BLIP](#)

**Finetuning Dataset**:
- **Link**: Synthetic data produced by GPT4
- **Description**: The dataset, with 158,000 samples, was synthetically generated by GPT4. It encompasses a blend of short question answers, detailed image descriptions, and higher-level reasoning questions.
- **License**: [CC-BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/)

## Inference

- **Engine**: Triton
- **Test Hardware**: Other

## References

- [Visual Instruction Tuning paper](#)
- [Blog](#)
- [Codebase](#)
- [Demo](#)

## Licensing

This project is licensed under the [MIT License](/LICENSE) license.


# Citation