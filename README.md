# ChartQuery-VLM

ChartQuery-VLM is a fine-tuned Vision-Language Model (VLM) built on Qwen2-VL-7B-Instruct to analyze and interpret charts. It enables automated chart-based question answering, helping users extract insights from visual data efficiently.

## Overview

This project demonstrates advanced techniques for fine-tuning large multimodal models, specifically optimized for chart analysis and question answering. We utilize parameter-efficient methods including LoRA (Low-Rank Adaptation) and 4-bit quantization to make the training process more accessible with limited computational resources.

## Key Features

The implementation includes several modern ML engineering practices:

- Fine-tuning of the Qwen2-VL-7B-Instruct model using the ChartQA dataset
- 4-bit quantization via bitsandbytes for reduced memory footprint
- Parameter-efficient fine-tuning using LoRA adapters
- Gradient checkpointing for memory-efficient training
- Supervised fine-tuning with the TRL library
- Comprehensive data processing pipeline for multimodal inputs

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ChartQuery-VLM.git
cd ChartQuery-VLM

# Install dependencies
pip install -r requirements.txt
```

Required dependencies:
- torch
- transformers
- bitsandbytes
- peft
- trl
- datasets

## Usage

### Training

To fine-tune the model on your own data:

```python
from trainer import train_model

# Configure training parameters
training_config = {
    "model_id": "Qwen/Qwen2-VL-7B-Instruct",
    "epochs": 1,
    "batch_size": 1,
    "learning_rate": 2e-5,
    "max_seq_length": 128
}

# Start training
train_model(training_config)
```

### Inference

To use the fine-tuned model for chart analysis:

```python
from model import ChartQueryVLM

# Initialize the model
model = ChartQueryVLM.from_pretrained("path/to/saved/model")

# Process an image and question
response = model.analyze_chart(
    image_path="path/to/chart.png",
    question="What was the highest value in 2023?"
)
print(response)
```

## Model Architecture

The system is built on three main components:

1. **Base Model**: Qwen2-VL-7B-Instruct, a powerful multimodal model capable of understanding both images and text
2. **LoRA Adapters**: Small trainable layers added to specific parts of the model for efficient fine-tuning
3. **Processing Pipeline**: Handles the conversion of images and questions into the format expected by the model

## Training Details

The model is trained using several optimization techniques:

- **Quantization**: 4-bit precision to reduce memory usage
- **Gradient Checkpointing**: Trades computation time for memory efficiency
- **LoRA Configuration**: 
  - Rank: 8
  - Alpha: 16
  - Dropout: 0.1
  - Target Modules: Query and Value projections

## Performance Considerations

The model requires:
- CUDA-capable GPU with at least 16GB VRAM for training
- Approximately 8GB VRAM for inference
- Python 3.8 or higher

## Example Outputs

The model can answer various types of questions about charts, such as:
- Identifying trends and patterns
- Finding maximum/minimum values
- Comparing different data points
- Analyzing relationships between variables

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this model in your research, please cite:

```bibtex
@software{chartquery_vlm,
  title = {ChartQuery-VLM: Fine-tuned Vision-Language Model for Chart Analysis},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/ChartQuery-VLM}
}
```

## Acknowledgments

- The Qwen team for the base model
- Hugging Face for their transformers library and datasets
- The developers of bitsandbytes, peft, and trl
