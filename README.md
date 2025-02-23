# MRI CycleGAN

This repository contains an implementation of CycleGAN for MRI image translation. The project focuses on unpaired image-to-image translation between different MRI modalities using CycleGAN architecture.

## Overview

CycleGAN is used to perform image translation between different MRI modalities without requiring paired training data. This makes it particularly useful for medical imaging applications where paired data can be difficult or impossible to obtain.

## Project Structure 

```.
├── mri-cyclegan.ipynb # Main implementation notebook
├── cyclegan.ipynb # CycleGAN model architecture and training
├── data/ # Directory containing MRI datasets
└── README.md # This file
```

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- numpy
- matplotlib
- PIL
- tqdm

## Installation

1. Clone this repository:

```bash
git clone https://github.com/your-username/mri-cyclegan.git
cd mri-cyclegan
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your MRI dataset and organize it in the following structure:
```
data/
├── trainA/    # Source domain training images
├── trainB/    # Target domain training images
├── testA/     # Source domain test images
└── testB/     # Target domain test images
```

2. Open and run the notebooks:
   - `mri-cyclegan.ipynb`: Contains the main implementation and training pipeline
   - `cyclegan.ipynb`: Contains the model architecture and training utilities

## Model Architecture

The CycleGAN architecture consists of:
- Two Generator networks (G_A2B and G_B2A)
- Two Discriminator networks (D_A and D_B)
- Cycle consistency loss
- Identity loss
- Adversarial loss

## Training

The model is trained using:
- Adam optimizer
- Learning rate: 0.0002
- Beta1: 0.5
- Beta2: 0.999
- Batch size: 1
- Number of epochs: 200

## Results

Include sample results and visualizations from your trained model here.

## Citation

If you use this code for your research, please cite:

```
@article{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  journal={arXiv preprint arXiv:1703.10593},
  year={2017}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

Your Name - your.email@example.com

Project Link: [https://github.com/your-username/mri-cyclegan](https://github.com/your-username/mri-cyclegan)

## Acknowledgments

- [CycleGAN Paper](https://arxiv.org/abs/1703.10593)
- [PyTorch](https://pytorch.org/)
- Any other resources or individuals you'd like to acknowledge