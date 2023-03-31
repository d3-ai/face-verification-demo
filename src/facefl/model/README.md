### Directory structure
```
.
├── README.md
├── __init__.py
├── base_net.py
├── cnn.py
├── driver.py
├── metric_learning.py
├── model.py
└── resnet.py
```
### Model architecture variation
- CNN
- ResNet18
  - Pytorch official implementation
  - replace BatchNorm layer with GroupNorm layer 
  - replace FC layer with ArcMarginProduct layer (ArcFace loss) 