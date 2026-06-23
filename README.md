# Spacecraft Telemetry Fault Detection

## Overview

This project implements a **Deep Learning-based Anomaly Detection System** for spacecraft telemetry data using **LSTM Autoencoders**. The system is designed to detect faults and anomalies in spacecraft systems by analyzing telemetry data from 50 different channels in real-time.

## Problem Statement

Spacecraft systems generate continuous telemetry data from multiple sensors. Early detection of faults or anomalies is critical for mission success and safety. This project addresses this challenge by building an unsupervised learning model that can identify anomalous patterns in telemetry data without requiring labeled fault examples.

## Dataset

- **Total Records**: 7,364,160 telemetry readings
- **Time Period**: January 1, 2000 - January 7, 2000
- **Channels**: 50 sensor channels capturing various spacecraft parameters
- **Temporal Resolution**: 1-minute intervals
- **Fault Labels**: 3,589 labeled fault events across multiple channels with time ranges
- **Data Quality**: No missing values in the telemetry data

### Data Structure

```
combined_spacecraft_telemetry.csv
├── datetime: timestamp of the measurement
├── channel_12 to channel_76: normalized sensor readings (values 0-1)
└── [50 features total]

labels.csv
├── ID: identifier for the fault event
├── Channel: affected channel
├── StartTime: when fault began
└── EndTime: when fault ended
```

## Methodology

### 1. **Data Preprocessing**
- Load telemetry data in chunks to handle 7M+ records
- Drop datetime column for feature engineering
- Scale features to [0, 1] range using MinMaxScaler for normalization

### 2. **Sequence Creation**
- Create sliding windows of 60-minute sequences from the time-series data
- Each sequence contains 60 timesteps × 50 features = 3,000 input dimensions
- Results in 99,940 sequences for training

### 3. **LSTM Autoencoder Architecture**

```
Input Layer
    ↓
LSTM Encoder (64 units, tanh activation)
    ↓
RepeatVector (repeats encoded vector)
    ↓
LSTM Decoder (64 units, tanh activation, return_sequences=True)
    ↓
TimeDistributed Dense Layer (reconstruct 50 features)
    ↓
Output Layer (reconstructed sequence)
```

**Model Parameters**: 65,714 trainable parameters

### 4. **Training Configuration**
- **Train-Test Split**: 80% training (79,952 sequences), 20% testing (19,988 sequences)
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Batch Size**: 128
- **Epochs**: 10 with early stopping (patience=3)
- **Validation**: Used for monitoring during training

### 5. **Anomaly Detection**
- Calculate reconstruction error (MSE) for each test sequence
- Set detection threshold at 95th percentile of error distribution
- Flag sequences with error above threshold as anomalies
- **Detection Rate**: ~5% of test data identified as anomalies

## Results

- **Training Loss**: Reduced from 0.0178 to 0.0080 over training
- **Validation Loss**: Converged at ~0.0082
- **Reconstruction Threshold**: Dynamically computed from error percentiles
- **Detected Anomalies**: Identified in test set using 95th percentile threshold

## Key Features

✅ **Unsupervised Learning**: No labeled training data required  
✅ **Real-Time Capable**: Processes 50-channel telemetry sequentially  
✅ **Scalable**: Handles millions of records through batch processing  
✅ **Interpretable**: Reconstruction error provides anomaly confidence  
✅ **Deep Learning**: LSTM networks capture temporal dependencies  

## Technologies & Libraries

- **Python 3**
- **TensorFlow/Keras**: Deep learning framework
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Preprocessing and model evaluation
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization
- **Google Colab**: Cloud computing environment

## File Structure

```
Spacecraft_Telemetry_fault_Detection/
├── DL_Project.ipynb          # Main Jupyter notebook with complete pipeline
└── README.md                  # This file
```

## Usage

### Prerequisites
```bash
pip install tensorflow pandas scikit-learn numpy matplotlib
```

### Running the Model

1. **Upload Data**: Place telemetry data in Google Drive
   ```
   /MyDrive/Spacecraft_Project_Data/combined_spacecraft_telemetry.csv
   /MyDrive/Spacecraft_Project_Data/labels.csv
   ```

2. **Mount Google Drive** (if using Colab):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Execute Notebook**: Run all cells in `DL_Project.ipynb`

4. **Interpret Results**:
   - Reconstruction error distribution plot
   - Anomaly detection threshold visualization
   - Detected anomalies count and ratio

## Model Performance Insights

### Strengths
- Successfully captures normal telemetry patterns
- Identifies genuine anomalies through reconstruction error spikes
- Efficient training with ~256KB model size
- Low false positive rate due to adaptive threshold

### Potential Improvements
- Increase sequence window length for longer-term patterns
- Experiment with deeper architectures or attention mechanisms
- Multi-channel fault correlation analysis
- Real-time streaming pipeline implementation
- Ensemble methods combining multiple anomaly detection approaches

## Anomaly Detection Strategy

The model uses **reconstruction error as an anomaly score**:
- **Normal behavior**: Low reconstruction error (model accurately reconstructs sequences)
- **Anomalous behavior**: High reconstruction error (model struggles to reconstruct abnormal patterns)
- **Threshold**: 95th percentile of training error distribution

## Applications

- 🛰️ Satellite health monitoring
- 🚀 Launch vehicle telemetry analysis
- ⚙️ Mission-critical system fault detection
- 📡 Real-time anomaly alerting
- 🔍 Historical fault pattern discovery

## Future Enhancements

1. **Multi-variate Anomaly Detection**: Correlate faults across channels
2. **Temporal Pattern Mining**: Identify fault sequences and precursors
3. **Adaptive Thresholds**: Dynamic threshold adjustment based on drift
4. **Explainability**: Feature importance analysis for detected anomalies
5. **Production Deployment**: REST API for real-time predictions
6. **Hybrid Approaches**: Combine LSTM with isolation forests or autoencoders

## Citation

If you use this project, please cite:
```
@article{spacecraft_telemetry_2024
  title={LSTM Autoencoder for Spacecraft Telemetry Anomaly Detection},
  author={Pawar, Ashish},
  year={2024}
}
```

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to:
- Report issues and bugs
- Suggest improvements
- Submit pull requests
- Share datasets

## Contact

For questions or collaborations:
- GitHub: [@imashishgpawar](https://github.com/imashishgpawar)

---

**Last Updated**: June 2024  
**Status**: Active Development
