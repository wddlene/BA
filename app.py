from flask import Flask, request, render_template_string 
import joblib 
import numpy as np 
import torch
import torch.nn as nn
import os

app = Flask(__name__) 

# Define diagnosis thresholds
DIAGNOSIS_THRESHOLDS = {
    'low_alg': 0.3,      # Low probability of ALG
    'medium_alg': 0.7,   # Moderate probability of ALG
    'high_alg': 0.7      # High probability of ALG
}

CLINICAL_RECOMMENDATIONS = {
    'low_alg': {
        'title': 'LOW PROBABILITY OF ALG SYNDROME',
        'subtitle': 'Higher probability of Biliary Atresia',
        'icon': 'fa-lungs',
        'color': 'info',
        'badge_class': 'ba-badge',
        'recommendation': 'This transfer learning-based differential diagnosis suggests higher probability of Biliary Atresia. However, comprehensive evaluation is needed for definitive diagnosis.',
        'actions': [
            'Consider surgical exploration (Kasai procedure)',
            'Perform intraoperative cholangiogram',
            'Liver biopsy for definitive diagnosis',
            'Multidisciplinary team consultation',
            'Monitor for complications of BA'
        ]
    },
    'medium_alg': {
        'title': 'MODERATE PROBABILITY OF ALG SYNDROME', 
        'subtitle': 'Further investigation needed',
        'icon': 'fa-question-circle',
        'color': 'warning',
        'badge_class': 'medium-badge',
        'recommendation': 'This transfer learning screening tool indicates moderate probability of ALG Syndrome vs Biliary Atresia - additional investigations are required.',
        'actions': [
            'Genetic testing for ALG syndrome mutations',
            'Comprehensive metabolic workup',
            'Consider alternative diagnoses',
            'Specialist pediatric hepatology referral',
            'Family genetic counseling'
        ]
    },
    'high_alg': {
        'title': 'HIGH PROBABILITY OF ALG SYNDROME',
        'subtitle': 'Lower probability of Biliary Atresia',
        'icon': 'fa-dna', 
        'color': 'success',
        'badge_class': 'alg-badge',
        'recommendation': 'This transfer learning differential diagnosis result indicates high probability of ALG Syndrome. Avoid unnecessary surgical interventions until confirmed.',
        'actions': [
            'Confirm with genetic testing',
            'Avoid unnecessary surgical exploration',
            'Metabolic disease management',
            'Genetic counseling for family',
            'Long-term developmental follow-up'
        ]
    }
}

# Transfer Learning Model Definition
class TransferLearningModel(nn.Module):
    def __init__(self, input_dim, base_model=None):
        super(TransferLearningModel, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Neural network architecture (same as train1.py)
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(self.device)
    
    def forward(self, x):
        return self.model(x)

# HTML Template for ALG vs BA Differential Diagnosis
HTML_TEMPLATE = ''' 
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ALG Syndrome vs Biliary Atresia Differential Diagnosis Tool</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #2196F3;      /* Blue for BA */
            --secondary: #4CAF50;    /* Green for ALG */
            --warning: #FF9800;      /* Orange for uncertain */
            --danger: #F44336;       /* Red for alerts */
            --success: #4CAF50;      /* Green for success */
            --info: #00BCD4;         /* Cyan for info */
            --light: #E3F2FD;        /* Light blue background */
            --dark: #1976D2;         /* Dark blue */
            --alg-color: #4CAF50;    /* ALG green */
            --ba-color: #2196F3;     /* BA blue */
            --alg-light: #E8F5E9;    /* Light ALG green */
            --ba-light: #E3F2FD;     /* Light BA blue */
            --card-shadow: 0 8px 30px rgba(33, 150, 243, 0.15);
            --transition: all 0.3s ease;
            --border-radius: 12px;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
            font-family: 'Nunito', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: var(--dark);
            line-height: 1.6;
            min-height: 100vh;
            padding-bottom: 50px;
            font-size: 1.05rem;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            background-attachment: fixed;
        }

        .navbar {
            background: linear-gradient(90deg, var(--ba-color) 0%, var(--alg-color) 100%);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            padding: 0.8rem 0;
        }

        .main-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: calc(100vh - 200px);
            width: 100%;
        }

        .content-wrapper {
            width: 100%;
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 20px;
        }

        .diagnosis-highlight {
            background: linear-gradient(135deg, var(--alg-light) 0%, var(--ba-light) 100%);
            border: 2px solid var(--primary);
            border-radius: var(--border-radius);
            padding: 1rem;
            margin-bottom: 1.5rem;
            text-align: center;
            font-weight: 700;
            color: var(--dark);
            font-size: 1.1rem;
            width: 100%;
        }

        .disclaimer-box {
            background: linear-gradient(135deg, #FFF3CD 0%, #FFECB5 100%);
            border: 2px solid var(--warning);
            border-radius: var(--border-radius);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            text-align: center;
            font-weight: 600;
            color: #856404;
            width: 100%;
        }

        .transfer-learning-badge {
            background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
            border: 2px solid var(--primary);
            border-radius: var(--border-radius);
            padding: 0.8rem;
            margin-bottom: 1rem;
            text-align: center;
            font-weight: 600;
            color: var(--dark);
            width: 100%;
        }

        .card {
            border-radius: var(--border-radius);
            box-shadow: var(--card-shadow);
            margin-bottom: 1.8rem;
            border: none;
            overflow: hidden;
            transition: var(--transition);
            background-color: white;
            border: 1px solid rgba(33, 150, 243, 0.1);
            width: 100%;
        }

        .card:hover {
            box-shadow: 0 12px 35px rgba(33, 150, 243, 0.2);
            transform: translateY(-5px);
        }

        .card-header {
            background: linear-gradient(90deg, rgba(33, 150, 243, 0.15) 0%, rgba(76, 175, 80, 0.15) 100%);
            border-bottom: 1px solid rgba(0, 0, 0, 0.05);
            padding: 1.2rem 1.8rem;
            font-weight: 700;
            color: var(--dark);
            font-size: 1.25rem;
            font-family: 'Poppins', sans-serif;
            text-align: center;
        }

        /* Diagnosis probability scale */
        .diagnosis-scale {
            position: relative;
            height: 20px;
            background: linear-gradient(90deg, var(--ba-color) 0%, var(--warning) 50%, var(--alg-color) 100%);
            border-radius: 10px;
            margin: 2rem auto;
            width: 90%;
            max-width: 600px;
        }

        .current-marker {
            position: absolute;
            top: -10px;
            width: 4px;
            height: 40px;
            background: var(--dark);
            transform: translateX(-50%);
        }

        .current-label {
            position: absolute;
            top: -35px;
            left: 50%;
            transform: translateX(-50%);
            background: var(--dark);
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
            white-space: nowrap;
        }

        .prediction-badge {
            font-size: 1.4rem;
            padding: 1.0rem 2.0rem;
            border-radius: 50px;
            font-weight: 800;
            letter-spacing: 0.5px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
            transition: var(--transition);
            font-family: 'Poppins', sans-serif;
            display: inline-block;
            margin: 0.8rem auto;
            text-align: center;
        }

        .ba-badge {
            background: linear-gradient(90deg, rgba(33, 150, 243, 0.15) 0%, rgba(66, 165, 245, 0.15) 100%);
            color: var(--ba-color);
            border: 2px solid var(--ba-color);
        }

        .medium-badge {
            background: linear-gradient(90deg, rgba(255, 152, 0, 0.15) 0%, rgba(255, 193, 7, 0.15) 100%);
            color: var(--warning);
            border: 2px solid var(--warning);
        }

        .alg-badge {
            background: linear-gradient(90deg, rgba(76, 175, 80, 0.15) 0%, rgba(102, 187, 106, 0.15) 100%);
            color: var(--alg-color);
            border: 2px solid var(--alg-color);
        }

        .action-item {
            padding: 0.8rem 1rem;
            margin-bottom: 0.5rem;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.7);
            border-left: 4px solid;
            transition: var(--transition);
            width: 100%;
        }

        .action-item:hover {
            transform: translateX(5px);
            background: white;
        }

        .action-item.ba {
            border-left-color: var(--ba-color);
        }

        .action-item.medium {
            border-left-color: var(--warning);
        }

        .action-item.alg {
            border-left-color: var(--alg-color);
        }

        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(227, 242, 253, 0.92);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 9999;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
            backdrop-filter: blur(8px);
        }

        .loading-overlay.active {
            opacity: 1;
            visibility: visible;
        }

        .spinner {
            width: 60px;
            height: 60px;
            border: 5px solid rgba(33, 150, 243, 0.2);
            border-top: 5px solid var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .compact-form {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.2rem;
            width: 100%;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        .section-title {
            text-align: center;
            margin-bottom: 1.5rem;
            color: var(--dark);
            font-weight: 600;
            border-bottom: 2px solid var(--primary);
            padding-bottom: 0.5rem;
        }

        .feature-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 0.8rem;
            margin-top: 0.5rem;
            border-left: 3px solid var(--primary);
        }

        .feature-title {
            font-weight: 600;
            color: var(--dark);
            margin-bottom: 0.3rem;
        }

        .feature-desc, .feature-range {
            font-size: 0.85rem;
            color: #666;
            margin-bottom: 0.2rem;
        }

        .result-card {
            text-align: center;
        }

        .result-icon {
            margin-bottom: 1rem;
        }

        .actions-list {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .probability-display {
            display: flex;
            justify-content: space-around;
            margin: 2rem 0;
        }

        .probability-box {
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            width: 45%;
        }

        .alg-probability {
            background: var(--alg-light);
            border: 2px solid var(--alg-color);
        }

        .ba-probability {
            background: var(--ba-light);
            border: 2px solid var(--ba-color);
        }

        .probability-value {
            font-size: 2rem;
            font-weight: 700;
            margin: 0.5rem 0;
        }

        @media (max-width: 768px) {
            .compact-form {
                grid-template-columns: 1fr;
            }
            
            .content-wrapper {
                padding: 0 15px;
            }
            
            .card-header {
                padding: 1rem;
                font-size: 1.1rem;
            }
            
            .prediction-badge {
                font-size: 1.1rem;
                padding: 0.8rem 1.5rem;
            }
            
            .probability-display {
                flex-direction: column;
                align-items: center;
            }
            
            .probability-box {
                width: 90%;
                margin-bottom: 1rem;
            }
        }

        @media (min-width: 1200px) {
            .content-wrapper {
                max-width: 1400px;
            }
        }
    </style>
</head>
<body>
    <!-- Loading animation -->
    <div class="loading-overlay" id="loadingOverlay">
        <div class="spinner"></div>
        <div class="loading-text">Running transfer learning differential diagnosis analysis, please wait...</div>
    </div>

    <nav class="navbar navbar-expand-lg navbar-dark">
        <div class="container-fluid">
            <a class="navbar-brand d-flex align-items-center" href="/">
                <i class="fas fa-brain fa-2x me-3"></i>
                <div>
                    <h4 class="mb-0">ALG Syndrome vs Biliary Atresia Differential Diagnosis</h4>
                    <small class="opacity-85">Transfer Learning AI Tool - NOT for Definitive Diagnosis</small>
                </div>
            </a>
        </div>
    </nav>

    <div class="main-container">
        <div class="content-wrapper">
            {{ content | safe }}
        </div>
    </div>

    <footer class="footer mt-5">
        <div class="container-fluid text-center py-3" style="background: linear-gradient(90deg, var(--ba-color) 0%, var(--alg-color) 100%); color: white;">
            <p class="mb-1">
                <i class="fas fa-robot me-1"></i>
                © 2024 ALG vs BA Differential Diagnosis Transfer Learning Tool
            </p>
            <small class="opacity-85">Transfer Learning Model: RF-based Neural Network | Six-feature differential diagnosis panel</small>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Form input validation
        document.querySelectorAll('input[type="number"]').forEach(input => {
            input.addEventListener('input', function() {
                const min = parseFloat(this.min);
                const max = parseFloat(this.max);
                const value = parseFloat(this.value);
                if (!isNaN(min) && value < min) {
                    this.value = min;
                } else if (!isNaN(max) && value > max) {
                    this.value = max;
                }
            });
        });

        // Show loading animation on form submit
        document.querySelector('form')?.addEventListener('submit', function() {
            document.getElementById('loadingOverlay').classList.add('active');
        });

        // Center all content on page load
        document.addEventListener('DOMContentLoaded', function() {
            const mainContainer = document.querySelector('.main-container');
            if (mainContainer) {
                mainContainer.style.display = 'flex';
            }
        });
    </script>
</body>
</html>
'''

# Home page form for ALG vs BA differential diagnosis - 使用原始值输入
HOME_PAGE = '''
<div class="row justify-content-center">
    <div class="col-12">
        <div class="diagnosis-highlight">
            <i class="fas fa-stethoscope me-2"></i>
            <strong>DIFFERENTIAL DIAGNOSIS TOOL:</strong> This AI-powered transfer learning tool distinguishes between ALG Syndrome (Alagille Syndrome) and Biliary Atresia in infants with cholestatic jaundice.
        </div>
        
        <div class="disclaimer-box">
            <i class="fas fa-exclamation-triangle me-2"></i>
            <strong>TRANSFER LEARNING DIFFERENTIAL DIAGNOSIS DISCLAIMER:</strong> This AI-powered tool provides probability estimates using transfer learning technology. 
            It is NOT a definitive diagnostic tool. Final diagnosis requires comprehensive clinical evaluation, imaging, and genetic testing.
        </div>
        
        <div class="transfer-learning-badge">
            <i class="fas fa-brain me-2"></i>
            <strong>TRANSFER LEARNING TECHNOLOGY:</strong> This tool uses a neural network initialized with Random Forest feature importance for enhanced differential diagnosis accuracy.
        </div>
        
        <div class="card mx-auto" style="max-width: 1400px;">
            <div class="card-header">
                <h5 class="m-0 d-flex align-items-center justify-content-center">
                    <i class="fas fa-network-wired me-2"></i>Transfer Learning Differential Diagnosis Panel - Enter Raw Values
                </h5>
            </div>
            <div class="card-body">
                <div class="row mb-3">
                    <div class="col-12">
                        <div class="alert alert-primary d-flex align-items-center mb-3">
                            <i class="fas fa-robot fa-2x me-3"></i>
                            <div>
                                <h5 class="mb-1">AI-Powered ALG Syndrome vs Biliary Atresia Differential Diagnosis</h5>
                                <p class="mb-0">This <strong>transfer learning model</strong> uses <strong>6 clinical and imaging features</strong> with neural network architecture for differential diagnosis. Please enter <strong>raw (unscaled) values</strong> below.</p>
                            </div>
                        </div>
                    </div>
                </div>
                <form action="/predict" method="post" id="diagnosisForm">
                    <div class="compact-form">
                        <!-- Column 1: Clinical Features -->
                        <div class="form-group">
                            <h5 class="section-title">Clinical & Imaging Features</h5>
                            
                            <!-- Acholic Stools -->
                            <div class="mb-3">
                                <label class="form-label">
                                    <i class="fas fa-poop"></i>Acholic Stools (0 or 1)
                                </label>
                                <input type="number" class="form-control form-control-lg feature-input" 
                                       name="Acholic_stools" min="0" max="1" step="1" 
                                       placeholder="Enter 0 or 1 (0=no, 1=yes)" required>
                                <div class="feature-card">
                                    <div class="feature-title">
                                        <i class="fas fa-microscope"></i>Transfer Learning Feature
                                    </div>
                                    <p class="feature-desc">Presence of pale, clay-colored stools</p>
                                    <p class="feature-range">Binary: 0 = Absent, 1 = Present</p>
                                </div>
                            </div>
                            
                            <!-- Abnormal GB Morphology -->
                            <div class="mb-3">
                                <label class="form-label">
                                    <i class="fas fa-procedures"></i>Abnormal Gallbladder Morphology (0 or 1)
                                </label>
                                <input type="number" class="form-control form-control-lg feature-input" 
                                       name="Abnormal_GB_morphology" min="0" max="1" step="1" 
                                       placeholder="Enter 0 or 1 (0=normal, 1=abnormal)" required>
                                <div class="feature-card">
                                    <div class="feature-title">
                                        <i class="fas fa-microscope"></i>Transfer Learning Feature
                                    </div>
                                    <p class="feature-desc">Ultrasound finding of gallbladder abnormalities</p>
                                    <p class="feature-range">Binary: 0 = Normal, 1 = Abnormal</p>
                                </div>
                            </div>
                            
                            <!-- Glisson Sheath Thickening -->
                            <div class="mb-3">
                                <label class="form-label">
                                    <i class="fas fa-layer-group"></i>Glisson Sheath Thickening (0 or 1)
                                </label>
                                <input type="number" class="form-control form-control-lg feature-input" 
                                       name="Glisson_Sheath_Thickening" min="0" max="1" step="1" 
                                       placeholder="Enter 0 or 1 (0=no, 1=yes)" required>
                                <div class="feature-card">
                                    <div class="feature-title">
                                        <i class="fas fa-microscope"></i>Transfer Learning Feature
                                    </div>
                                    <p class="feature-desc">Imaging finding of peri-portal thickening</p>
                                    <p class="feature-range">Binary: 0 = Absent, 1 = Present</p>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Column 2: Quantitative Measurements -->
                        <div class="form-group">
                            <h5 class="section-title">Quantitative Measurements (Raw Values)</h5>
                            
                            <!-- Length -->
                            <div class="mb-3">
                                <label class="form-label">
                                    <i class="fas fa-ruler"></i>Length (cm)
                                </label>
                                <input type="number" class="form-control form-control-lg feature-input" 
                                       name="Length" min="0" max="10" step="0.1" 
                                       placeholder="Enter length in cm" required>
                                <div class="feature-card">
                                    <div class="feature-title">
                                        <i class="fas fa-microscope"></i>Transfer Learning Feature
                                    </div>
                                    <p class="feature-desc">Patient length/height</p>
                                    <p class="feature-range">Typical infant range: 1.5-3.4 cm</p>
                                </div>
                            </div>
                            
                            <!-- LSM -->
                            <div class="mb-3">
                                <label class="form-label">
                                    <i class="fas fa-wave-square"></i>LSM (Liver Stiffness, kPa)
                                </label>
                                <input type="number" class="form-control form-control-lg feature-input" 
                                       name="LSM" min="0" max="50" step="0.01" 
                                       placeholder="Enter LSM in kPa" required>
                                <div class="feature-card">
                                    <div class="feature-title">
                                        <i class="fas fa-microscope"></i>Transfer Learning Feature
                                    </div>
                                    <p class="feature-desc">Liver stiffness measurement by FibroScan</p>
                                    <p class="feature-range">Typical range: 3-5.5 kPa</p>
                                </div>
                            </div>
                            
                            <!-- GGT -->
                            <div class="mb-3">
                                <label class="form-label">
                                    <i class="fas fa-vial"></i>GGT (U/L)
                                </label>
                                <input type="number" class="form-control form-control-lg feature-input" 
                                       name="GGT" min="0" max="5000" step="0.1" 
                                       placeholder="Enter GGT in U/L" required>
                                <div class="feature-card">
                                    <div class="feature-title">
                                        <i class="fas fa-microscope"></i>Transfer Learning Feature
                                    </div>
                                    <p class="feature-desc">Gamma-glutamyl transferase level</p>
                                    <p class="feature-range">Typical range: 0-178 U/L</p>
                                </div>
                            </div>
                            
                            <!-- Model Information -->
                            <div class="disclaimer-box mt-4">
                                <h6 class="text-center"><i class="fas fa-network-wired me-2"></i>Transfer Learning Model Information</h6>
                                <div class="row text-center">
                                    <div class="col-md-6">
                                        <p class="mb-1"><strong>Architecture:</strong> 6-64-32-1 Neural Network</p>
                                        <p class="mb-1"><strong>Technology:</strong> RF-initialized Transfer Learning</p>
                                    </div>
                                    <div class="col-md-6">
                                        <p class="mb-1"><strong>Features:</strong> 6 clinical/imaging parameters</p>
                                        <p class="mb-0"><strong>Note:</strong> Enter raw values - AI handles normalization automatically</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="d-grid mt-4">
                        <button type="submit" class="btn btn-primary btn-lg py-3" style="max-width: 500px; margin: 0 auto;">
                            <i class="fas fa-brain me-2"></i>Run Transfer Learning Differential Diagnosis
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </div>
</div>
'''

# Result page for ALG vs BA differential diagnosis
RESULT_PAGE = '''
<div class="row justify-content-center">
    <div class="col-12">
        <div class="diagnosis-highlight">
            <i class="fas fa-stethoscope me-2"></i>
            <strong>DIFFERENTIAL DIAGNOSIS RESULT:</strong> This AI-powered transfer learning analysis distinguishes between ALG Syndrome and Biliary Atresia.
        </div>
        
        <div class="disclaimer-box">
            <i class="fas fa-robot me-2"></i>
            <strong>TRANSFER LEARNING DIFFERENTIAL DIAGNOSIS - NOT DEFINITIVE:</strong> This AI-powered result indicates probability estimates only. Clinical evaluation and confirmatory tests are required.
        </div>
        
        <div class="transfer-learning-badge">
            <i class="fas fa-network-wired me-2"></i>
            <strong>TRANSFER LEARNING ANALYSIS COMPLETE:</strong> Neural network differential diagnosis based on 6 clinical/imaging features with RF initialization.
        </div>
        
        <div class="card result-card mx-auto" style="max-width: 1200px;">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5 class="m-0 d-flex align-items-center justify-content-center w-100">
                    <i class="fas fa-chart-line me-2"></i>Transfer Learning Differential Diagnosis Results
                </h5>
                <a href="/" class="btn btn-outline-primary position-absolute" style="right: 20px;">
                    <i class="fas fa-redo me-1"></i>New Analysis
                </a>
            </div>
            <div class="card-body">
                <div class="text-center mb-4">
                    <div class="result-icon">
                        <i class="fas {{ risk_info.icon }} fa-3x text-{{ risk_info.color }}"></i>
                    </div>
                    <h3 class="mb-2">{{ risk_info.title }}</h3>
                    <h5 class="mb-3 text-muted">{{ risk_info.subtitle }}</h5>
                    <div class="d-flex justify-content-center align-items-center mb-3">
                        <span class="prediction-badge {{ risk_info.badge_class }}">
                            <i class="fas {{ risk_info.icon }} me-2"></i>
                            {{ diagnosis }}
                        </span>
                    </div>
                    
                    <!-- Probability Display -->
                    <div class="probability-display">
                        <div class="probability-box alg-probability">
                            <h6>ALG Syndrome Probability</h6>
                            <div class="probability-value" style="color: var(--alg-color);">
                                {{ alg_percentage }}%
                            </div>
                            <p class="mb-0">Probability: {{ alg_probability }}</p>
                        </div>
                        
                        <div class="probability-box ba-probability">
                            <h6>Biliary Atresia Probability</h6>
                            <div class="probability-value" style="color: var(--ba-color);">
                                {{ ba_percentage }}%
                            </div>
                            <p class="mb-0">Probability: {{ ba_probability }}</p>
                        </div>
                    </div>
                    
                    <!-- Diagnosis Scale Visualization -->
                    <div class="diagnosis-scale">
                        <div class="current-marker" style="left: {{ alg_percentage }}%;"></div>
                        <div class="current-label" style="left: {{ alg_percentage }}%;">
                            {{ alg_percentage }}% ALG
                        </div>
                    </div>
                    <div class="d-flex justify-content-between mt-4 text-muted fw-medium" style="max-width: 600px; margin: 0 auto;">
                        <span style="color: var(--ba-color);">BA Likely &lt;30% ALG</span>
                        <span style="color: var(--warning);">Uncertain 30-70% ALG</span>
                        <span style="color: var(--alg-color);">ALG Likely &gt;70% ALG</span>
                    </div>
                </div>

                <!-- Differential Diagnosis Management Recommendations -->
                <div class="alert alert-{{ risk_info.color }} mt-3 mx-auto" style="max-width: 1000px;">
                    <h5 class="alert-heading d-flex align-items-center justify-content-center">
                        <i class="fas {{ risk_info.icon }} me-2"></i>
                        {{ risk_info.title }} - Clinical Management Recommendations
                    </h5>
                    <p class="mb-3 text-center">{{ risk_info.recommendation}}</p>
                    <hr>
                    <h6 class="mb-2 text-center">Recommended Diagnostic & Management Actions:</h6>
                    <div class="actions-list mx-auto" style="max-width: 800px;">
                        {% for action in risk_info.actions %}
                        <div class="action-item {{ risk_level }}">
                            <i class="fas fa-check-circle me-2 text-{{ risk_info.color }}"></i>
                            {{ action }}
                        </div>
                        {% endfor %}
                    </div>
                </div>
                
                <!-- Final Transfer Learning Disclaimer -->
                <div class="disclaimer-box mx-auto" style="max-width: 1000px;">
                    <h6 class="text-center"><i class="fas fa-robot me-2"></i>Transfer Learning Model Disclaimer</h6>
                    <p class="mb-2 text-center">This AI differential diagnosis tool uses transfer learning technology (RF-initialized neural network) with six clinical and imaging features to distinguish between ALG Syndrome and Biliary Atresia.</p>
                    <p class="mb-0 text-center"><strong>This is a differential diagnosis aid only - definitive diagnosis requires:</strong> Clinical examination, imaging studies, genetic testing (for ALG), and surgical exploration with cholangiogram (for BA).</p>
                </div>

                <div class="text-center mt-4">
                    <a href="/" class="btn btn-primary btn-lg py-3 px-5">
                        <i class="fas fa-redo me-2"></i>Perform New Differential Diagnosis
                    </a>
                </div>
            </div>
        </div>
    </div>
</div>
'''

ERROR_PAGE = '''
<div class="row justify-content-center">
    <div class="col-lg-8">
        <div class="card border-danger mx-auto" style="max-width: 800px;">
            <div class="card-header bg-danger text-white d-flex align-items-center justify-content-center">
                <i class="fas fa-exclamation-triangle me-2"></i>
                <h5 class="m-0">Transfer Learning Analysis Error</h5>
            </div>
            <div class="card-body text-center">
                <div class="alert alert-danger">
                    <h4 class="alert-heading d-flex align-items-center justify-content-center">
                        <i class="fas fa-bug me-2"></i>AI Differential Diagnosis Calculation Error
                    </h4>
                    <p>{{ error_message }}</p>
                    <hr>
                    <p class="mb-0">Please verify input values and try again.</p>
                </div>
                <div class="text-center mt-3">
                    <a href="/" class="btn btn-danger btn-lg py-2 px-4">
                        <i class="fas fa-arrow-left me-2"></i>Return to Diagnosis Form
                    </a>
                </div>
            </div>
        </div>
    </div>
</div>
'''

# Transfer Learning Predictor class for ALG vs BA differential diagnosis
class ALGBATransferLearningDiagnoser:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            self.scaler = joblib.load("alg_ba_scaler.pkl")
        except FileNotFoundError:
            raise FileNotFoundError("Scaler file 'alg_ba_scaler.pkl' not found. Please run train1.py first.")
        
        # Define feature order (6 features from ALG_TL_AS.ipynb)
        self.required_features = ['Acholic_stools', 'Abnormal_GB_morphology', 'Glisson_Sheath_Thickening', 'Length', 'LSM', 'GGT']
        
        # Load transfer learning model
        self.model = TransferLearningModel(len(self.required_features))
        try:
            self.model.load_state_dict(torch.load("alg_ba_transfer_model.pth", 
                                                map_location=self.device, 
                                                weights_only=True))
        except TypeError:
            # Fallback for older PyTorch versions
            self.model.load_state_dict(torch.load("alg_ba_transfer_model.pth", 
                                                map_location=self.device))
        except FileNotFoundError:
            raise FileNotFoundError("Model file 'alg_ba_transfer_model.pth' not found. Please run train1.py first.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Feature descriptions with typical ranges for validation
        self.feature_descriptions = {
            'Acholic_stools': 'Acholic Stools', 
            'Abnormal_GB_morphology': 'Abnormal Gallbladder Morphology',
            'Glisson_Sheath_Thickening': 'Glisson Sheath Thickening',
            'Length': 'Patient Length (cm)',
            'LSM': 'Liver Stiffness Measurement (kPa)',
            'GGT': 'Gamma-glutamyl Transferase (U/L)'
        }
        
        # Typical ranges for validation (loose validation)
        self.feature_ranges = {
            'Acholic_stools': (0, 1),
            'Abnormal_GB_morphology': (0, 1),
            'Glisson_Sheath_Thickening': (0, 1),
            'Length': (40, 100),
            'LSM': (1, 50),
            'GGT': (0, 2000)
        }
    
    def validate_input(self, input_data):
        """Validate input data against reasonable ranges"""
        for i, feat in enumerate(self.required_features):
            val = input_data[i]
            min_val, max_val = self.feature_ranges[feat]
            
            if val < min_val * 0.5 or val > max_val * 2:
                print(f"Warning: {self.feature_descriptions[feat]} value {val} is outside typical range ({min_val}-{max_val})")
            
            # For binary features, ensure they are 0 or 1
            if feat in ['Acholic_stools', 'Abnormal_GB_morphology', 'Glisson_Sheath_Thickening']:
                if val not in [0, 1]:
                    raise ValueError(f"{self.feature_descriptions[feat]} must be 0 or 1, got {val}")

    def get_risk_level(self, alg_probability):
        """Determine diagnosis level based on ALG probability thresholds"""
        if alg_probability < DIAGNOSIS_THRESHOLDS['low_alg']:
            return 'low_alg'  # More likely BA
        elif alg_probability < DIAGNOSIS_THRESHOLDS['high_alg']:
            return 'medium_alg'  # Uncertain/indeterminate
        else:
            return 'high_alg'  # More likely ALG

# Initialize transfer learning diagnoser
try:
    diagnoser = ALGBATransferLearningDiagnoser()
    print("ALG vs BA Transfer Learning Diagnoser initialized successfully")
except Exception as e:
    print(f"Error initializing diagnoser: {e}")
    diagnoser = None

# Application routes
@app.route('/')
def home():
    """Home page with transfer learning differential diagnosis input form"""
    return render_template_string(HTML_TEMPLATE, content=HOME_PAGE)

@app.route('/predict', methods=['POST'])
def predict():
    """Transfer learning differential diagnosis prediction endpoint"""
    try:
        # Check if diagnoser is initialized
        if diagnoser is None:
            raise ValueError("Diagnosis service is not available. Please check if model files are present.")
        
        # Collect input data - RAW VALUES
        input_data = []
        feature_names = ['Acholic_stools', 'Abnormal_GB_morphology', 'Glisson_Sheath_Thickening', 'Length', 'LSM', 'GGT']
        
        for feature in feature_names:
            value = request.form.get(feature, '').strip()
            if not value:
                raise ValueError(f"{feature} value is required")
            try:
                # Convert to appropriate type
                if feature in ['Acholic_stools', 'Abnormal_GB_morphology', 'Glisson_Sheath_Thickening']:
                    # Binary features - accept 0 or 1
                    val = int(float(value))
                    if val not in [0, 1]:
                        raise ValueError(f"{feature} must be 0 or 1")
                    input_data.append(float(val))
                else:
                    # Continuous features
                    input_data.append(float(value))
            except ValueError as e:
                raise ValueError(f"Invalid value for {feature}: '{value}'. {str(e)}")
        
        print(f"Received raw input data: {input_data}")
        
        # Validate input (loose validation)
        diagnoser.validate_input(input_data)
        
        # Convert to numpy array and reshape for scaler
        input_array = np.array([input_data])
        
        # Use scaler to normalize the data (same as training)
        scaled_data = diagnoser.scaler.transform(input_array)
        print(f"Scaled data: {scaled_data}")
        
        # Calculate ALG probability using transfer learning model
        with torch.no_grad():
            input_tensor = torch.tensor(scaled_data, dtype=torch.float32).to(diagnoser.device)
            alg_proba = diagnoser.model(input_tensor).cpu().numpy()[0][0]
        
        ba_proba = 1 - alg_proba
        alg_percentage = round(alg_proba * 100, 1)
        ba_percentage = round(ba_proba * 100, 1)
        
        # Determine diagnosis stratification
        risk_level = diagnoser.get_risk_level(alg_proba)
        risk_info = CLINICAL_RECOMMENDATIONS[risk_level]
        
        # Determine diagnosis text
        if risk_level == 'low_alg':
            diagnosis = "Higher Probability of Biliary Atresia (BA)"
        elif risk_level == 'medium_alg':
            diagnosis = "Uncertain - Further Investigation Needed"
        else:
            diagnosis = "Higher Probability of ALG Syndrome"
        
        # Render differential diagnosis results
        result_content = render_template_string(
            RESULT_PAGE,
            diagnosis=diagnosis,
            alg_probability=round(alg_proba, 4),
            ba_probability=round(ba_proba, 4),
            alg_percentage=alg_percentage,
            ba_percentage=ba_percentage,
            risk_level=risk_level,
            risk_info=risk_info
        )
        
        return render_template_string(HTML_TEMPLATE, content=result_content)
        
    except Exception as e:
        # Print detailed error for debugging
        print(f"Error in diagnosis: {str(e)}")
        print(f"Form data: {dict(request.form)}")
        
        # Error handling
        error_content = render_template_string(ERROR_PAGE, error_message=str(e))
        return render_template_string(HTML_TEMPLATE, content=error_content)

if __name__ == '__main__':
    # Check if required files exist
    required_files = ['alg_ba_transfer_model.pth', 'alg_ba_scaler.pkl']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
            print(f"WARNING: {file} not found.")
    
    if missing_files:
        print("Please run train1.py first to generate the required model files.")
    
    # Start Flask application
    print("Starting Flask application for ALG vs BA Differential Diagnosis...")
    app.run(debug=True, host='0.0.0.0', port=5000)