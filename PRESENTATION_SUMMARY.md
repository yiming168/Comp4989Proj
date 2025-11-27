# Presentation Summary: Visual Screening of Vitamin Deficiency Symptoms

## Quick Overview for Slides

### Slide 1: Title Slide
- **Title**: Visual Screening of Vitamin Deficiency-Related Symptoms
- **Subtitle**: A Deep Learning Approach with AI-Powered Nutritional Recommendations
- **Your Name, Course, Date**

### Slide 2: Problem Statement
**Key Points:**
- Nutritional deficiencies affect millions globally
- Early visual symptom detection can prompt intervention
- Traditional methods require lab tests and medical consultation
- Need for accessible screening tools

**Visual**: Statistics on global deficiency prevalence

### Slide 3: Project Goals
**Objectives:**
1. Classify 9 deficiency types from medical images
2. Provide interpretable results (Grad-CAM)
3. Generate AI-powered food recommendations
4. Create deployable system (web, API, edge)

**Visual**: High-level architecture diagram showing data flow from images -> model -> predictions -> food suggestions

### Slide 4: Dataset Overview
**9 Classes:**
- Class_00: Unrelated Control (healthy)
- Class_01: Vitamin A - Bitot's Spots
- Class_02: Iron - Koilonychia
- Class_03: B Vitamins - Glossitis
- Class_04: B Vitamins - Angular Cheilitis
- Class_05: Vitamin C - Gum problems
- Class_06: Vitamin A - Keratosis Pilaris
- Class_07: Vitamin B3 - Pellagra
- Class_08: Zinc - Acrodermatitis

**Stats:**
- ~100 images per class
- Stratified 70/15/15 train/val/test split
- Images from public medical sources

**Visual**: One representative image per class (class examples grid)

### Slide 5: Model Architecture
**MobileNetV3-Small:**
- Pre-trained on ImageNet
- Transfer learning approach
- Custom 9-class classification head
- Input: 224x224 RGB images

**Why MobileNetV3?**
- Efficient (mobile/edge ready)
- Proven performance
- ONNX exportable

**Visual**: Architecture diagram showing backbone + new classification head

### Slide 6: Training Process
**Key Features:**
- Data augmentation (rotation, flip, color jitter)
- Class-weighted loss (handles imbalance)
- Macro-F1 score for evaluation
- 25 epochs with validation monitoring
- Best model selection based on val F1

**Hyperparameters:**
- Batch size: 16
- Learning rate: 1e-4 (Adam)
- Loss: Weighted Cross-Entropy

**Visual**: Training curves (loss, accuracy, macro-F1) over epochs

### Slide 7: Model Interpretability (Grad-CAM)
**What is Grad-CAM?**
- Visualizes which image regions model focuses on
- Gradient-weighted class activation mapping
- Overlays heatmap on original image

**Benefits:**
- Verifies model uses relevant features
- Builds user trust
- Educational value
- Debugging tool

**Visual**: Side-by-side original and Grad-CAM heatmap for each class

### Slide 8: Food Suggestions Integration
**AI-Powered Recommendations:**
- Google Gemini API integration
- Context-aware prompts
- Deficiency-specific suggestions
- Includes explanations and dietary tips

**Features:**
- 5-7 food recommendations per deficiency
- Explains why each food is beneficial
- General dietary tips
- Medical disclaimers included

**Visual**: Example food suggestion output + optional comparison table (before/after suggested diet)

### Slide 9: System Architecture
**Complete Stack:**
1. **Training**: PyTorch training pipeline
2. **Inference**: CLI tool with Grad-CAM and suggestions
3. **API**: FastAPI REST service
4. **Frontend**: React web interface
5. **Edge**: ONNX deployment for Jetson

**Visual**: End-to-end system flow diagram (from image capture -> backend -> web UI -> food suggestions)

### Slide 10: Results
**Model Performance:**
- Primary metric: Macro-F1 score
- Balanced performance across all classes
- Validation-based model selection
- Test set evaluation

**Grad-CAM Results:**
- Model focuses on relevant anatomical regions
- Eye images -> conjunctiva attention
- Nail images -> nail shape attention
- Tongue images -> tongue surface attention

**Visual**: Performance metrics table, per-class Grad-CAM examples, optional comparison table (prediction + suggestions)

### Slide 11: Demo/Features
**Key Features:**
- Image upload or camera capture
- Real-time prediction
- Interactive Grad-CAM visualization
- AI food recommendations
- Probability distribution display

**Visual**: Screenshots of the web interface (upload view, prediction view, Grad-CAM overlay, food suggestions)

### Slide 12: Ethical Considerations
**Important Disclaimers:**
- Research/educational tool only
- NOT a medical diagnostic device
- Users must consult healthcare professionals
- Clear disclaimers in all outputs

**Data Ethics:**
- Public source images
- No personal health data
- Transparent about limitations

**Visual**: Example disclaimers and "not a medical device" banner

### Slide 13: Challenges & Limitations
**Dataset:**
- Limited sample size (~100 per class)
- Web-sourced images (potential bias)
- Image quality variability

**Model:**
- Performance depends on image quality
- May struggle with ambiguous cases
- Requires clear, well-lit images

**AI Suggestions:**
- External API dependency
- Requires internet connectivity
- Should be verified with professionals

### Slide 14: Future Work
**Dataset:**
- Expand with more diverse images
- Collaborate with medical institutions
- Include different demographics

**Model:**
- Experiment with larger architectures
- Ensemble methods
- Multi-task learning

**Features:**
- Mobile app development
- Offline mode
- Multi-language support
- Clinical validation

**Visual**: Roadmap / timeline diagram

### Slide 15: Conclusion
**Key Achievements:**
- 9-class classification system
- Interpretable AI (Grad-CAM)
- AI-powered food recommendations
- Complete software stack (training, API, frontend, edge)
- Edge deployment ready (ONNX / Jetson)

**Impact:**
- Educational value
- Proof-of-concept for medical AI
- Foundation for future research

**Final Message:**
- Demonstrates potential of deep learning in medical imaging
- Maintains appropriate boundaries as a research tool
- Emphasizes responsible AI development

### Slide 16: Q&A
**Thank You!**
- Questions?
- Contact information
- Repository link

---

## Key Talking Points for Each Slide

### Slide 2 (Problem)
"Nutritional deficiencies are a global health concern. Early detection through visual screening can help, but traditional methods require lab access. Our system provides an accessible alternative."

### Slide 4 (Dataset)
"We curated a dataset of 9 classes representing different deficiency symptoms. Each class has approximately 100 images from public medical sources, split into train/validation/test sets."

### Slide 5 (Architecture)
"We use MobileNetV3-Small with transfer learning. This gives us the efficiency of a mobile model with the power of ImageNet pre-training, perfect for our use case."

### Slide 7 (Grad-CAM)
"Grad-CAM shows us exactly where the model is looking. For example, in eye images, we see attention on the conjunctiva where Bitot's spots appear. This builds trust and helps us verify the model is working correctly."

### Slide 8 (Food Suggestions)
"Beyond classification, we integrate with Google's Gemini AI to provide personalized food recommendations. The system generates context-aware prompts based on the detected deficiency and confidence score."

### Slide 9 (Architecture)
"Our system is a complete stack: training pipeline, inference tools, REST API, web interface, and edge deployment capabilities. This makes it versatile for different use cases."

### Slide 12 (Ethics)
"Critical to our project: we include clear disclaimers that this is NOT a medical device. It's a research tool for education. Users must consult healthcare professionals for actual diagnosis."

### Slide 14 (Future)
"Future work includes expanding the dataset, experimenting with larger models, and potentially clinical validation. We're also interested in mobile apps and offline capabilities."

---

## Visual Suggestions

1. **Architecture Diagram**: Show data flow from images -> model -> predictions -> suggestions
2. **Grad-CAM Examples**: Side-by-side original and heatmap images
3. **Web Interface Screenshots**: Show the complete user experience
4. **Training Curves**: Loss, accuracy, macro-F1 over epochs
5. **Class Examples**: One representative image per class
6. **System Flow**: End-to-end pipeline visualization
7. **Comparison Table**: Before/after with food suggestions

### Visuals by Slide

- Architecture diagram / system flow -> Slides 3 and 9
- Class examples (one image per class) -> Slide 4
- Training curves (loss, accuracy, macro-F1) -> Slide 6
- Grad-CAM examples -> Slides 7 and 10
- Food suggestions + comparison table (before/after diet) -> Slides 8, 10, and 11 (if time allows)
- Web interface screenshots -> Slide 11

---

## Time Allocation (15-20 minute presentation)

- Introduction/Problem: 2 min
- Dataset: 2 min
- Model Architecture: 3 min
- Training: 2 min
- Grad-CAM: 2 min
- Food Suggestions: 2 min
- Results/Demo: 3 min
- Ethics/Future: 2 min
- Q&A: 2-5 min

---

## Common Questions & Answers

**Q: Why not use a larger model like ResNet?**  
A: MobileNetV3 provides a good balance of accuracy and efficiency. It's also designed for edge deployment, which is important for our use case. We can always experiment with larger models in future work.

**Q: How accurate is the model?**  
A: We use macro-averaged F1 score as our primary metric to ensure balanced performance across all classes. The exact numbers depend on the dataset, but we prioritize balanced performance over raw accuracy.

**Q: Can this be used for diagnosis?**  
A: No. This is explicitly a research and educational tool. It's not a medical device and should not be used for diagnosis. Users must consult healthcare professionals.

**Q: How do you handle class imbalance?**  
A: We use class-weighted loss functions and evaluate with macro-F1 score, which treats all classes equally regardless of their frequency in the dataset.

**Q: What if the AI gives bad food suggestions?**  
A: The food suggestions are generated by Google's Gemini API and should be verified with healthcare professionals. We include clear disclaimers about this.

**Q: Can it work offline?**  
A: The classification model can work offline. The food suggestions require internet connectivity for the AI API. Future work could include a local language model for offline suggestions.
