### To do list:
- [x] NYU VPN
- [x] NYU GPU
- [x] official GEO-BLEU and DTW
- [x] 2023 task1: LP-BERT
- [x] 2024 LlaMa solution
- [ ] ...
- [ ] ...


---
### Conda env:
LP-BERT --> lpbert
Llama-Mob --> llm_mob


### Git: large .csv files are excluded in .gitignore
git status

git add .

git commit -m "update"

git push


# Human Mobility Trajectory Prediction Research Summary

## Related Work Overview

### 1. Cross-city-aware Spatiotemporal BERT
- **Key Innovation**: City embedding integration
- **Method**: Calculate location probabilities for 30/90-minute windows with exponential decay over time
- **Implementation**: Create a reference lookup table for direct position prediction mapping

### 2. MoE BERT (Mixture of Experts BERT)
- **Core Concept**: Automatic pattern learning with expert generation
- **Architecture**: Automatically generate specialized experts for different mobility patterns
- **Learning**: Train expert weights for different prediction modes using gating networks

### 3. Personalized and On-device Trajectory Mobility Prediction
- **Data Structure**: Transform each prediction time point into an independent data row
- **Features**: 
  - Time2Vec temporal features
  - Historical location information
  - Weekend/weekday indicators
  - Structured like a timetable format

### 4. Trajectory Prediction Using Random Forests with Time Decay and Periodic Features
- **Feature Engineering**:
  - **Location Stay Frequency**: Total stay counts at each coordinate (x,y)
  - **Daily Statistics**: Stay counts at specific locations per day
  - **Time-slot Statistics**: Stay counts by time periods
  - **Day-of-week Statistics**: Stay patterns across different weekdays
  - **Weekend/Weekday Statistics**: Behavioral differences between weekends and workdays
- **Method**: Time-decay weighting applied to three cities separately using Random Forest

### 5. Tuning LLaMA for Trajectory Prediction
- **Task Transformation**: Convert 15-day human mobility prediction into Q&A dialogue format with JSON I/O
- **Data Sampling Strategy**:
  - Instead of using all 113,600 trajectories, sample selectively:
  - Randomly sample 1,000 trajectories from City A (100,000 users)
  - Use all 17,600 trajectories from City B (80% of 22,000 users)
  - Total: 18,600 trajectories (16% of total data)
- **Fine-tuning Details**:
  - LoRA (Low-Rank Adaptation) with rank=16
  - 4-bit quantization
  - Batch size=1, gradient accumulation=4, learning rate=2e-3, 3 epochs

## Our Proposed Solution

### Statistical Foundation
- Establish baseline patterns for weekday vs. weekend behavior

### Per-User Input Enhancement
For each input user in the LLM:

1. **Add User Attributes**:
   - User's location stay frequency patterns
   - User's city identifier
   - Delay time information

2. **Add Statistical Tables**:
   - Time-window statistics (30-min for weekdays, 90-min for weekends)
   - Historical pattern summaries

3. **Loss Function Enhancement**:
   - Integrate GEOBLEU loss function for trajectory-specific evaluation

### Ablation Studies

1. **City Selection Strategy**:
   - Test necessity of City A: Should each city train independently?
   - Compare with approach similar to "24 LLaMA attributes": Use A+B data to predict B, C, D

2. **Data Sampling Rates**:
   - Experiment with different sampling percentages after city selection

3. **LLM Architecture Comparison**:
   - Model families: Qwen, LLaMA, Gemma, DeepSeek
   - Parameter sizes: 1-10B parameters
   - Same architecture with different parameter counts
```
experiment_matrix = {
    "Qwen Scale": ["0.6B", "1.7B", "4B", "8B", "32B"],
    "Thinking mode": ["enable_thinking=True", "enable_thinking=False"],
    "In all": "5×2 = 10 combinations"
}
```
   - Investigation of deep reasoning capabilities

### Experimental Design
- Control variables systematically
- Run comprehensive experiments across all configurations
