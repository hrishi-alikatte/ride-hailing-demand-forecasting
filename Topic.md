# Project: LPE-STGTN Model Reproduction

## Goal
Reproduce the taxi demand prediction model from the paper.

## Model Overview

**What it does:**  
Predicts taxi/ride-hailing demand across a city

**Key feature:**  
Captures both local patterns and global trends

**Input:**  
Historical demand data (last 3 hours)

**Output:**  
Next 3 hours of demand predictions

## Main Parts to Build

### 1. Data Processing
- Format taxi trip data into 15-minute intervals  
- Add day/time information  
- Normalize the data  

### 2. Local Module (Dynamic Patterns)
- Dynamic graph generator: Creates graphs that change over time  
- AFT-local: A simplified attention mechanism  
- GRU + GCN: Captures time and space patterns together  

### 3. Global Module (Regular Patterns)
- Distance graph: Nearby zones influence each other  
- OD flow graph: Zones with lots of trips between them  
- Attention fusion: Combines both graphs intelligently  

### 4. Fusion & Prediction
- Combine local and global features  
- Make final predictions  

## Datasets
- **NYC Taxi:** Yellow taxi trips in Manhattan  
- **NYC Ride-Hailing:** Uber/Lyft trips in Manhattan  
- **Beijing Taxi:** Taxi trips in Beijing  



> Start simple — build a basic version first, then add complexity.
