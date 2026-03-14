# Deep learning  project

## 1. Project overview

In this project, students will **reimplement the architecture of a published deep learning research paper**, reproduce its experimental results as closely as possible, and critically compare their findings with those reported by the original authors.

The goal is not only to obtain similar performance metrics, but also to deeply understand:

* the model architecture,
* the training procedure,
* the role of design choices and hyperparameters,
* and the challenges of reproducibility in modern deep learning research.

---

## 2. Learning objectives

By the end of this project, students should be able to:

* Read and interpret a deep learning research paper
* Translate a model description into a working implementation
* Reproduce experimental results using a modern deep learning framework
* Analyze discrepancies between reproduced and reported results
* Communicate technical findings clearly and rigorously

---

## 3. Topics

The papers to be reimplemented **are provided by the instructor (Check the folder Topics)**.

Each paper has been selected to:

* Contain a clearly defined deep learning architecture
* Include experimental results suitable for reproduction
* Illustrate common challenges in reproducibility (e.g., missing details, sensitivity to hyperparameters)

Each group shall work  on the assigned paper unless explicit permission is granted to change the paper. 

---

## 4. Project tasks

### 4.1 Architecture reimplementation

Students must:

* Reimplement the model architecture described in the paper
* Clearly justify any assumptions or design decisions not explicitly specified
* Follow the paper’s notation and layer structure as closely as possible

If the authors provide official code, students **may consult it for clarification**, but the core implementation must be written independently.
It is worth mentioning that copying the code provided by the authors is deemed as an academic misconduct and shall results in disciplinary actions.

---

### 4.2 Training and experimental setup

Students must reproduce the experimental setup, including:

* Dataset(s)
* Preprocessing steps
* Loss functions
* Optimization algorithms
* Learning rate schedules
* Regularization techniques

Any deviations from the original setup must be explicitly documented and justified.

---

### 4.3 Evaluation and results

Students should:

* Evaluate their model using the same metrics reported in the paper
* Compare reproduced results with the original results
* Report results in tables and/or plots matching the paper’s format where possible

---

### 4.4 Analysis and discussion

A critical analysis is required, including:

* Differences between reproduced and reported results
* Possible reasons for performance gaps
* Sensitivity to hyperparameters or random initialization
* Insights gained about the architecture’s strengths and weaknesses

---

## 5. Deliverables

### 5.1 Code

* Clean, well‑documented code
* Reproducible training and evaluation scripts
* Clear instructions for running the experiments

### 5.2 Written report

The report must be written in PDF format (preferably using LateX) and should include:

1. Introduction and paper summary
2. Model architecture description
3. Implementation details
4. Experimental setup
5. Results and comparison
6. Analysis and discussion
7. Conclusion and lessons learned

### 5.3 Presentation

Students may be asked to give a short presentation summarizing:

* The paper
* The reimplementation
* Key challenges
* Main findings

---

## 6. Evaluation Criteria

| Criterion                                  | Weight |
| ------------------------------------------ | ------ |
| Correctness of architecture implementation | 30%    |
| Experimental rigor and reproducibility     | 30%    |
| Quality of analysis and discussion         | 15%    |
| Code clarity and organization              | 15%    |
| Report quality and presentation            | 10%    |

---

## 7. Academic integrity and reproducibility

* All sources (papers, codebases, libraries, gen AI) must be properly cited
* Any use of existing implementations must be clearly acknowledged
* Experiments must be reproducible using the provided code and instructions

---

## 8. Suggested tools

* Frameworks: PyTorch, TensorFlow, JAX
* Experiment tracking: w&b, TensorBoard
* Version control: Git

---

## 9. Timeline and project milestones

- Paper selection and proposal: Before 24-02-2026
- Architecture implementation: 25-02-2026 to 15-03-2026
- Training and experimentation: 16-03-2026 to 09-04-2026
- Analysis and report writing: 10-04-2026 to 23-04-2026
- Final submission / presentations: By 28-04-2026
