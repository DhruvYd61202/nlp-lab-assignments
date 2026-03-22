# NLP Lab Assignments

This repository contains implementations of core Natural Language Processing (NLP) models, focusing on word embeddings and vector similarity. The project is structured to separate source code, interactive exploration, and executable lab assignments.

## 📁 Repository Structure

* **`src/`**: Contains the core logic and reusable Python modules for the NLP models.
* **`notebooks/`**: Jupyter notebooks for exploratory data analysis, testing out concepts, and visualizing word vectors.
* **`assignments/`**: Executable Python scripts for the final lab deliverables.
* **`outputs/`**: Generated screenshots, terminal outputs, and text files required for the physical lab record.

## 🧪 Implemented Labs

### Lab 1: Continuous Bag of Words (CBOW)
* **Description**: Implementation of the CBOW architecture from scratch.
* **Key Technologies**: Python, NumPy.
* **Objective**: Predict a target word based on its surrounding context words.

### Lab 2: Skip-gram
* **Description**: Implementation of the Skip-gram architecture from scratch.
* **Key Technologies**: Python, NumPy.
* **Objective**: Predict surrounding context words given a target word.

### Lab 3: Word Embeddings & Similarity (Gensim)
* **Description**: Generating and analyzing word vectors using the Gensim library for both CBOW and Skip-gram models.
* **Key Technologies**: Python, Gensim.
* **Objective**: Extract word vectors, calculate cosine similarity, and perform vector addition/algebra (e.g., finding semantic relationships).

## ⚙️ Prerequisites and Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/DhruvYd61202/nlp-lab-assignments
    cd nlp-lab-assignments
    ```
2.  Install the required dependencies:
    ```bash
    pip install numpy gensim jupyter
    ```

## 🚀 Usage

To run a specific lab assignment, navigate to the root directory and execute the corresponding script in the `assignments/` folder. For example:

```bash
python assignments/lab1_cbow.py