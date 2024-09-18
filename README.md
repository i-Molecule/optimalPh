# OptimalPh

Enzymes are widely used in biotechnology due to their ability to catalyze chemical reactions: food making, laundry, pharmaceutics, textile, brewing─all these areas benefit from utilizing various enzymes. Proton concentration (pH) is one of the key factors that define the enzyme's functioning and efficiency. Usually, there is only a narrow range of pH values where the enzyme is active. This is a common problem in biotechnology to design an enzyme with optimal activity in a given pH range. A large part of this task can be completed in silico, by predicting the optimal pH of designed candidates. The success of such computational methods critically depends on the available data. In this study, we developed a language-model-based approach to predict the optimal pH range from the enzyme sequence. We used different splitting strategies based on sequence similarity, protein family annotation, and enzyme classification to validate the robustness of the proposed approach. The derived machine-learning models demonstrated high accuracy across proteins from different protein families and proteins with lower sequence similarities compared with the training set. The proposed method is fast enough for the high-throughput virtual exploration of protein space for the search for sequences with desired optimal pH levels.

This repository contains everything to train and run OphPred models to predict optimal enzyme pH from its sequence.
You can also run the OphPred model through the Constructor Research Platform without coding [here](https://research.constructor.tech/platform/public/project/optimalph).

Please check our publication [Approaching Optimal pH Enzyme Prediction with Large Language Models](https://pubs.acs.org/doi/full/10.1021/acssynbio.4c00465) in ACS Synthetic Biology.


# Environment
To create the environment with conda (tested on Ubuntu 22.04):
```bash
conda create -n esm_env
conda activate esm_env

conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install fire fair-esm xgboost -y
pip install fairscale
```

# Inference

You can download the model weights from the Constructor Research Platform project [here](https://research.constructor.tech/platform/public/project/optimalph) or from google drive [here](https://drive.google.com/drive/folders/1CRKzq3DGFjlTH3MzTZ8a3AQaB23YrfCn?usp=sharing)


```bash
python3 code/predict.py --input_csv sequences.csv --seq_col sequence --model_fname weights/model_xgboost --output_csv sequences_scored.csv
```

# Train

To obtain embeddings from ESM-2 protein LLM (takes ~50Mb for 10k sequences):

```bash
python3 esm_embeddings/ml/dataloader_v1.py --input_csv data/datasets/brenda_new.csv --seq_col sequence --target_col mean_pH --output_emb data/embeddings/brenda_new_emb.npy
```

Train your model.
JSON file containing info about your model should look as follows:
```json
{
    "1": {
        "type": "knn",
        "params": {
            "n_neighbors": 8
        }
    }
}
```

```bash
python3 code/final_train.py --models experiments/best_models.jso --data data/embeddings/brenda_new_emb.npy --output_dir weights/ --prefix model --csv_input data/datasets/brenda_new.csv
```
