import sklearn
from sklearn.metrics import classification_report
import pandas as pd
from utils import get_data_task1
from bert_tfidf import *


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

import joblib


def pred_from_transformer(test_df, reddit_df, labels2id, model_path, vectorizer_path, liwc_features=None, model_name="roberta-base"):
  vectorizer = joblib.load(vectorizer_path)
  test_dataloader, reddit_dataloader = data_batcher_evaluate(test_df, reddit_df, vectorizer, labels2id, batch_size=32, model_name=model_name, numerical_fields=liwc_features)
  model = torch.load(model_path)

  # For AA PAPER
  pred, true_label = evaluate_test(model, test_dataloader, labels2id, test_df["alg"])

  # For Reddit
  # reddit_predict = predict_labels(model, reddit_dataloader, labels2id, reddit_df["alg"])

  return pred, true_label


def print_metric_task1(pred_df, label2id, model_path, vectorizer_path,
                       source="aa_paper", batch_size=20):
    df_source = pred_df[pred_df["src"] == source]
    eval_df = get_data_task1(df_source, 20)
    idx_set = set(eval_df.first_idx.to_list() + eval_df.second_idx.to_list())
    filtered_df = pred_df[pred_df.index.isin(idx_set)]

    preds, true_label = pred_from_transformer(filtered_df, filtered_df,
                                              label2id, model_path,
                                              vectorizer_path)
    # indices = [i[0].reshape(len(i[0]), 1) for i in preds]
    # pred_labels = [i[1].reshape(len(i[1]), 1) for i in preds]

    # filtered_df["pred"] = np.vstack(pred_labels)
    filtered_df["pred"] = preds

    id2label = {id: alg for alg, id in label2id.items()}

    eval_df["pred_labels"] = eval_df[["first_idx", "second_idx"]].apply(
        lambda x: (id2label[filtered_df.loc[x[0]]['pred']],
                   id2label[filtered_df.loc[x[1]]['pred']]), axis=1)
    eval_df["pred"] = eval_df["pred_labels"].apply(
        lambda x: "Same" if x[0] == x[1] else "Not Same")

    print(classification_report(eval_df["ground_truth"], eval_df["pred"]))

    # example ground truth and predictions
    y_true = eval_df["ground_truth"].to_list()
    y_pred = eval_df["pred"].to_list()

    # create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # plot the confusion matrix
    classes = ["Same", "Not Same"]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    return eval_df

