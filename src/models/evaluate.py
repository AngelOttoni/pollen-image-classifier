'''
Avalia um classificador salvo e gera:
- accuracy_vs_K csv/plot (quando K sweep é usado)
- classification_report txt
- confusion matrix png
- csv e cópia das imagens mal classificadas
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils.io import unique_path
import csv
import shutil

def evaluate_and_save(test_paths, y_test, extractor, kmeans, clf, classes, out_dir, run_id):
    os.makedirs(out_dir, exist_ok=True)
    K = kmeans.n_clusters
    # build X_test
    X_test = []
    for p in test_paths:
        des = extractor.extract_from_path(p)
        h = descriptors_to_bovw(des, kmeans, K)
        X_test.append(h)
    X_test = np.vstack(X_test)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia (avaliacao): {acc:.4f}")

    # classification report and confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=classes, digits=4)

    # save report
    report_txt = unique_path(os.path.join(out_dir, f"classification_report_K{K}_{run_id}.txt"))
    with open(report_txt, 'w') as f:
        f.write(f"RUN_ID: {run_id}\n")
        f.write(f"accuracy: {acc:.6f}\n\n")
        f.write("confusion_matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nclassification_report:\n")
        f.write(report)
    print("Relatório salvo em:", report_txt)

    # save confusion matrix png
    cm_png = unique_path(os.path.join(out_dir, f"confusion_matrix_K{K}_{run_id}.png"))
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', aspect='auto', cmap='Blues')
    plt.title(f"Matriz de Confusão (K={K})")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes)
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(cm_png)
    plt.close()
    print("Matriz de confusão salva em:", cm_png)

    # misclassified csv + copy images
    mis_idx = np.where(y_test != y_pred)[0]
    mis_csv = unique_path(os.path.join(out_dir, f"misclassified_K{K}_{run_id}.csv"))
    with open(mis_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "filepath", "true_label", "pred_label"])
        for i in mis_idx:
            fp = test_paths[i]
            true = classes[int(y_test[i])]
            pred = classes[int(y_pred[i])]
            writer.writerow([i, fp, true, pred])
    print("CSV de mal classificados salvo em:", mis_csv)

    mis_dir = unique_path(os.path.join(out_dir, f"misclassified_K{K}_{run_id}"))
    os.makedirs(mis_dir, exist_ok=True)
    for i in mis_idx:
        src = test_paths[i]
        base = os.path.basename(src)
        true = classes[int(y_test[i])]
        pred = classes[int(y_pred[i])]
        dst_name = f"{i:03d}_true-{true}_pred-{pred}_{base}"
        dst = os.path.join(mis_dir, dst_name)
        try:
            shutil.copy(src, dst)
        except Exception as e:
            print(f"Falha ao copiar {src} -> {dst}: {e}")
    print("Imagens mal classificadas copiadas para:", mis_dir)

    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "report_path": report_txt,
        "cm_png": cm_png,
        "mis_csv": mis_csv,
        "mis_dir": mis_dir
    }

# local helper used here to compute histogram
def descriptors_to_bovw(des, kmeans, K):
    if des is None or des.shape[0] == 0:
        h = np.zeros(K, dtype=np.float32)
    else:
        words = kmeans.predict(des)
        h = np.bincount(words, minlength=K).astype(np.float32)
        h /= (np.linalg.norm(h) + 1e-8)
    return h
