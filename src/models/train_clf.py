'''
Cria histograma BoVW para cada imagem, treina LinearSVC e salva o classificador.
Fornece uma função utilitária descriptors_to_bovw_static para reuso.
'''
import os
import joblib
import numpy as np
from sklearn.svm import LinearSVC
from utils.io import unique_path, make_run_id

def descriptors_to_bovw_static(des, kmeans, K):
    if des is None or des.shape[0] == 0:
        h = np.zeros(K, dtype=np.float32)
    else:
        words = kmeans.predict(des)
        h = np.bincount(words, minlength=K).astype(np.float32)
        h /= (np.linalg.norm(h) + 1e-8)
    return h

def train_and_save(X_train, y_train, cfg):
    clf_cfg = cfg.get('classifier', {})
    C = clf_cfg.get('C', 1.0)
    run_id = cfg.get('run_id', make_run_id())
    out_dir = cfg.get('out_dir', 'results')
    os.makedirs(out_dir, exist_ok=True)
    clf = LinearSVC(C=C, dual=False, max_iter=20000, random_state=cfg.get('seed', 42))
    print("Treinando LinearSVC...")
    clf.fit(X_train, y_train)
    clf_path = unique_path(os.path.join(out_dir, f"clf_linear_{run_id}.joblib"))
    joblib.dump(clf, clf_path)
    print("Classificador salvo em:", clf_path)
    return clf_path

# export for import in run_bovw
train_and_save.__doc__ = "Treina classificador LinearSVC e retorna path para o arquivo salvo."
