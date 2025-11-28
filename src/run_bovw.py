'''
Orquestrador de alto nível para o pipeline.
Funções públicas: build_vocab(cfg), train_clf(cfg), evaluate_run(cfg), run_all(cfg)
'''
import os
import yaml
import joblib
from pathlib import Path
from features.sift_features import SIFTExtractor
from clustering.build_vocab import build_kmeans
from models.train_clf import train_classifier
from models.evaluate import evaluate_and_save
from utils.io import gather_paths, make_run_id, unique_path, save_yaml
import numpy as np

def _ensure_out_dir(cfg):
    out = cfg.get('out_dir', 'results')
    os.makedirs(out, exist_ok=True)
    return out

def build_vocab(cfg):
    # gather train paths
    root = cfg['root']
    classes = cfg['classes']
    train_paths, _ = gather_paths(root, classes, cfg.get('split_train', 'train'))
    extractor = SIFTExtractor(cfg.get('descriptor', {}))
    des_list = []
    for p in train_paths:
        des = extractor.extract_from_path(p)
        if des is not None and des.shape[0] > 0:
            des_list.append(des)
    if len(des_list) == 0:
        raise RuntimeError("Nenhum descritor encontrado para construir vocabulário.")
    des_all = np.vstack(des_list)
    # subsample if requested
    sample = cfg.get('kmeans', {}).get('sample_descriptors')
    if sample is not None and des_all.shape[0] > sample:
        rng = np.random.RandomState(cfg.get('seed', 42))
        idx = rng.choice(des_all.shape[0], sample, replace=False)
        des_for_kmeans = des_all[idx]
    else:
        des_for_kmeans = des_all
    model_path = build_kmeans(des_for_kmeans, cfg)
    return model_path

def train_clf(cfg):
    root = cfg['root']
    classes = cfg['classes']
    K = cfg['kmeans']['K']
    kmeans_path = cfg.get('kmeans_model')
    if kmeans_path is None:
        raise RuntimeError("kmeans_model não está definido no config. Rode build_vocab ou forneça kmeans_model.")
    kmeans = joblib.load(kmeans_path)
    extractor = SIFTExtractor(cfg.get('descriptor', {}))

    train_paths, y_train = gather_paths(root, classes, cfg.get('split_train', 'train'))
    X_train = []
    for p in train_paths:
        des = extractor.extract_from_path(p)
        h = train_classifier.descriptors_to_bovw_static(des, kmeans, K)
        X_train.append(h)
    X_train = np.vstack(X_train)
    clf_path = train_classifier.train_and_save(X_train, y_train, cfg)
    return clf_path

def evaluate_run(cfg):
    # expects clf_model and kmeans_model in config
    root = cfg['root']
    classes = cfg['classes']
    kmeans_path = cfg.get('kmeans_model')
    clf_path = cfg.get('clf_model')
    if kmeans_path is None or clf_path is None:
        raise RuntimeError("kmeans_model e clf_model devem estar setados no config para avaliar.")
    kmeans = joblib.load(kmeans_path)
    clf = joblib.load(clf_path)
    extractor = SIFTExtractor(cfg.get('descriptor', {}))
    test_paths, y_test = gather_paths(root, classes, cfg.get('split_test', 'test'))
    out_dir = _ensure_out_dir(cfg)
    run_id = cfg.get('run_id', make_run_id())
    evaluate_and_save(test_paths, y_test, extractor, kmeans, clf, classes, out_dir, run_id)
    return True

def run_all(cfg):
    # orchestrates build_vocab -> train -> evaluate, and writes models paths back to cfg copy
    out_dir = _ensure_out_dir(cfg)
    run_id = cfg.get('run_id', make_run_id())
    cfg['run_id'] = run_id
    # 1) build vocab
    kmeans_path = build_vocab(cfg)
    cfg['kmeans_model'] = kmeans_path
    # 2) train classifier
    clf_path = train_clf(cfg)
    cfg['clf_model'] = clf_path
    # 3) evaluate
    evaluate_run(cfg)
    # save used config for reproducibility
    cfg_path = unique_path(os.path.join(out_dir, f'config_used_{run_id}.yaml'))
    save_yaml(cfg, cfg_path)
    print("Run completa. Config salva em:", cfg_path)
