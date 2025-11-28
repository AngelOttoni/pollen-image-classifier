'''
Treina KMeans ou MiniBatchKMeans com descritores empilhados.
Retorna o path do modelo salvo.
'''
import os
import joblib
from sklearn.cluster import MiniBatchKMeans, KMeans
from utils.io import unique_path, make_run_id

def build_kmeans(descriptors, cfg):
    k_cfg = cfg.get('kmeans', {})
    K = int(k_cfg.get('K', 100))
    use_minibatch = k_cfg.get('method', 'MiniBatchKMeans').lower().startswith('mini')
    run_id = cfg.get('run_id', make_run_id())
    out_dir = cfg.get('out_dir', 'results')
    os.makedirs(out_dir, exist_ok=True)
    model_name = f"kmeans_K{K}_{run_id}.joblib"
    model_path = unique_path(os.path.join(out_dir, model_name))

    if descriptors.shape[0] < K:
        raise ValueError(f"Número de descritores ({descriptors.shape[0]}) < K ({K})")

    if use_minibatch:
        kmeans = MiniBatchKMeans(n_clusters=K, random_state=cfg.get('seed', 42),
                                 batch_size=1000, n_init=10)
    else:
        kmeans = KMeans(n_clusters=K, random_state=cfg.get('seed', 42), n_init=10, max_iter=300)

    print(f"Treinando {'MiniBatchKMeans' if use_minibatch else 'KMeans'} K={K} com {descriptors.shape[0]} descritores...")
    kmeans.fit(descriptors)
    joblib.dump(kmeans, model_path)
    print("KMeans salvo em:", model_path)
    return model_path
