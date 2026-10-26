import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
DESCONHECIDO = -1


def dividir(labels):
    rng = np.random.default_rng(42)
    classes = np.unique(labels)
    rng.shuffle(classes)
    n = max(2, len(classes) // 5)
    ignoradas = classes[:n]
    conhecidas = classes[n:]
    treino, validacao, teste = [], [], []
    for classe in conhecidas:
        ids = np.where(labels == classe)[0]
        rng.shuffle(ids)
        treino.extend(ids[:int(len(ids) * .6)])
        validacao.extend(ids[int(len(ids)*.6):int(len(ids)*.8)])
        teste.extend(ids[int(len(ids)*.8):])
    desconhecidos = [np.where(labels == classe)[0] for classe in ignoradas]
    meio = len(desconhecidos) // 2
    validacao.extend(np.concatenate(desconhecidos[:meio]).tolist())
    teste.extend(np.concatenate(desconhecidos[meio:]).tolist())
    conhecidos = set(conhecidas.tolist())
    y = np.array([int(v) if v in conhecidos else DESCONHECIDO for v in labels])
    return np.array(treino), np.array(validacao), np.array(teste), y


def extrair_sface(imagens):
    from app.metric_learning import MODELOS, baixar_modelos
    baixar_modelos()
    modelo = cv2.FaceRecognizerSF.create(str(MODELOS / 'sface.onnx'), '')
    vetores = []
    for imagem in imagens:
        rgb = cv2.cvtColor(np.uint8(np.clip(imagem, 0, 255)), cv2.COLOR_GRAY2BGR)
        face = cv2.resize(rgb, (112, 112))
        v = modelo.feature(face).flatten().astype('float32')
        vetores.append(v / max(np.linalg.norm(v), 1e-12))
    return np.stack(vetores)


def resnet_emb(imagens, rede, dispositivo):
    import torch
    from PIL import Image
    from torchvision import transforms
    pre = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(),
         transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    saida = []
    rede.eval()
    with torch.no_grad():
        for inicio in range(0, len(imagens), 32):
            imgs = [pre(Image.fromarray(np.uint8(np.clip(im, 0, 255))).convert('RGB')) for im in imagens[inicio:inicio+32]]
            x = torch.stack(imgs).to(dispositivo)
            z = rede(x).cpu().numpy()
            z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
            saida.append(z)
    return np.vstack(saida)


def resnet_modelo():
    from torchvision import models
    from torch import nn
    rede = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    rede.fc = nn.Identity()
    return rede


def ajustar_resnet(imagens, labels, treino, validacao, epocas, dispositivo):
    import torch
    from torch import nn
    from PIL import Image
    from torchvision import transforms
    from torch.utils.data import DataLoader, Dataset
    transform = transforms.Compose([transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    neutra = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    classes = sorted(set(labels[treino]))
    codigos = {v: i for i, v in enumerate(classes)}
    class Fotos(Dataset):
        def __init__(self, ids, tr):
            self.ids, self.tr = ids, tr
        def __len__(self):
            return len(self.ids)
        def __getitem__(self, i):
            idx = self.ids[i]
            foto = Image.fromarray(np.uint8(np.clip(imagens[idx], 0, 255))).convert('RGB')
            return self.tr(foto), codigos[labels[idx]]
    torch.manual_seed(42)
    rede = resnet_modelo().to(dispositivo)
    for parametro in rede.parameters():
        parametro.requires_grad = False
    for parametro in rede.layer4.parameters():
        parametro.requires_grad = True
    class Classificador(nn.Module):
        def __init__(self, encoder, quantidade):
            super().__init__()
            self.encoder = encoder
            self.fc = nn.Linear(512, quantidade)
        def forward(self, x):
            return self.fc(self.encoder(x))
    rede = Classificador(rede, len(classes)).to(dispositivo)
    otimizador = torch.optim.Adam(filter(lambda p: p.requires_grad, rede.parameters()), lr=1e-4)
    entropia = nn.CrossEntropyLoss()
    loader = DataLoader(Fotos(treino, transform), batch_size=16, shuffle=True)
    loader_val = DataLoader(Fotos(np.array([i for i in validacao if labels[i] in codigos]), neutra), batch_size=32)
    melhor, estado = -1., None
    for epoca in range(epocas):
        rede.train()
        for x, y in loader:
            x, y = x.to(dispositivo), y.to(dispositivo)
            otimizador.zero_grad()
            loss = entropia(rede(x), y)
            loss.backward()
            otimizador.step()
        rede.eval()
        acertos, total = 0, 0
        with torch.no_grad():
            for x, y in loader_val:
                pred = rede(x.to(dispositivo)).argmax(1).cpu()
                acertos += int((pred == y).sum())
                total += len(y)
        score = acertos / max(1, total)
        print(f'Época {epoca+1}: acurácia={score:.3f}')
        if score > melhor:
            melhor = score
            estado = {chave: tensor.detach().cpu().clone() for chave, tensor in rede.encoder.state_dict().items()}
    rede.encoder.load_state_dict(estado)
    return rede.encoder.eval()


def prever(vetores, rotulos, treino, consulta, abordagem, limiar):
    X, y = vetores[treino], rotulos[treino]
    if abordagem == 'knn':
        knn = KNeighborsClassifier(n_neighbors=3, metric='cosine')
        knn.fit(X, y)
        dist, ids = knn.kneighbors(vetores[consulta], n_neighbors=3)
        similares = 1 - dist
        saida = []
        scores = []
        for pesos, vizinhos in zip(similares, ids):
            notas = {}
            for peso, idx in zip(pesos, vizinhos):
                notas[int(y[idx])] = notas.get(int(y[idx]), 0) + float(max(0, peso))
            nome = max(notas, key=notas.get)
            score = max(float(pesos[y[vizinhos] == nome].max()), -1.)
            saida.append(nome if score >= limiar else DESCONHECIDO)
            scores.append(score)
        return np.array(saida), np.array(scores)
    centros = {}
    for classe in np.unique(y):
        media = X[y == classe].mean(axis=0)
        centros[int(classe)] = media / max(np.linalg.norm(media), 1e-12)
    chaves = np.array(list(centros))
    sims = vetores[consulta] @ np.stack(list(centros.values())).T
    indices = sims.argmax(axis=1)
    scores = sims[np.arange(len(indices)), indices]
    return np.where(scores >= limiar, chaves[indices], DESCONHECIDO), scores


def metricas(y, esperado, scores):
    tp_desconhecidos = esperado == DESCONHECIDO
    conhecidos = ~tp_desconhecidos
    return dict(accuracy=float(accuracy_score(esperado, y)),
                f1_macro=float(f1_score(esperado, y, average='macro', zero_division=0)),
                far=float(np.mean(y[tp_desconhecidos] != DESCONHECIDO)) if tp_desconhecidos.any() else None,
                frr=float(np.mean(y[conhecidos] == DESCONHECIDO)) if conhecidos.any() else None,
                amostras=len(y))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epocas', type=int, default=3)
    p.add_argument('--min-fotos', type=int, default=20)
    p.add_argument('--saida', type=Path, default=RAIZ / 'resultados')
    args = p.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    random.seed(42)
    dados = fetch_lfw_people(min_faces_per_person=args.min_fotos, resize=.5, color=False, download_if_missing=True)
    imagens, labels = dados.images, dados.target
    treino, val, teste, alvo = dividir(labels)
    print('Imagens:', len(imagens), 'Treino:', len(treino), 'Validação:', len(val), 'Teste:', len(teste))
    import torch
    dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = resnet_modelo().to(dispositivo)
    resultados = []
    for modelo in ['sface', 'resnet_pre', 'resnet_ajustada']:
        if modelo == 'sface':
            vetores = extrair_sface(imagens)
        elif modelo == 'resnet_pre':
            vetores = resnet_emb(imagens, encoder, dispositivo)
        else:
            ajustada = ajustar_resnet(imagens, labels, treino, val, args.epocas, dispositivo)
            vetores = resnet_emb(imagens, ajustada, dispositivo)
        for abordagem in (['prototipos', 'knn'] if modelo == 'sface' else ['prototipos']):
            candidatos = np.arange(.1, .951, .025)
            avaliacoes = []
            for limiar in candidatos:
                pred, scores = prever(vetores, alvo, treino, val, abordagem, limiar)
                m = metricas(pred, alvo[val], scores)
                avaliacoes.append((m['f1_macro'], -m['far'], float(limiar)))
            _, _, limiar = max(avaliacoes)
            pred_val, scores_val = prever(vetores, alvo, treino, val, abordagem, limiar)
            m_val = metricas(pred_val, alvo[val], scores_val)
            pred, scores = prever(vetores, alvo, treino, teste, abordagem, limiar)
            m = metricas(pred, alvo[teste], scores)
            linha = {'modelo': modelo, 'comparacao': abordagem, 'threshold_validacao': round(limiar, 3),
                     'f1_validacao': m_val['f1_macro'], 'accuracy_validacao': m_val['accuracy'], **m}
            resultados.append(linha)
            print(linha)
            np.savez_compressed(args.saida / f'scores_{modelo}_{abordagem}.npz',
                scores=scores, conhecido=alvo[teste] != DESCONHECIDO)
    with (args.saida / 'metricas.csv').open('w', newline='', encoding='utf8') as f:
        escritor = csv.DictWriter(f, fieldnames=resultados[0].keys())
        escritor.writeheader()
        escritor.writerows(resultados)
    config = {'dataset': 'LFW sklearn', 'min_fotos': args.min_fotos, 'seed': 42,
        'treino': len(treino), 'validacao': len(val), 'teste': len(teste),
        'epocas': args.epocas, 'device': dispositivo, 'torch': torch.__version__,
        'opencv': cv2.__version__, 'numpy': np.__version__}
    (args.saida / 'configuracao.json').write_text(json.dumps(config, indent=2))
    vencedor = max(resultados, key=lambda row: row['f1_validacao'])
    print('Melhor pela validação:', vencedor['modelo'], vencedor['comparacao'])
    print('Resultados salvos.')


if __name__ == '__main__':
    main()
