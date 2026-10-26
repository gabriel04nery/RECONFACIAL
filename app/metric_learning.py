import argparse
import json
import os
import re
import urllib.request
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
MODELOS = BASE / 'modelos'
GALERIA = BASE / 'galeria.json'
URLS = {
    'yunet.onnx': 'https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
    'sface.onnx': 'https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx',
}
LIMIAR_PADRAO = 0.40


def baixar_modelos():
    MODELOS.mkdir(exist_ok=True)
    for nome, url in URLS.items():
        destino = MODELOS / nome
        if destino.exists() and destino.stat().st_size > 100_000:
            continue
        temporario = destino.with_suffix('.tmp')
        print(f'Baixando {nome} (primeira execução)...')
        try:
            urllib.request.urlretrieve(url, temporario)
            if temporario.stat().st_size < 100_000:
                raise RuntimeError('Download incompleto.')
            temporario.replace(destino)
        except Exception:
            temporario.unlink(missing_ok=True)
            raise RuntimeError(f'Falha no download: {nome}. Baixe do OpenCV Zoo em {destino}.')


class Motor:
    def __init__(self):
        baixar_modelos()
        self.detector = cv2.FaceDetectorYN.create(str(MODELOS / 'yunet.onnx'), '', (320, 320), 0.85, 0.3, 5000)
        self.reconhecedor = cv2.FaceRecognizerSF.create(str(MODELOS / 'sface.onnx'), '')

    def extrair(self, imagem):
        altura, largura = imagem.shape[:2]
        self.detector.setInputSize((largura, altura))
        _, faces = self.detector.detect(imagem)
        resultado = []
        if faces is None:
            return resultado
        for face in faces:
            alinhada = self.reconhecedor.alignCrop(imagem, face)
            vetor = self.reconhecedor.feature(alinhada).flatten().astype(np.float32)
            norma = np.linalg.norm(vetor)
            if norma <= 0:
                continue
            resultado.append((face[:4].astype(int), vetor / norma))
        return resultado


def carregar_galeria():
    if not GALERIA.exists():
        return {}
    with GALERIA.open(encoding='utf8') as f:
        dados = json.load(f)
    return {nome: np.asarray(v, dtype=np.float32) for nome, v in dados.items()}


def salvar_galeria(galeria):
    temporario = GALERIA.with_suffix('.tmp')
    with temporario.open('w', encoding='utf8') as f:
        json.dump({nome: vec.astype(float).tolist() for nome, vec in galeria.items()}, f, indent=2, ensure_ascii=False)
    if os.name != 'nt':
        temporario.chmod(0o600)
    temporario.replace(GALERIA)


def prototipos(galeria):
    saida = {}
    for nome, matriz in galeria.items():
        if matriz.ndim != 2 or len(matriz) == 0:
            continue
        media = matriz.mean(axis=0)
        norma = np.linalg.norm(media)
        if norma > 0:
            saida[nome] = media / norma
    return saida


def identificar(vetor, protos, limiar):
    if not protos:
        return 'Desconhecido', 0.0
    pontuacoes = {nome: float(np.dot(vetor, proto)) for nome, proto in protos.items()}
    nome = max(pontuacoes, key=pontuacoes.get)
    score = pontuacoes[nome]
    return (nome if score >= limiar else 'Desconhecido'), score


def webcam(indice):
    cap = cv2.VideoCapture(indice)
    if not cap.isOpened():
        raise RuntimeError(f'Não foi possível abrir a câmera {indice}.')
    return cap


def cadastrar(motor, nome, quantidade, indice):
    nome = nome.strip()
    if not re.fullmatch(r'[\wÀ-ÿ][\wÀ-ÿ .-]{0,49}', nome):
        raise ValueError('Nome inválido.')
    galeria = carregar_galeria()
    if nome in galeria:
        raise ValueError('Pessoa já cadastrada.')
    cap = webcam(indice)
    vetores = []
    quadros = 0
    print('Apenas seu rosto deve estar na câmera. Q para parar.')
    try:
        while len(vetores) < quantidade:
            ok, frame = cap.read()
            if not ok:
                break
            faces = motor.extrair(frame)
            if len(faces) == 1:
                bbox, vetor = faces[0]
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 0), 2)
                if quadros % 7 == 0:
                    vetores.append(vetor)
            cv2.putText(frame, f'Cadastro {len(vetores)}/{quantidade} (apenas 1 rosto) - Q: sair', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .58, (0, 255, 0), 2)
            cv2.imshow('Cadastro', frame)
            quadros += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    if vetores:
        novas = np.stack(vetores)
        galeria[nome] = np.vstack((galeria[nome], novas)) if nome in galeria else novas
        salvar_galeria(galeria)
    print(f'{len(vetores)} embeddings armazenados para {nome}. Nenhuma foto foi salva.')


def excluir(nome):
    galeria = carregar_galeria()
    if nome not in galeria:
        raise ValueError('Pessoa não cadastrada.')
    del galeria[nome]
    salvar_galeria(galeria)
    print(f'Cadastro de {nome} excluído.')


def reconhecer(motor, limiar, indice):
    galeria = carregar_galeria()
    if not galeria:
        raise RuntimeError('Galeria vazia. Cadastre uma pessoa antes.')
    protos = prototipos(galeria)
    cap = webcam(indice)
    print(f'Reconhecimento iniciado (limiar {limiar:.2f}). Pressione Q para sair.')
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for bbox, vetor in motor.extrair(frame):
                nome, score = identificar(vetor, protos, limiar)
                x, y, w, h = bbox
                cor = (0, 220, 0) if nome != 'Desconhecido' else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), cor, 2)
                cv2.putText(frame, f'{nome} | sim. {score:.2f}', (x, max(y-8, 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, cor, 2)
            cv2.imshow('Reconhecimento - SFace', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def avaliar(motor, pasta, limiar):
    protos = prototipos(carregar_galeria())
    if not protos:
        raise RuntimeError('Cadastre alguém antes de avaliar.')
    total, corretas, falsas_aceitacoes, rejeicoes_corretas, desconhecidos = 0, 0, 0, 0, 0
    for caminho in sorted(pasta.rglob('*')):
        if caminho.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        imagem = cv2.imread(str(caminho))
        if imagem is None:
            continue
        faces = motor.extrair(imagem)
        if len(faces) != 1:
            print(f'Ignorando {caminho.name}: imagem precisa conter exatamente 1 rosto.')
            continue
        esperado = caminho.parent.name
        previsto, score = identificar(faces[0][1], protos, limiar)
        total += 1
        corretas += previsto == esperado
        if esperado == 'Desconhecido':
            desconhecidos += 1
            rejeicoes_corretas += previsto == 'Desconhecido'
            falsas_aceitacoes += previsto != 'Desconhecido'
        print(f'{caminho.name}: esperado={esperado}, previsto={previsto}, similaridade={score:.3f}')
    if total:
        print(f'Acurácia de identificação: {corretas}/{total} = {corretas/total:.1%}')
        if desconhecidos:
            print(f'Taxa de falso aceite (FAR nesta amostra): {falsas_aceitacoes}/{desconhecidos} = {falsas_aceitacoes/desconhecidos:.1%}')
            print(f'Rejeições corretas: {rejeicoes_corretas}/{desconhecidos}')
    else:
        print('Nenhuma foto válida para teste.')


def main():
    p = argparse.ArgumentParser(description='Reconhecedor por deep metric learning (SFace)')
    sub = p.add_subparsers(dest='acao', required=True)
    c = sub.add_parser('cadastrar')
    c.add_argument('--nome', required=True)
    c.add_argument('--consentimento', action='store_true', help='Confirma consentimento para cadastro biométrico local')
    c.add_argument('--quantidade', type=int, default=20)
    c.add_argument('--camera', type=int, default=0)
    r = sub.add_parser('reconhecer')
    r.add_argument('--limiar', type=float, default=LIMIAR_PADRAO)
    r.add_argument('--camera', type=int, default=0)
    a = sub.add_parser('avaliar')
    a.add_argument('--pasta', type=Path, required=True)
    a.add_argument('--limiar', type=float, default=LIMIAR_PADRAO)
    e = sub.add_parser('excluir')
    e.add_argument('--nome', required=True)
    args = p.parse_args()
    if hasattr(args, 'limiar') and not -1 <= args.limiar <= 1:
        p.error('O limiar deve estar entre -1 e 1.')
    if args.acao == 'cadastrar' and not args.consentimento:
        p.error('Confirme o consentimento informado usando --consentimento.')
    if args.acao == 'cadastrar' and args.quantidade < 1:
        p.error('A quantidade precisa ser positiva.')
    if args.acao == 'excluir':
        excluir(args.nome)
        return
    motor = Motor()
    if args.acao == 'cadastrar':
        cadastrar(motor, args.nome, args.quantidade, args.camera)
    elif args.acao == 'reconhecer':
        reconhecer(motor, args.limiar, args.camera)
    else:
        avaliar(motor, args.pasta, args.limiar)


if __name__ == '__main__':
    main()
