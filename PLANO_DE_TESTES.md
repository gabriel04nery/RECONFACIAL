# Plano de testes e evidências

- [ ] Cadastro consentido de uma pessoa, 20 capturas; guardar apenas evidência desidentificada.
- [ ] Tentar cadastrar a mesma pessoa duas vezes; verificar erro controlado.
- [ ] Tentar cadastro sem `--consentimento`; verificar bloqueio.
- [ ] Webcam com uma pessoa cadastrada; conferir reconhecimento.
- [ ] Webcam com pessoa não cadastrada que consentiu em participar; conferir rejeição.
- [ ] Webcam sem rosto e com vários rostos; observar comportamento.
- [ ] Executar `excluir` e confirmar remoção da identidade do JSON local.
- [ ] Testar câmera inexistente com `--camera 99`.
- [ ] Testes unitários com `python -m unittest discover -s tests -v`.
- [ ] Experimento público com `python -m experimentos.experimento --epocas 3`.
- [ ] Registrar métricas reais, gráficos e justificativa de escolha do modelo.
- [ ] Verificar `git status` para garantir ausência de arquivos privados.
- [ ] Gravar vídeo Parte 1: dados, hipóteses, experimentos e resultados.
- [ ] Gravar vídeo Parte 2: cadastro, reconhecimento, desconhecido, erro e exclusão.
