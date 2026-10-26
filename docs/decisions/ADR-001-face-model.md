# ADR 001 — Escolha do modelo

**Estado:** provisório.

**Decisão atual:** YuNet + SFace na aplicação; comparar com ResNet-18 pré-treinada e ajustada no benchmark LFW.

**Justificativa provisória:** modelo leve e disponível no OpenCV, sem treinamento local para uso da câmera. A escolha final depende do desempenho medido na Parte 1.

**Consequência:** se ResNet ajustada for superior, migrar o pipeline ou justificar concretamente o custo de inferência e de manutenção.
