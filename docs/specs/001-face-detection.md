# SPEC 001 — Detecção

**Requisito:** receber quadro da câmera e detectar faces com YuNet.

**Aceitação:** zero faces não causa falha; uma face retorna bounding box; múltiplas faces são detectadas. No cadastro apenas uma face é aceita. Erros de câmera exibem mensagem.

**Verificação:** vídeo ao vivo com zero, uma e duas pessoas consentidas.
