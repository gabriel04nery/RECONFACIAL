# ADR 002 — Armazenamento

**Estado:** adotado em protótipo acadêmico.

**Decisão:** armazenar embeddings e nomes somente em `app/galeria.json` local, sem fotografias de cadastro. Solicitar autorização e permitir exclusão.

**Risco:** JSON em texto puro não é criptografia e deve ficar em dispositivo confiável com permissões restritas. Nunca publicar o banco nem backups; `.gitignore` não substitui inspeção do Git.
