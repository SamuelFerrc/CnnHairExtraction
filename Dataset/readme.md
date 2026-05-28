\# Diretório de Datasets



Este diretório contém os datasets utilizados no projeto.



Os dados podem estar disponíveis em diferentes formatos, dependendo do volume e da estrutura necessária para armazenamento e processamento.



\## Estrutura dos Dados



Os datasets podem ser disponibilizados como:



\- Arquivos locais organizados em estrutura Lakehouse;

\- Links externos para repositórios de datasets;

\- Referências para bases massivas armazenadas em serviços de nuvem;

\- Conjuntos de dados particionados devido ao grande volume de informações.



\## Observações



\- Alguns datasets podem não estar armazenados diretamente neste diretório devido ao tamanho elevado dos arquivos;

\- Em casos de armazenamento externo, serão fornecidos links ou instruções de acesso;

\- Recomenda-se verificar os requisitos de armazenamento e processamento antes de realizar o download completo dos dados.



\## Organização Recomendada



```text

datasets/

├── raw/                # Dados brutos

├── processed/          # Dados processados

├── lakehouse/          # Estrutura Lakehouse

├── external\_links/     # Links e referências externas

└── README.md

```



\## Objetivo



A organização deste diretório visa facilitar:



\- Escalabilidade no armazenamento;

\- Processamento distribuído de grandes volumes de dados;

\- Versionamento e rastreabilidade dos datasets;

\- Integração com pipelines de Machine Learning e análise de dados.

