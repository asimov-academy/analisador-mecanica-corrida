# Analisador da Mecânica de Corrida

Aplicação desktop em PyQt5 que utiliza OpenCV e Mediapipe para analisar biomecânica de corrida em tempo real ou a partir de vídeos. A interface permite escolher diferentes análises (oscilação corporal, postura lateral e passada) e alternar entre câmeras conectadas ou arquivos locais.

## Requisitos
- Python 3.12 (ou versão compatível listada em `pyproject.toml`)
- Sistema operacional Linux com suporte a Qt (testado no Pop!\_OS 22.04)
- Dependências do Qt disponíveis no sistema:
  ```bash
  sudo apt-get install \
    libxcb-xinerama0 libxcb-xinerama0-dev libxcb1 libxcb1-dev \
    libxkbcommon-x11-0 libxkbcommon-x11-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6
  ```

## Passo a passo

1. **Clonar o repositório**
   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd analisador-mecanica-corrida
   ```

2. **Criar e ativar um ambiente virtual**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   > Ajuste o comando caso utilize pyenv ou outra versão de Python.

3. **Instalar as dependências Python**
   ```bash
   pip install mediapipe==0.10.21 \
              opencv-python==4.11.0.86 \
              pyqt5==5.15.11 \
              pyqtgraph==0.13.7
   ```
   Esses valores seguem o `pyproject.toml`. Atualize-os se o arquivo mudar.

4. **Executar a aplicação**
   ```bash
   python mechanical/main.py
   ```

## Estrutura básica
- `mechanical/main.py`: interface principal (Qt) e orquestração das análises.
- `mechanical/analysis/`: classes específicas para cada análise (`OscillationAnalysis`, `PostureAnalysis`, `StrideAnalysis`).
- `models/pose_landmarker_full.task`: modelo Mediapipe utilizado nas análises.

## Uso
1. Escolha a câmera disponível ou carregue um vídeo (`Carregar Vídeo`).
2. Selecione o tipo de análise no combo box.
3. Clique em `Iniciar` para começar a captura; `Parar` encerra a sessão e libera a câmera.
4. Os resultados são exibidos no painel lateral direito conforme cada análise atualiza seus widgets.

## Solução de problemas
- **Qt não encontra o plugin `xcb`**: instale as bibliotecas listadas em requisitos e garanta que não existam variáveis `QT_QPA_PLATFORM_PLUGIN_PATH` conflitantes (o código já define o caminho padrão).
- **Erro ao acessar a câmera**: confirme se o dispositivo não está em uso por outro aplicativo e ajuste o índice no seletor de câmera.
- **Performance baixa**: reduza a resolução do vídeo ou use o modo de arquivo em vez da webcam para testes.

Com isso o projeto estará pronto para ser executado e ajustado conforme necessário.

