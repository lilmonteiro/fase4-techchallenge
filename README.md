# Video Analysis Project

## Video do Projeto: https://youtu.be/QwQz5zwjO5A 
Este projeto realiza a análise de vídeo aplicando anotações e salvando o resultado em um novo arquivo.

## Configuração do Ambiente Python

### 1. Criar Ambiente Virtual
Recomenda-se criar um ambiente virtual para evitar conflitos com outras bibliotecas instaladas globalmente.

```bash
py -m venv venv
```

### 2. Ativar o Ambiente Virtual

- **Windows:**
  
  ```bash
  venv\Scripts\activate
  ```

- **Linux/Mac:**

  ```bash
  source venv/bin/activate
  ```

### 3. Instalar Dependências
As bibliotecas necessárias estão listadas no arquivo `requirements.txt`. Execute o comando abaixo para instalá-las:

```bash
pip install -r requirements.txt
```

### 4. Executar o Script Principal
Após configurar o ambiente e instalar as dependências, execute o script principal para iniciar o pipeline de análise de vídeo:

```bash
py VideoAnalysis.py
```

O script abrirá o vídeo, processará os frames, aplicará as anotações e salvará o resultado como `output_video.mp4`.

## Observações Importantes

- **CMake:**  
  Durante a instalação das dependências, a biblioteca `dlib` exige o CMake. Baixe e instale o CMake a partir de [cmake.org](https://cmake.org/). Após a instalação, reinicie sua máquina.

- **Visual Studio Build Tools:**  
  Também será necessário instalar o **Visual Studio Build Tools**. Selecione todas as opções relacionadas ao Python durante a instalação.

---

Siga estas etapas para garantir que o ambiente esteja corretamente configurado e o script funcione como esperado.
