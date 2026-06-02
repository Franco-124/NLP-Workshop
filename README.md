# Taller de NLP: Clasificador Jerárquico de Tickets de Soporte

Este repositorio contiene una solución de grado de producción para afinar (**fine-tune**) un modelo de lenguaje Transformers (`distilbert-base-multilingual-cased`) mediante **Transfer Learning**, enfocado en la clasificación jerárquica y multi-tarea de tickets de soporte de TI en español.

A continuación, se presenta la guía paso a paso del taller y del funcionamiento interno de cada módulo de código.

---

## 🧠 1. Conceptos Clave del Taller

### A. Transfer Learning y Fine-Tuning
En lugar de entrenar una red neuronal desde cero (lo cual requeriría millones de textos y recursos de cómputo inalcanzables), utilizamos **Transfer Learning** (Aprendizaje por Transferencia):
1. **Pre-entrenamiento**: Partimos de un modelo base que ya sabe estructurar frases, sintaxis y semántica básica en español tras leer toda la Wikipedia.
2. **Fine-Tuning (Ajuste Fino)**: Ajustamos sutilmente las neuronas de ese modelo pre-entrenado y entrenamos cabezales nuevos con nuestro dataset especializado de 1000 tickets de soporte.

### B. ¿Qué es DistilBERT?
Usamos `distilbert-base-multilingual-cased`. Es una versión "destilada" (comprimida) del modelo **mBERT** de Google:
* Es un **40% más pequeño** en número de parámetros y un **60% más rápido** en inferencia.
* Retiene el **97% de la precisión** de mBERT.
* Es **multilingüe** (soporta 104 idiomas, ideal para procesar español e inglés técnico de IT) y **sensible a mayúsculas** (distingue nombres de marcas, servidores y siglas).

### C. Aprendizaje Multi-Tarea (Multi-Task Learning)
El problema es jerárquico. Un ticket pertenece a un **Servicio**, a una **Categoría** y a una **Subcategoría**. En lugar de entrenar 3 modelos distintos, entrenamos **un solo modelo** con **3 cabezales lineales de salida** independientes que comparten el mismo extractor de características (backbone).

La pérdida total del modelo es la suma ponderada de la pérdida de entropía cruzada de cada tarea:
$$\text{Loss}_{\text{total}} = 0.2 \cdot \text{Loss}_{\text{Servicio}} + 0.3 \cdot \text{Loss}_{\text{Categoria}} + 0.5 \cdot \text{Loss}_{\text{Subcategoria}}$$

---

## 📂 2. Explicación del Código Paso a Paso

El código está estructurado en el directorio `src/` bajo principios de separación de responsabilidades y tipado estricto:

```
TallerNLPAI/
│
├── src/
│   ├── config.py           # Configuración e Hiperparámetros
│   ├── model.py            # Arquitectura del modelo (PyTorch)
│   ├── utils.py            # Preparación de datos, Datasets y Métricas
│   ├── train.py            # Ciclo de entrenamiento y validación
│   ├── predict.py          # Interfaz de comandos (CLI) interactiva
│   ├── main.py             # Configuración del Servidor FastAPI
│   ├── api/routes.py       # Controladores de endpoints API
│   ├── schemas/ticket.py   # Validación de datos de entrada/salida
│   └── core/exceptions.py  # Excepciones personalizadas
│
├── tests/                  # Pruebas unitarias con mocks
├── Dockerfile              # Empaquetado Docker optimizado
└── docker-compose.yml      # Levantamiento de contenedores local
```

### Paso 1: Configuración Centralizada — `src/config.py`
Evita el hardcoding en el entrenamiento. Define la ruta de los datos, directorios de guardado, tasas de aprendizaje (`learning_rate=2e-5`), número de épocas (`num_epochs=4`), dimensiones de clases y lógica para alternar automáticamente el batch size si estamos en GPU (16) o CPU (4) para evitar errores de memoria (OOM).

### Paso 2: La Red Neuronal Multi-Cabezal — `src/model.py`
Define la clase `DistilBertMultiHead` utilizando PyTorch (`nn.Module`):
1. **Backbone**: Carga el extractor DistilBERT `AutoModel.from_pretrained(model_name)`.
2. **Capas Lineales**: Define tres capas `nn.Linear` que reciben la dimensión oculta de DistilBERT (768) y mapean a la cantidad de clases de cada jerarquía.
3. **Propagación hacia adelante (`forward`)**: Pasa los tokens por DistilBERT, extrae el vector representativo de la frase (token `[CLS]` en la posición 0), aplica Dropout para evitar sobreajuste, y pasa ese vector por los 3 cabezales retornando un diccionario con los logits correspondientes.

### Paso 3: Preparación del Dataset y Métricas — `src/utils.py`
Contiene la infraestructura de procesamiento de datos:
1. **Codificadores**: Traduce etiquetas string (ej: *"Cloud Infrastructure"*) a números identificadores y viceversa.
2. **Dataset de PyTorch (`TicketDataset`)**: Estructura los tensores de tokens de entrada (`input_ids`, `attention_mask`) y las etiquetas para poder ser iterados eficientemente por los cargadores de PyTorch (`DataLoader`).
3. **Métricas Macro**: Calcula métricas de evaluación balanceadas (F1-Score, Precision y Recall Macro) para medir el rendimiento de manera justa frente a clases muy desbalanceadas.

### Paso 4: El Pipeline de Entrenamiento — `src/train.py`
Es el orquestador principal:
1. Divide el dataset 80/20 de forma reproducible usando semillas aleatorias fijas.
2. Tokeniza los textos usando el tokenizador de Hugging Face.
3. En cada época, ejecuta el bucle de entrenamiento, calcula la pérdida ponderada de las tres tareas, aplica optimización mediante el algoritmo `AdamW`, e imprime la pérdida media.
4. Al final del entrenamiento, evalúa el rendimiento sobre el 20% de validación y exporta todos los artefactos (`metrics.json`, `loss_curves.png`, logs y los modelos entrenados).

### Paso 5: Exposición de Endpoints — `src/main.py` & `src/api/routes.py`
Monta un servidor de producción con **FastAPI**:
* Carga el modelo pesado una sola vez al arrancar (`lifespan`) para agilizar las respuestas.
* Expone un endpoint `/api/v1/classify` que recibe un JSON verificado con Pydantic, ejecuta inferencias asíncronas y retorna las tres predicciones jerárquicas en milisegundos.

---

## 🚀 3. Instrucciones de Ejecución

### Requisitos Previos
El proyecto requiere **Python 3.13** y utiliza **uv** para la gestión veloz de dependencias.

```bash
# Sincronizar el entorno e instalar dependencias
uv sync
```

### Ejecutar Entrenamiento
```bash
uv run python -m src.train
```

### Probar en Consola (Modo Interactivo)
```bash
uv run python -m src.predict
```

### Correr Pruebas Unitarias (Con Mocks de Inferencia)
```bash
uv run pytest
```

### Levantar API con Docker
```bash
# Construir la imagen
docker compose build

# Encender el servidor
docker compose up -d
```
*La documentación interactiva de la API estará en: http://localhost:8000/docs*
