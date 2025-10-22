# token-journey
Transformations level micro to the tokens throughout a transformer

# 🔍 Análisis Paso a Paso de la Arquitectura Aitana-2B-S

## 📋 Descripción General

Este notebook realiza un **análisis exhaustivo y detallado** de la arquitectura interna del modelo de lenguaje **Aitana-2B-S** (modelo de 2 mil millones de parámetros desarrollado por GPLSI). El objetivo es **visualizar y comprender cada transformación matricial** que sufre un texto desde que entra como tokens hasta que se genera la predicción del siguiente token.

### ¿Qué hace este notebook?

**A nivel macro**, el notebook:
- Carga el modelo Aitana-2B-S desde HuggingFace
- Toma una frase de entrada en valenciano
- **Sigue paso a paso** todas las transformaciones matemáticas que ocurren internamente
- Muestra las dimensiones de cada matriz en cada etapa
- Visualiza los valores numéricos (primeros y últimos elementos) de cada transformación
- Culmina prediciendo el siguiente token con probabilidades

**A nivel micro**, desglosa:
1. **Tokenización**: Cómo se convierte el texto en IDs numéricos
2. **Embeddings**: Proyección de tokens a espacio vectorial (256K vocabulario → 2048 dimensiones)
3. **24 Capas Transformer**: Cada una con:
   - RMSNorm (normalización)
   - Multi-Head Attention (16 heads, 4 KV heads - Grouped Query Attention)
   - Residual connections
   - MLP con activación SwiGLU
4. **Proyección final**: De 2048 dimensiones → 256K vocabulario (logits)
5. **Softmax**: Conversión de logits a probabilidades
6. **Predicción**: Selección del siguiente token

---

## 🏗️ Arquitectura del Modelo Aitana-2B-S

### Especificaciones Técnicas

| Parámetro | Valor |
|-----------|-------|
| **Parámetros totales** | ~2 mil millones |
| **Vocabulario** | 256,000 tokens |
| **Dimensión oculta** | 2048 |
| **Número de capas** | 24 |
| **Attention heads** | 16 |
| **KV heads** | 4 (Grouped Query Attention) |
| **Dimensión intermedia (MLP)** | 5440 |
| **Embeddings rotacionales** | RoPE (theta=10000) |
| **Normalización** | RMSNorm |
| **Activación MLP** | SwiGLU |

# 🚀 Cómo Usar Este Notebook
El notebook ha sido creado en [Google Colab](https://colab.research.google.com/) y funciona bien siempre que tengas acceso a la TPU

### Personalización

**Cambiar texto de entrada:**
```python
text = "Tu propia frase en valenciano o castellano"
```

**Cambiar parámetros de generación:**
```python
generation = generator(
    input_text,
    do_sample=True,
    temperature=1.2,    # Más creativo
    top_k=50,
    top_p=0.95,
    max_new_tokens=100
)
```

---

## 📚 Conceptos Explicados

### RMSNorm
### Grouped Query Attention (GQA)
### RoPE (Rotary Position Embeddings)
### SwiGLU
### Residual Connections

---

## 🎯 Propósito Educativo

Este notebook es ideal para:
- ✅ **Estudiantes** aprendiendo arquitecturas transformer
- ✅ **Investigadores** que necesitan entender implementaciones específicas
- ✅ **Ingenieros ML** debuggeando modelos o implementando desde cero
- ✅ **Curiosos** que quieren ver "dentro" de un LLM

**Lo que aprenderás:**
1. Cómo fluyen los datos a través de un transformer
2. Dimensiones exactas de cada matriz
3. Por qué ciertos diseños (GQA, RoPE, SwiGLU) se eligen
4. Dónde está el costo computacional
5. Cómo se generan predicciones realmente

---

## ⚠️ Advertencias

1. **Requiere GPU con >8GB VRAM** para cargar el modelo completo
2. **No incluye optimizaciones** (KV-cache, quantization) - es didáctico
3. **Los valores mostrados** pueden cambiar ligeramente entre ejecuciones (inicialización de weights)
4. **No es código de producción** - prioriza claridad sobre eficiencia

---

## 📖 Referencias

- **Modelo:** [gplsi/Aitana-2B-S en HuggingFace](https://huggingface.co/gplsi/Aitana-2B-S)
- **Arquitectura base:** LLaMA 2 (Meta AI)

---

## 📝 Licencia

Este notebook es material educativo. Respeta las licencias del modelo Aitana-2B-S y las bibliotecas utilizadas.

---

## 🎉 Conclusión

Has completado un viaje desde **texto → tokens → embeddings → 24 capas transformer → logits → predicción**. ¡Ahora entiendes cada multiplicación matricial que ocurre en un LLM moderno!

**Siguiente paso:** Implementa tu propio transformer desde cero con estos conocimientos. 🚀
