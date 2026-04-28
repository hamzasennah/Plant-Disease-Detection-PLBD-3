---
title: AgriScan Plant Disease Detection
emoji: 🌿
colorFrom: green
colorTo: teal
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: true
---

# 🌿 AgriScan — Détection de Maladies Végétales par IA

Système de diagnostic automatique basé sur **EfficientNet-B0** (99.64%) et **ResNet-50** (99.42%).

## 🌱 15 classes — Tomate · Pomme de terre · Poivron

| Modèle | Test Accuracy |
|---|---|
| EfficientNet-B0 | **99.64%** 🏆 |
| ResNet-50 | 99.42% |

## 📁 Structure
```
├── app.py              ← Interface Gradio
├── requirements.txt
├── models/             ← Modèles .pth entraînés
└── notebook/           ← Notebook Colab
```
